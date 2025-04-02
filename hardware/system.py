"""
ImaginaryConsole Emulator - Hardware System Module
This module emulates the complete hardware system of the imaginary console,
including CPU, GPU, memory, and storage subsystems.
"""

import logging
import time
from pathlib import Path

from hardware.cpu import ArmCpu
from hardware.gpu import NvidiaGpu
from hardware.memory import Memory
from hardware.storage import Storage


class System:
    """Main hardware system class that coordinates all hardware components"""
    
    def __init__(self, config):
        """Initialize the hardware system with the given configuration"""
        self.logger = logging.getLogger("ImaginaryConsole.System")
        self.config = config
        
        self.logger.info("Initializing hardware components")
        
        # Initialize hardware components
        self.cpu = ArmCpu(
            cores=config.HARDWARE_SPECS["cpu"]["cores"],
            l3_cache=config.HARDWARE_SPECS["cpu"]["l3_cache"],
            architecture=config.HARDWARE_SPECS["cpu"]["architecture"],
            version=config.HARDWARE_SPECS["cpu"]["version"],
            frequency=config.get_cpu_frequency()
        )
        
        self.gpu = NvidiaGpu(
            cuda_cores=config.HARDWARE_SPECS["gpu"]["cuda_cores"],
            sms=config.HARDWARE_SPECS["gpu"]["streaming_multiprocessors"],
            clusters=config.HARDWARE_SPECS["gpu"]["graphics_clusters"],
            frequency=config.get_gpu_frequency()
        )
        
        self.memory = Memory(
            size=config.HARDWARE_SPECS["memory"]["size"] * 1024 * 1024 * 1024,  # Convert GB to bytes
            frequency=config.get_memory_frequency(),
            bandwidth=config.get_memory_bandwidth()
        )
        
        self.storage = Storage(
            size=config.HARDWARE_SPECS["storage"]["size"] * 1024 * 1024 * 1024  # Convert GB to bytes
        )
        
        # Connect components
        self.cpu.set_memory(self.memory)
        self.gpu.set_memory(self.memory)
        
        # System memory map
        self.memory_map = {
            'boot_rom': None,
            'ram_start': 0x10000,  # Start of regular RAM
            'mmio_start': 0xE0000000,  # Start of memory-mapped I/O
            'gpu_registers': 0xE0100000,
            'audio_registers': 0xE0200000,
            'usb_registers': 0xE0300000,
            'system_registers': 0xE0400000,
            'rom_base': None
        }
        
        # Allocate GPU memory regions in system memory
        self._allocate_gpu_memory()
        
        # Runtime tracking
        self.running = False
        self.pause = False
        self.fps = 60
        self.frame_time = 1.0 / self.fps
        self.last_frame_time = 0
        
        self.rom_loaded = False
        self.rom_info = None
        
        self.logger.info("Hardware system initialized")
    
    def _allocate_gpu_memory(self):
        """Allocate GPU memory regions in system memory"""
        # Allocate memory for framebuffer - assuming 1080p RGBA (4 bytes per pixel)
        framebuffer_size = 1920 * 1080 * 4
        framebuffer_addr = self.memory.allocate(framebuffer_size, name="GPU_Framebuffer")
        
        # Allocate memory for GPU command buffers
        cmd_buffer_size = 1024 * 1024  # 1MB command buffer
        cmd_buffer_addr = self.memory.allocate(cmd_buffer_size, name="GPU_CommandBuffer")
        
        # Allocate memory for shader code
        shader_buffer_size = 4 * 1024 * 1024  # 4MB shader buffer
        shader_buffer_addr = self.memory.allocate(shader_buffer_size, name="GPU_ShaderBuffer")
        
        # Allocate memory for textures
        texture_buffer_size = 128 * 1024 * 1024  # 128MB texture buffer
        texture_buffer_addr = self.memory.allocate(texture_buffer_size, name="GPU_TextureBuffer")
        
        # Add to memory map
        self.memory_map.update({
            'gpu_framebuffer': framebuffer_addr,
            'gpu_cmd_buffer': cmd_buffer_addr,
            'gpu_shader_buffer': shader_buffer_addr,
            'gpu_texture_buffer': texture_buffer_addr
        })
        
        self.logger.info(f"Allocated GPU memory regions: "
                         f"Framebuffer at 0x{framebuffer_addr:08X}, "
                         f"Command buffer at 0x{cmd_buffer_addr:08X}, "
                         f"Shader buffer at 0x{shader_buffer_addr:08X}, "
                         f"Texture buffer at 0x{texture_buffer_addr:08X}")
    
    def update_mode(self):
        """Update hardware frequencies based on current docked/handheld mode"""
        self.cpu.set_frequency(self.config.get_cpu_frequency())
        self.gpu.set_frequency(self.config.get_gpu_frequency())
        self.memory.set_frequency(self.config.get_memory_frequency(), self.config.get_memory_bandwidth())
        
        # Set GPU framebuffer size based on mode
        width = 1920 if self.config.docked_mode else 1280
        height = 1080 if self.config.docked_mode else 720
        self.gpu.set_framebuffer_size(width, height)
        
        self.logger.info(f"Hardware frequencies updated for {'docked' if self.config.docked_mode else 'handheld'} mode")
    
    def load_bios(self, firmware_manager):
        """Load system BIOS and boot code
        
        Args:
            firmware_manager: Firmware manager containing the firmware files
            
        Returns:
            True if BIOS was loaded, False otherwise
        """
        if not firmware_manager.is_firmware_loaded():
            self.logger.error("Cannot load BIOS: Firmware not loaded")
            return False
        
        try:
            # Load BOOT0 (primary BIOS)
            boot0_data = firmware_manager.get_firmware_data("BOOT0")
            if not boot0_data:
                self.logger.error("BOOT0 firmware not found")
                return False
            
            # Allocate memory for BIOS and load it
            boot0_addr = self.memory.allocate(len(boot0_data), name="BOOT0")
            self.memory.write(boot0_addr, boot0_data)
            self.memory_map['boot_rom'] = boot0_addr
            
            # Set CPU to start executing from BIOS
            self.cpu.set_program_counter(boot0_addr)
            
            self.logger.info(f"BIOS loaded successfully at 0x{boot0_addr:08X}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load BIOS: {e}")
            return False
    
    def load_rom(self, rom_loader):
        """Load a ROM into the system memory"""
        try:
            self.logger.info(f"Loading ROM: {rom_loader.rom_path}")
            rom_data, rom_info = rom_loader.load()
            
            # Store ROM information
            self.rom_info = rom_info
            
            # Allocate memory for the ROM
            rom_addr = self.memory.allocate(len(rom_data), name=f"ROM_{rom_info['title']}")
            self.memory.write(rom_addr, rom_data)
            self.memory_map['rom_base'] = rom_addr
            
            # Store ROM address for CPU/GPU access
            if self.memory_map['boot_rom'] is None:
                # If BIOS isn't loaded, directly start executing the ROM
                self.cpu.set_program_counter(rom_addr)
            
            # Initialize ROM-specific memory regions
            self._init_rom_memory(rom_info)
            
            self.rom_loaded = True
            self.logger.info(f"ROM loaded successfully: {rom_info['title']} at 0x{rom_addr:08X}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load ROM: {e}")
            return False
    
    def _init_rom_memory(self, rom_info):
        """Initialize memory regions specific to the loaded ROM
        
        Args:
            rom_info: Information about the loaded ROM
        """
        # Allocate save data region
        save_size = 512 * 1024  # Default 512KB
        save_addr = self.memory.allocate(save_size, name=f"SaveData_{rom_info['title']}")
        
        # Allocate game-specific work RAM
        work_ram_size = 64 * 1024 * 1024  # 64MB work RAM
        work_ram_addr = self.memory.allocate(work_ram_size, name=f"WorkRAM_{rom_info['title']}")
        
        # Update memory map
        self.memory_map.update({
            'save_data': save_addr,
            'work_ram': work_ram_addr
        })
        
        self.logger.info(f"Allocated ROM-specific memory: "
                         f"Save data at 0x{save_addr:08X}, "
                         f"Work RAM at 0x{work_ram_addr:08X}")
    
    def start(self):
        """Start the system execution"""
        if not self.rom_loaded and self.memory_map['boot_rom'] is None:
            self.logger.error("Cannot start system: No ROM or BIOS loaded")
            return False
        
        self.running = True
        self.pause = False
        self.last_frame_time = time.time()
        
        # Start GPU rendering
        self.gpu.start_rendering()
        
        self.logger.info("System execution started")
        return True
    
    def stop(self):
        """Stop the system execution"""
        self.running = False
        
        # Stop GPU rendering
        self.gpu.stop_rendering()
        
        self.logger.info("System execution stopped")
    
    def toggle_pause(self):
        """Toggle pause state"""
        self.pause = not self.pause
        
        # Update GPU mode based on pause state
        if self.pause:
            self.gpu.set_mode(self.gpu.mode.IDLE)
        else:
            self.gpu.start_rendering()
            
        self.logger.info(f"System {'paused' if self.pause else 'resumed'}")
        return self.pause
    
    def step(self):
        """Execute a single frame of system processing"""
        if not self.running or self.pause:
            return False
        
        # Calculate delta time
        current_time = time.time()
        delta = current_time - self.last_frame_time
        
        # Execute hardware components for one frame
        self.cpu.execute_cycle(delta)
        self.gpu.execute_cycle(delta)
        
        # Draw a test pattern if no ROM is loaded
        if not self.rom_loaded and self.running:
            self._draw_test_pattern(int(self.frame_count * 4) % 256)
        
        # Check for any DMA transfers or interrupts
        self._check_interrupts()
        
        # Cap to desired frame rate
        elapsed = time.time() - current_time
        if elapsed < self.frame_time:
            time.sleep(self.frame_time - elapsed)
        
        self.last_frame_time = current_time
        self.frame_count += 1
        return True
    
    def _draw_test_pattern(self, offset=0):
        """Draw a test pattern to the framebuffer when no ROM is loaded"""
        # Draw color gradient
        for y in range(0, self.gpu.framebuffer_height, 4):
            for x in range(0, self.gpu.framebuffer_width, 4):
                r = (x + offset) % 256
                g = (y + offset) % 256
                b = (x + y + offset) % 256
                color = (r, g, b, 255)
                
                self.gpu.draw_rect(x, y, 4, 4, color)
    
    def _check_interrupts(self):
        """Check for any pending interrupts and trigger them"""
        # Simple implementation - just check for timer interrupts
        # In a real system, we would check all interrupt sources (I/O, timers, etc.)
        pass
    
    def save_state(self, path):
        """Save the current system state to a file"""
        save_path = Path(path)
        try:
            # Create state dictionary with all component states
            state = {
                "cpu": self.cpu.get_state(),
                "memory": self.memory.get_state(),
                "gpu": self.gpu.get_state(),
                "rom_info": self.rom_info,
                "memory_map": self.memory_map,
                "config": {
                    "docked_mode": self.config.docked_mode
                }
            }
            
            # Actual saving would involve serializing this state
            # For a full implementation, we would save to a compressed file
            # Here we'll just log it
            self.logger.info(f"System state saved to {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save system state: {e}")
            return False
    
    def load_state(self, path):
        """Load a system state from a file"""
        load_path = Path(path)
        try:
            # Actual loading would involve deserializing from a file
            # For a full implementation, we would load from a compressed file
            # Here we'll just log it
            
            # For illustration, we would restore component states:
            # self.cpu.set_state(state["cpu"])
            # self.memory.set_state(state["memory"])
            # self.gpu.set_state(state["gpu"])
            # self.rom_info = state["rom_info"]
            # self.memory_map = state["memory_map"]
            
            # Update config if it changed in the save state
            # if "config" in state and state["config"]["docked_mode"] != self.config.docked_mode:
            #     self.config.set_docked_mode(state["config"]["docked_mode"])
            #     self.update_mode()
            
            self.logger.info(f"System state loaded from {load_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load system state: {e}")
            return False
    
    def get_framebuffer_data(self):
        """Get the current framebuffer data from the GPU
        
        Returns:
            Bytes object containing framebuffer data
        """
        return self.gpu.get_framebuffer_data()
    
    def get_system_stats(self):
        """Get comprehensive system statistics
        
        Returns:
            Dictionary with system statistics
        """
        return {
            "cpu": {
                "frequency": self.cpu.frequency,
                "performance": self.cpu.get_performance(),
                "cores": len(self.cpu.cores)
            },
            "gpu": {
                "frequency": self.gpu.frequency,
                "tflops": self.gpu.calculate_tflops(),
                "frame_rate": 1000 / self.gpu.frame_time_ms if self.gpu.frame_time_ms > 0 else 0
            },
            "memory": {
                "frequency": self.memory.frequency,
                "bandwidth": self.memory.bandwidth,
                "usage": self.memory.get_allocation_stats()
            },
            "storage": self.storage.get_usage_stats(),
            "mode": "docked" if self.config.docked_mode else "handheld"
        } 