"""
ImaginaryConsole Emulator - Configuration Module
Handles all configuration settings for the emulator, including hardware specs and operational modes.
"""

import os
import json
import logging
from pathlib import Path


class Config:
    """Manages the emulator configuration settings"""
    
    # Hardware specifications as defined for the imaginary console
    HARDWARE_SPECS = {
        "cpu": {
            "name": "ARM Cortex-A78c",
            "cores": 8,
            "architecture": "ARM64",
            "version": "ARMv8.2-A",
            "l3_cache": "8MB",
            "handheld_freq": 998.4,  # MHz
            "docked_freq": 1100.8    # MHz
        },
        "gpu": {
            "name": "Nvidia T239 Ampere",
            "graphics_clusters": 1,
            "streaming_multiprocessors": 12,
            "cuda_cores": 1534,
            "handheld_freq": 561.0,      # MHz
            "docked_freq": 1007.25,      # MHz
            "handheld_tflops": 1.72,
            "docked_tflops": 3.09
        },
        "memory": {
            "size": 12,  # GB
            "type": "LPDDR5",
            "handheld_freq": 4266,    # MHz
            "docked_freq": 6400,      # MHz
            "handheld_bandwidth": 68.256,  # GB/s
            "docked_bandwidth": 102.4     # GB/s
        },
        "storage": {
            "size": 256  # GB
        }
    }
    
    def __init__(self):
        """Initialize configuration with default values"""
        self.logger = logging.getLogger("ImaginaryConsole.Config")
        
        # Default paths
        self.config_dir = Path(os.path.expanduser("~/.imaginary_console"))
        self.config_file = self.config_dir / "config.json"
        self.roms_path = self.config_dir / "roms"
        self.firmware_path = self.config_dir / "firmware"
        self.keys_path = self.config_dir / "keys"  # Now points to the keys directory, not a specific file
        self.saves_path = self.config_dir / "saves"
        
        # Default settings
        self.docked_mode = False
        self.fullscreen = False
        self.window_size = (1280, 720)  # Default for handheld mode
        self.vsync = True
        self.controller_profile = "default"
        
        # Ensure config directory structure exists
        self._ensure_dirs_exist()
    
    def _ensure_dirs_exist(self):
        """Ensure all required directories exist"""
        for path in [self.config_dir, self.roms_path, self.firmware_path, 
                     self.keys_path.parent, self.saves_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def load_config(self):
        """Load configuration from file if it exists"""
        if not self.config_file.exists():
            self.logger.info("Config file not found, creating default configuration")
            self.save_config()
            return
        
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
                
            # Update settings from the loaded data
            self.docked_mode = config_data.get('docked_mode', False)
            self.fullscreen = config_data.get('fullscreen', False)
            self.window_size = tuple(config_data.get('window_size', (1280, 720)))
            self.vsync = config_data.get('vsync', True)
            self.controller_profile = config_data.get('controller_profile', 'default')
            
            # Update paths
            if 'paths' in config_data:
                paths_data = config_data['paths']
                self.roms_path = Path(paths_data.get('roms', str(self.roms_path)))
                self.firmware_path = Path(paths_data.get('firmware', str(self.firmware_path)))
                self.keys_path = Path(paths_data.get('keys', str(self.keys_path)))
                self.saves_path = Path(paths_data.get('saves', str(self.saves_path)))
            
            self.logger.info("Configuration loaded successfully")
        
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self.logger.info("Using default configuration")
    
    def save_config(self):
        """Save current configuration to file"""
        config_data = {
            'docked_mode': self.docked_mode,
            'fullscreen': self.fullscreen,
            'window_size': self.window_size,
            'vsync': self.vsync,
            'controller_profile': self.controller_profile,
            'paths': {
                'roms': str(self.roms_path),
                'firmware': str(self.firmware_path),
                'keys': str(self.keys_path),
                'saves': str(self.saves_path)
            }
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
            self.logger.info("Configuration saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
    
    def set_docked_mode(self, enable):
        """Toggle between docked and handheld mode"""
        self.docked_mode = enable
        # Update window size based on mode if not in fullscreen
        if not self.fullscreen:
            self.window_size = (1920, 1080) if enable else (1280, 720)
        
        self.logger.info(f"System mode switched to {'docked' if enable else 'handheld'}")
    
    def get_cpu_frequency(self):
        """Get current CPU frequency based on mode"""
        return self.HARDWARE_SPECS["cpu"]["docked_freq"] if self.docked_mode else self.HARDWARE_SPECS["cpu"]["handheld_freq"]
    
    def get_gpu_frequency(self):
        """Get current GPU frequency based on mode"""
        return self.HARDWARE_SPECS["gpu"]["docked_freq"] if self.docked_mode else self.HARDWARE_SPECS["gpu"]["handheld_freq"]
    
    def get_memory_frequency(self):
        """Get current memory frequency based on mode"""
        return self.HARDWARE_SPECS["memory"]["docked_freq"] if self.docked_mode else self.HARDWARE_SPECS["memory"]["handheld_freq"]
    
    def get_memory_bandwidth(self):
        """Get current memory bandwidth based on mode"""
        return self.HARDWARE_SPECS["memory"]["docked_bandwidth"] if self.docked_mode else self.HARDWARE_SPECS["memory"]["handheld_bandwidth"] 