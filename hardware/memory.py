"""
ImaginaryConsole Emulator - Memory Module
Emulates the 12GB LPDDR5 memory subsystem of the imaginary console.
"""

import logging
import numpy as np
import time
from enum import Enum


class MemoryAccessType(Enum):
    """Types of memory access operations"""
    READ = 0
    WRITE = 1
    DMA = 2


class MemoryRegion:
    """Represents a region of allocated memory"""
    
    def __init__(self, address, size, name="Unknown"):
        """Initialize a memory region"""
        self.address = address
        self.size = size
        self.name = name
        self.allocated_time = time.time()
        self.last_access_time = self.allocated_time
        self.access_count = 0
    
    def contains_address(self, address):
        """Check if this region contains the given address"""
        return self.address <= address < (self.address + self.size)
    
    def record_access(self):
        """Record an access to this memory region"""
        self.last_access_time = time.time()
        self.access_count += 1


class Memory:
    """Emulates the LPDDR5 memory subsystem of the imaginary console"""
    
    def __init__(self, size=12*1024*1024*1024, frequency=4266, bandwidth=68.256):
        """Initialize the memory subsystem with the given specifications
        
        Args:
            size: Memory size in bytes (default 12GB)
            frequency: Memory frequency in MHz
            bandwidth: Memory bandwidth in GB/s
        """
        self.logger = logging.getLogger("ImaginaryConsole.Memory")
        
        self.size = size
        self.frequency = frequency  # MHz
        self.bandwidth = bandwidth  # GB/s
        
        # Create memory data array (using a bytearray for memory efficiency)
        # Note: In a real emulator with larger RAM size, we would use memory-mapped files or other approaches
        try:
            # Allocate memory
            self.data = bytearray(size)
            self.logger.info(f"Allocated {size/(1024*1024*1024):.2f} GB of memory")
        except MemoryError:
            # Fallback to a smaller size if we can't allocate the full amount
            fallback_size = 1024 * 1024 * 1024  # 1GB
            self.data = bytearray(fallback_size)
            self.size = fallback_size
            self.logger.warning(f"Failed to allocate full memory size, falling back to {fallback_size/(1024*1024*1024):.2f} GB")
        
        # Memory allocation tracking (address -> MemoryRegion)
        self.allocations = {}
        self.next_address = 0x10000  # Start allocations after a small offset
        
        # Memory access statistics
        self.read_count = 0
        self.write_count = 0
        self.dma_count = 0
        
        # Performance tracking
        self.access_times = []  # List of (timestamp, amount) tuples
        
        self.logger.info(f"Initialized memory: {frequency} MHz, {bandwidth} GB/s bandwidth")
    
    def set_frequency(self, frequency, bandwidth):
        """Update the memory frequency and bandwidth"""
        self.frequency = frequency
        self.bandwidth = bandwidth
        self.logger.info(f"Memory settings updated: {frequency} MHz, {bandwidth} GB/s bandwidth")
    
    def allocate(self, size, alignment=8, name="Unknown"):
        """Allocate a region of memory with the given size and alignment
        
        Args:
            size: Size of the memory region to allocate in bytes
            alignment: Alignment requirement in bytes (default 8)
            name: Name of the allocation for tracking purposes
            
        Returns:
            Start address of the allocated region
        """
        # Align the address to the requested alignment
        address = (self.next_address + alignment - 1) & ~(alignment - 1)
        
        # Check if we have enough memory
        if address + size > self.size:
            self.logger.error(f"Memory allocation failed: requested {size} bytes, but only {self.size-address} available")
            return 0
        
        # Create and store the allocation record
        region = MemoryRegion(address, size, name)
        self.allocations[address] = region
        
        # Update next allocation address
        self.next_address = address + size
        
        self.logger.debug(f"Allocated {size} bytes at address 0x{address:08X} for {name}")
        return address
    
    def free(self, address):
        """Free a previously allocated memory region
        
        Args:
            address: The starting address of the region to free
            
        Returns:
            True if the region was freed, False otherwise
        """
        if address in self.allocations:
            region = self.allocations.pop(address)
            self.logger.debug(f"Freed {region.size} bytes at address 0x{address:08X} ({region.name})")
            
            # In a real memory allocator, we would handle fragmentation and reuse
            # For simplicity, we don't reuse freed memory in this emulator
            
            return True
        else:
            self.logger.warning(f"Attempted to free unallocated memory at 0x{address:08X}")
            return False
    
    def read(self, address, size):
        """Read data from memory
        
        Args:
            address: Memory address to read from
            size: Number of bytes to read
            
        Returns:
            Bytes read from memory
        """
        # Check if the address is valid
        if address < 0 or address + size > self.size:
            self.logger.error(f"Memory read out of bounds: 0x{address:08X} + {size} bytes")
            return bytearray(size)  # Return zeroed buffer on error
        
        # Record the access
        self._record_access(MemoryAccessType.READ, size)
        
        # Find and update the associated memory region
        for region_addr, region in self.allocations.items():
            if region.contains_address(address):
                region.record_access()
                break
        
        # Return the memory data
        return bytes(self.data[address:address+size])
    
    def write(self, address, data):
        """Write data to memory
        
        Args:
            address: Memory address to write to
            data: Bytes to write
            
        Returns:
            Number of bytes written
        """
        size = len(data)
        
        # Check if the address is valid
        if address < 0 or address + size > self.size:
            self.logger.error(f"Memory write out of bounds: 0x{address:08X} + {size} bytes")
            return 0
        
        # Record the access
        self._record_access(MemoryAccessType.WRITE, size)
        
        # Find and update the associated memory region
        for region_addr, region in self.allocations.items():
            if region.contains_address(address):
                region.record_access()
                break
        
        # Write the data
        self.data[address:address+size] = data
        return size
    
    def dma_transfer(self, src_address, dst_address, size):
        """Perform a DMA transfer between two memory regions
        
        Args:
            src_address: Source memory address
            dst_address: Destination memory address
            size: Number of bytes to transfer
            
        Returns:
            Number of bytes transferred
        """
        # Check if the addresses are valid
        if (src_address < 0 or src_address + size > self.size or
            dst_address < 0 or dst_address + size > self.size):
            self.logger.error(f"DMA transfer out of bounds: 0x{src_address:08X} -> 0x{dst_address:08X}, {size} bytes")
            return 0
        
        # Record the access
        self._record_access(MemoryAccessType.DMA, size)
        
        # Copy the data
        self.data[dst_address:dst_address+size] = self.data[src_address:src_address+size]
        return size
    
    def _record_access(self, access_type, size):
        """Record a memory access for performance tracking"""
        now = time.time()
        
        # Add access to performance tracking
        self.access_times.append((now, size))
        
        # Trim old entries (keep last 1000)
        if len(self.access_times) > 1000:
            self.access_times.pop(0)
        
        # Update counters
        if access_type == MemoryAccessType.READ:
            self.read_count += 1
        elif access_type == MemoryAccessType.WRITE:
            self.write_count += 1
        elif access_type == MemoryAccessType.DMA:
            self.dma_count += 1
    
    def get_bandwidth_usage(self, window_seconds=1.0):
        """Calculate current memory bandwidth usage over the specified time window
        
        Args:
            window_seconds: Time window in seconds to calculate bandwidth usage
            
        Returns:
            Current bandwidth usage in GB/s
        """
        now = time.time()
        window_start = now - window_seconds
        
        # Filter access entries in the time window
        window_accesses = [(t, s) for t, s in self.access_times if t >= window_start]
        
        # Calculate total bytes accessed in the window
        total_bytes = sum(size for _, size in window_accesses)
        
        # Calculate bandwidth in GB/s
        if window_accesses:
            oldest_time = min(t for t, _ in window_accesses)
            time_span = now - oldest_time
            if time_span > 0:
                bandwidth_gb_s = (total_bytes / (1024*1024*1024)) / time_span
                return bandwidth_gb_s
        
        return 0.0
    
    def get_allocation_stats(self):
        """Get statistics about memory allocations"""
        total_allocated = sum(region.size for region in self.allocations.values())
        allocation_count = len(self.allocations)
        
        return {
            "total_allocated": total_allocated,
            "allocation_count": allocation_count,
            "free_memory": self.size - total_allocated,
            "usage_percent": (total_allocated / self.size) * 100
        }
    
    def get_access_stats(self):
        """Get statistics about memory accesses"""
        return {
            "read_count": self.read_count,
            "write_count": self.write_count,
            "dma_count": self.dma_count,
            "total_accesses": self.read_count + self.write_count + self.dma_count,
            "current_bandwidth": self.get_bandwidth_usage()
        }
    
    def get_state(self):
        """Get the current state of the memory subsystem (excluding the actual data)"""
        return {
            "size": self.size,
            "frequency": self.frequency,
            "bandwidth": self.bandwidth,
            "allocations": len(self.allocations),
            "next_address": self.next_address,
            "stats": {
                "allocation": self.get_allocation_stats(),
                "access": self.get_access_stats()
            }
        } 