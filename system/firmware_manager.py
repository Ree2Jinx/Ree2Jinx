"""
ImaginaryConsole Emulator - Firmware Manager Module
Handles the loading and management of system firmware.
"""

import logging
import os
import hashlib
import json
from pathlib import Path
import struct


class FirmwareError(Exception):
    """Exception raised for firmware-related errors"""
    pass


class FirmwareManager:
    """Manages firmware loading and verification for the emulator"""
    
    # Firmware file structure - based on similar console formats
    FIRMWARE_FILES = {
        "BOOT0": {"required": True, "size_range": (4*1024*1024, 8*1024*1024)},
        "BOOT1": {"required": True, "size_range": (1*1024*1024, 4*1024*1024)},
        "BCPKG2": {"required": True, "size_range": (8*1024*1024, 32*1024*1024)},
        "SAFE_MODE": {"required": False, "size_range": (1*1024*1024, 4*1024*1024)},
        "SYSTEM_VERSION": {"required": True, "size_range": (1024, 32*1024)}
    }
    
    def __init__(self, firmware_path):
        """Initialize the firmware manager
        
        Args:
            firmware_path: Path to the firmware directory
        """
        self.logger = logging.getLogger("ImaginaryConsole.FirmwareManager")
        self.firmware_path = Path(firmware_path)
        
        # Create firmware directory if it doesn't exist
        self.firmware_path.mkdir(parents=True, exist_ok=True)
        
        # Firmware info
        self.firmware_loaded = False
        self.firmware_info = {}
        self.firmware_version = None
        self.firmware_data = {}
        
        self.logger.info(f"Firmware manager initialized with path: {self.firmware_path}")
    
    def verify_firmware_file(self, filename, data):
        """Verify a firmware file's integrity
        
        Args:
            filename: Name of the firmware file
            data: Binary content of the file
            
        Returns:
            True if the file is valid, False otherwise
        """
        # Check if this is a known firmware file
        if filename not in self.FIRMWARE_FILES:
            self.logger.warning(f"Unknown firmware file: {filename}")
            return False
        
        file_spec = self.FIRMWARE_FILES[filename]
        
        # Check file size
        min_size, max_size = file_spec["size_range"]
        if len(data) < min_size or len(data) > max_size:
            self.logger.error(f"Firmware file {filename} has invalid size: {len(data)} bytes "
                              f"(expected {min_size} to {max_size} bytes)")
            return False
        
        # Check file magic - specific to each firmware file type
        if filename == "BOOT0":
            # First 4 bytes should be a specific magic number
            magic = struct.unpack("<I", data[:4])[0]
            if magic != 0x43534642:  # 'BFSC' in little-endian
                self.logger.error(f"Firmware file {filename} has invalid magic: 0x{magic:08X}")
                return False
        
        elif filename == "BOOT1":
            # First 4 bytes should be a specific magic number
            magic = struct.unpack("<I", data[:4])[0]
            if magic != 0x30544F42:  # 'BOT0' in little-endian
                self.logger.error(f"Firmware file {filename} has invalid magic: 0x{magic:08X}")
                return False
        
        # Additional validation could be implemented for other firmware files
        
        return True
    
    def load_firmware_file(self, filepath):
        """Load a firmware file from the given path
        
        Args:
            filepath: Path to the firmware file
            
        Returns:
            Tuple of (filename, data) if successful, (None, None) otherwise
        """
        try:
            # Extract the base filename
            filename = os.path.basename(filepath)
            
            # Read the file
            with open(filepath, 'rb') as f:
                data = f.read()
            
            # Verify the file
            if self.verify_firmware_file(filename, data):
                self.logger.debug(f"Loaded firmware file: {filename} ({len(data)} bytes)")
                return filename, data
            else:
                self.logger.warning(f"Invalid firmware file: {filepath}")
                return None, None
            
        except Exception as e:
            self.logger.error(f"Failed to load firmware file {filepath}: {e}")
            return None, None
    
    def load_firmware(self):
        """Load and verify all firmware files from the firmware directory
        
        Returns:
            True if all required firmware files were loaded successfully, False otherwise
        """
        self.firmware_loaded = False
        self.firmware_data = {}
        self.firmware_info = {}
        
        # Find all firmware files
        firmware_files = {}
        for filename in self.FIRMWARE_FILES:
            filepath = self.firmware_path / filename
            if filepath.exists():
                firmware_files[filename] = str(filepath)
        
        # Check for missing required files
        missing_files = []
        for filename, file_spec in self.FIRMWARE_FILES.items():
            if file_spec["required"] and filename not in firmware_files:
                missing_files.append(filename)
        
        if missing_files:
            missing_list = ", ".join(missing_files)
            self.logger.error(f"Missing required firmware files: {missing_list}")
            return False
        
        # Load all available firmware files
        for filename, filepath in firmware_files.items():
            name, data = self.load_firmware_file(filepath)
            if name and data:
                self.firmware_data[name] = data
        
        # Check if we have all required files
        for filename, file_spec in self.FIRMWARE_FILES.items():
            if file_spec["required"] and filename not in self.firmware_data:
                self.logger.error(f"Required firmware file failed to load: {filename}")
                return False
        
        # Parse firmware version from SYSTEM_VERSION file
        if "SYSTEM_VERSION" in self.firmware_data:
            try:
                version_data = self.firmware_data["SYSTEM_VERSION"]
                self.firmware_version = version_data.decode('utf-8').strip()
                self.firmware_info["version"] = self.firmware_version
                self.logger.info(f"Firmware version: {self.firmware_version}")
            except Exception as e:
                self.logger.warning(f"Failed to parse firmware version: {e}")
        
        # Add file checksums to firmware info
        checksums = {}
        for filename, data in self.firmware_data.items():
            checksums[filename] = hashlib.sha256(data).hexdigest()
        self.firmware_info["checksums"] = checksums
        
        # Save firmware info
        self._save_firmware_info()
        
        self.firmware_loaded = True
        self.logger.info(f"Firmware loaded successfully: {len(self.firmware_data)} files")
        return True
    
    def _save_firmware_info(self):
        """Save firmware information to a JSON file"""
        info_path = self.firmware_path / "firmware_info.json"
        try:
            with open(info_path, 'w') as f:
                json.dump(self.firmware_info, f, indent=4)
        except Exception as e:
            self.logger.warning(f"Failed to save firmware info: {e}")
    
    def get_firmware_data(self, filename):
        """Get the data for a specific firmware file
        
        Args:
            filename: Name of the firmware file
            
        Returns:
            Binary data of the firmware file, or None if not found
        """
        return self.firmware_data.get(filename)
    
    def get_firmware_version(self):
        """Get the firmware version string
        
        Returns:
            Firmware version string, or None if not available
        """
        return self.firmware_version
    
    def is_firmware_loaded(self):
        """Check if firmware is loaded
        
        Returns:
            True if firmware is loaded, False otherwise
        """
        return self.firmware_loaded 