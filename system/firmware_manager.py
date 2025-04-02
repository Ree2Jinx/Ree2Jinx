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
    
    # NCA types and their properties
    NCA_TYPES = {
        # Each NCA type with its properties
        "PROGRAM": {"required": True, "size_range": (4*1024*1024, 64*1024*1024)},
        "CONTROL": {"required": True, "size_range": (1*1024*1024, 8*1024*1024)},
        "DATA": {"required": True, "size_range": (1*1024*1024, 32*1024*1024)},
        "META": {"required": True, "size_range": (1*1024, 1*1024*1024)},
        "PUBLIC_DATA": {"required": False, "size_range": (1*1024, 8*1024*1024)},
    }
    
    # NCA magic header
    NCA_MAGIC = b'NCA\x00'
    
    def __init__(self, firmware_path):
        """Initialize the firmware manager
        
        Args:
            firmware_path: Path to the firmware directory containing .nca files
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
        self.nca_files = {}
        
        self.logger.info(f"Firmware manager initialized with path: {self.firmware_path}")
    
    def verify_firmware_file(self, filepath, data):
        """Verify an NCA firmware file's integrity
        
        Args:
            filepath: Path to the NCA file
            data: Binary content of the file
            
        Returns:
            Tuple of (nca_type, is_valid) where nca_type is the determined type and is_valid is a boolean
        """
        # Check file size - general minimum
        if len(data) < 1*1024*1024:  # Less than 1MB is likely invalid
            self.logger.error(f"NCA file {filepath.name} is too small: {len(data)} bytes")
            return None, False
        
        # Check NCA magic in header
        if data[:4] != self.NCA_MAGIC:
            self.logger.error(f"NCA file {filepath.name} has invalid magic: {data[:4]}")
            return None, False
        
        # Extract content type from NCA header (simplified - real implementation would parse the full header)
        # In a real implementation, we'd parse the proper NCA header structure
        try:
            # Content type is typically at offset 0x0C in the NCA header
            content_type = data[0x0C]
            
            # Map content type to NCA type string
            if content_type == 0:
                nca_type = "PROGRAM"
            elif content_type == 1:
                nca_type = "META"
            elif content_type == 2:
                nca_type = "CONTROL"
            elif content_type == 3:
                nca_type = "DATA"
            elif content_type == 4:
                nca_type = "PUBLIC_DATA"
            else:
                self.logger.warning(f"Unknown NCA content type: {content_type} in {filepath.name}")
                nca_type = "UNKNOWN"
                
            # Verify size range if this is a known type
            if nca_type in self.NCA_TYPES:
                type_spec = self.NCA_TYPES[nca_type]
                min_size, max_size = type_spec["size_range"]
                
                if len(data) < min_size or len(data) > max_size:
                    self.logger.warning(f"NCA file {filepath.name} of type {nca_type} has unusual size: {len(data)} bytes "
                                      f"(expected {min_size} to {max_size} bytes)")
                    # Continue anyway as size ranges are approximate
            
            return nca_type, True
            
        except Exception as e:
            self.logger.error(f"Error parsing NCA header for {filepath.name}: {e}")
            return None, False
    
    def load_firmware_file(self, filepath):
        """Load an NCA firmware file from the given path
        
        Args:
            filepath: Path to the NCA firmware file
            
        Returns:
            Tuple of (nca_id, nca_type, data) if successful, (None, None, None) otherwise
        """
        try:
            # Generate a unique ID for this NCA file
            nca_id = filepath.stem
            
            # Read the file
            with open(filepath, 'rb') as f:
                data = f.read()
            
            # Verify the file and determine its type
            nca_type, valid = self.verify_firmware_file(filepath, data)
            
            if valid:
                self.logger.debug(f"Loaded NCA firmware file: {filepath.name} ({len(data)} bytes), type: {nca_type}")
                return nca_id, nca_type, data
            else:
                self.logger.warning(f"Invalid NCA firmware file: {filepath}")
                return None, None, None
            
        except Exception as e:
            self.logger.error(f"Failed to load NCA firmware file {filepath}: {e}")
            return None, None, None
    
    def load_firmware(self):
        """Load and verify all NCA firmware files from the firmware directory
        
        Returns:
            True if all required NCA types were loaded successfully, False otherwise
        """
        self.firmware_loaded = False
        self.firmware_data = {}
        self.firmware_info = {}
        self.nca_files = {}
        
        # Find all .nca files in the firmware directory
        nca_files = list(self.firmware_path.glob("*.nca"))
        
        if not nca_files:
            self.logger.error(f"No .nca files found in firmware directory: {self.firmware_path}")
            return False
        
        self.logger.info(f"Found {len(nca_files)} .nca files in firmware directory")
        
        # Load all available NCA files
        loaded_types = set()
        for filepath in nca_files:
            nca_id, nca_type, data = self.load_firmware_file(filepath)
            if nca_id and nca_type and data:
                # Store the NCA file data
                self.nca_files[nca_id] = {
                    "type": nca_type,
                    "data": data,
                    "path": str(filepath),
                    "size": len(data)
                }
                
                # Keep track of loaded NCA types
                loaded_types.add(nca_type)
        
        # Check if we have all required NCA types
        missing_types = []
        for nca_type, type_spec in self.NCA_TYPES.items():
            if type_spec["required"] and nca_type not in loaded_types:
                missing_types.append(nca_type)
        
        if missing_types:
            missing_list = ", ".join(missing_types)
            self.logger.error(f"Missing required NCA firmware types: {missing_list}")
            return False
        
        # Extract firmware version from META NCA if available
        meta_ncas = [nca for nca in self.nca_files.values() if nca["type"] == "META"]
        if meta_ncas:
            # In a real implementation, we'd parse the META NCA properly to extract version
            # This is a simplified version for demonstration purposes
            try:
                meta_data = meta_ncas[0]["data"]
                # Assume version string is stored at a specific offset in the META NCA
                version_offset = 0x100  # This would be different in a real implementation
                version_length = 32
                version_bytes = meta_data[version_offset:version_offset+version_length]
                
                # Extract null-terminated string
                null_pos = version_bytes.find(b'\x00')
                if null_pos > 0:
                    version_bytes = version_bytes[:null_pos]
                
                self.firmware_version = version_bytes.decode('utf-8', errors='ignore').strip()
                self.firmware_info["version"] = self.firmware_version
                self.logger.info(f"Firmware version: {self.firmware_version}")
            except Exception as e:
                self.logger.warning(f"Failed to parse firmware version from META NCA: {e}")
        
        # Add file checksums to firmware info
        checksums = {}
        nca_info = {}
        for nca_id, nca in self.nca_files.items():
            checksums[nca_id] = hashlib.sha256(nca["data"]).hexdigest()
            nca_info[nca_id] = {
                "type": nca["type"],
                "size": nca["size"],
                "path": nca["path"]
            }
        
        self.firmware_info["checksums"] = checksums
        self.firmware_info["nca_files"] = nca_info
        
        # Save firmware info
        self._save_firmware_info()
        
        self.firmware_loaded = True
        self.logger.info(f"Firmware loaded successfully: {len(self.nca_files)} NCA files")
        return True
    
    def _save_firmware_info(self):
        """Save firmware information to a JSON file"""
        info_path = self.firmware_path / "firmware_info.json"
        try:
            # Copy only serializable data (remove actual binary data)
            serializable_info = {k: v for k, v in self.firmware_info.items()}
            
            with open(info_path, 'w') as f:
                json.dump(serializable_info, f, indent=4)
        except Exception as e:
            self.logger.warning(f"Failed to save firmware info: {e}")
    
    def get_nca_data(self, nca_id):
        """Get the data for a specific NCA file by ID
        
        Args:
            nca_id: ID of the NCA file
            
        Returns:
            Binary data of the NCA file, or None if not found
        """
        nca = self.nca_files.get(nca_id)
        return nca["data"] if nca else None
    
    def get_nca_by_type(self, nca_type):
        """Get NCA files of a specific type
        
        Args:
            nca_type: Type of NCA to retrieve
            
        Returns:
            List of NCA IDs matching the requested type
        """
        return [nca_id for nca_id, nca in self.nca_files.items() if nca["type"] == nca_type]
    
    def get_firmware_data(self, filename):
        """Get the data for a specific firmware file (legacy method for compatibility)
        
        Args:
            filename: Name of the firmware file
            
        Returns:
            Binary data of the firmware file, or None if not found
        """
        # Map legacy firmware filenames to NCA types for backward compatibility
        legacy_map = {
            "BOOT0": "PROGRAM",
            "BOOT1": "CONTROL",
            "BCPKG2": "DATA",
            "SYSTEM_VERSION": "META"
        }
        
        if filename in legacy_map:
            nca_type = legacy_map[filename]
            nca_ids = self.get_nca_by_type(nca_type)
            if nca_ids:
                return self.get_nca_data(nca_ids[0])
        
        return None
    
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