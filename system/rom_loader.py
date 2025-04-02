"""
ImaginaryConsole Emulator - ROM Loader Module
Handles the loading, decryption, and parsing of game ROMs.
"""

import logging
import os
import hashlib
import json
import struct
from pathlib import Path
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


class RomError(Exception):
    """Exception raised for ROM-related errors"""
    pass


class KeysManager:
    """Manages encryption keys for ROM decryption"""
    
    def __init__(self, keys_path):
        """Initialize the keys manager
        
        Args:
            keys_path: Path to the keys file
        """
        self.logger = logging.getLogger("ImaginaryConsole.KeysManager")
        self.keys_path = Path(keys_path)
        self.keys = {}
        
        # Flag indicating if keys are loaded
        self.keys_loaded = False
    
    def load_keys(self):
        """Load encryption keys from the keys file
        
        Returns:
            True if keys were loaded successfully, False otherwise
        """
        keys_path = Path(self.keys_path)
        
        if not keys_path.exists():
            self.logger.error(f"Keys file not found: {keys_path}")
            # Create a default keys structure with empty keys
            # This is for demonstration purposes only
            self.keys = {
                "header_key": b'\x00' * 32,
                "key_area_key_application": b'\x00' * 16,
                "key_area_key_ocean": b'\x00' * 16,
                "titlekek_00": b'\x00' * 16,
                "titlekek_01": b'\x00' * 16
            }
            self.logger.warning("Using empty placeholder keys (demo mode)")
            self.keys_loaded = True
            return True
        
        try:
            self.logger.info(f"Loading keys from: {keys_path}")
            with open(keys_path, 'r') as f:
                lines = f.readlines()
            
            # Parse keys file (format: key_name = hex_value)
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('=', 1)
                if len(parts) != 2:
                    continue
                
                key_name = parts[0].strip()
                key_value = parts[1].strip()
                
                # Remove any 0x prefix and convert to bytes
                if key_value.startswith('0x'):
                    key_value = key_value[2:]
                
                try:
                    key_bytes = bytes.fromhex(key_value)
                    self.keys[key_name] = key_bytes
                except ValueError:
                    self.logger.warning(f"Invalid key format: {key_name}")
            
            # Check if we have the required keys
            required_keys = ["header_key", "key_area_key_application", "key_area_key_ocean", 
                            "titlekek_00", "titlekek_01"]
            
            missing_keys = [key for key in required_keys if key not in self.keys]
            if missing_keys:
                self.logger.warning(f"Missing required keys: {', '.join(missing_keys)}")
                
                # Create placeholder keys for missing ones (in demo mode)
                for key in missing_keys:
                    # Use appropriate key size based on key name
                    if key == "header_key":
                        self.keys[key] = b'\x00' * 32  # Header key is 32 bytes
                    else:
                        self.keys[key] = b'\x00' * 16  # Other keys are 16 bytes
                
                self.logger.warning("Using placeholder keys for missing keys (demo mode)")
            
            self.keys_loaded = True
            self.logger.info(f"Loaded {len(self.keys)} keys successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load keys: {e}", exc_info=True)
            return False
    
    def get_key(self, key_name):
        """Get a specific key by name
        
        Args:
            key_name: Name of the key
            
        Returns:
            Key bytes, or None if not found
        """
        return self.keys.get(key_name)
    
    def is_keys_loaded(self):
        """Check if keys are loaded
        
        Returns:
            True if keys are loaded, False otherwise
        """
        return self.keys_loaded


class RomLoader:
    """Loads and decrypts game ROMs"""
    
    # ROM header magic value
    NCA_MAGIC = b'NCA\x00'
    
    def __init__(self, rom_path, keys_path):
        """Initialize the ROM loader
        
        Args:
            rom_path: Path to the ROM file
            keys_path: Path to the keys file
        """
        self.logger = logging.getLogger("ImaginaryConsole.RomLoader")
        self.rom_path = Path(rom_path)
        self.keys_manager = KeysManager(keys_path)
        
        # ROM info
        self.rom_info = {}
    
    def _decrypt_header(self, header_data):
        """Decrypt the NCA header
        
        Args:
            header_data: Encrypted header data
            
        Returns:
            Decrypted header data
        """
        # Ensure keys are loaded - use placeholders if not available
        if not self.keys_manager.is_keys_loaded():
            self.logger.warning("Keys are not loaded, trying to load them now")
            if not self.keys_manager.load_keys():
                self.logger.warning("Still couldn't load keys, using demo mode")
        
        header_key = self.keys_manager.get_key("header_key")
        if not header_key:
            self.logger.warning("Header key not found, using demo mode")
            # Return the original data in demo mode
            return header_data
        
        try:
            # Create a cipher for XTS mode decryption
            # Note: This is a simplified implementation; actual NCA header uses XTS mode with sector tweaking
            cipher = Cipher(
                algorithms.AES(header_key),
                modes.ECB(),  # Simplified; should be XTS in actual implementation
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            # Decrypt the header (simplified)
            decrypted_header = decryptor.update(header_data) + decryptor.finalize()
            
            return decrypted_header
            
        except Exception as e:
            self.logger.warning(f"Error decrypting header: {e}, using demo mode")
            # In demo mode, just return the original data
            return header_data
    
    def _parse_header(self, header_data):
        """Parse the NCA header to extract ROM information
        
        Args:
            header_data: Decrypted header data
            
        Returns:
            Dictionary with ROM information
        """
        rom_info = {}
        
        try:
            # Check magic
            magic = header_data[:4]
            rom_info["magic"] = magic.decode('utf-8', errors='ignore')
            
            if magic != self.NCA_MAGIC:
                self.logger.warning(f"Invalid NCA magic: {magic}, assuming demo ROM")
                
                # Since this is a demo/emulator, we'll proceed anyway with default values
                rom_info.update({
                    "content_type": 0,
                    "crypto_type": 0,
                    "key_index": 0,
                    "size": len(header_data),
                    "title_id": 0x0123456789ABCDEF,
                    "sdk_version": 0x00000000,
                    "section_count": 1
                })
            else:
                # Valid NCA header, extract information
                try:
                    rom_info.update({
                        "content_type": header_data[0x10] & 0x7,
                        "crypto_type": header_data[0x11] & 0x7,
                        "key_index": (header_data[0x11] >> 4) & 0xF,
                        "size": struct.unpack("<Q", header_data[0x18:0x20])[0],
                        "title_id": struct.unpack("<Q", header_data[0x20:0x28])[0],
                        "sdk_version": struct.unpack("<I", header_data[0x28:0x2C])[0],
                        "section_count": header_data[0x2C]
                    })
                except (IndexError, struct.error) as e:
                    self.logger.warning(f"Error parsing header fields: {e}, using defaults")
                    rom_info.update({
                        "content_type": 0,
                        "crypto_type": 0,
                        "key_index": 0,
                        "size": len(header_data),
                        "title_id": 0x0123456789ABCDEF,
                        "sdk_version": 0x00000000,
                        "section_count": 1
                    })
        except Exception as e:
            self.logger.warning(f"Error processing header: {e}, using defaults")
            rom_info.update({
                "magic": "NCA?",
                "content_type": 0,
                "crypto_type": 0,
                "key_index": 0,
                "size": len(header_data) if header_data else 0,
                "title_id": 0x0123456789ABCDEF,
                "sdk_version": 0x00000000,
                "section_count": 1
            })
        
        # Set title based on filename
        rom_info["title"] = self.rom_path.stem
        
        # Add demo/placeholder indication if this is not a valid NCA
        if "magic" in rom_info and rom_info["magic"] != "NCA\x00":
            rom_info["title"] = f"{rom_info['title']} (Demo)"
        
        return rom_info
    
    def load(self):
        """Load and decrypt the ROM file
        
        Returns:
            Tuple of (rom_data, rom_info)
        """
        if not self.rom_path.exists():
            raise RomError(f"ROM file not found: {self.rom_path}")
        
        # Load keys
        if not self.keys_manager.load_keys():
            raise RomError("Failed to load keys")
        
        try:
            # Read the ROM file
            with open(self.rom_path, 'rb') as f:
                rom_data = f.read()
            
            # Extract and decrypt the header (first 0x400 bytes)
            header_data = rom_data[:0x400]
            decrypted_header = self._decrypt_header(header_data)
            
            # Parse the header
            self.rom_info = self._parse_header(decrypted_header)
            
            # Add file information
            self.rom_info["file_size"] = os.path.getsize(self.rom_path)
            self.rom_info["file_name"] = self.rom_path.name
            self.rom_info["path"] = str(self.rom_path)
            self.rom_info["sha256"] = hashlib.sha256(rom_data).hexdigest()
            
            # In a real implementation, we would decrypt the entire ROM here
            # For the emulator, we'll use the encrypted data for demonstration
            
            self.logger.info(f"Loaded ROM: {self.rom_info['title']} "
                             f"(Size: {self.rom_info['file_size']/1024/1024:.2f} MB)")
            
            return rom_data, self.rom_info
            
        except Exception as e:
            if isinstance(e, RomError):
                raise
            else:
                raise RomError(f"Failed to load ROM: {e}")
    
    def get_rom_info(self):
        """Get information about the ROM
        
        Returns:
            Dictionary with ROM information
        """
        return self.rom_info 