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
    """Manages encryption keys for ROM decryption
    
    This class handles the loading and management of two types of key files:
    1. prod.keys: Contains system-level keys such as header_key, key_area_keys, etc.
    2. title.keys: Contains game-specific decryption keys indexed by titleId
    
    Both files should be located in a 'keys' directory and follow the format:
    key_name = hex_value
    """
    
    def __init__(self, keys_path):
        """Initialize the keys manager
        
        Args:
            keys_path: Path to the keys directory or prod.keys file
        """
        self.logger = logging.getLogger("ImaginaryConsole.KeysManager")
        
        # Handle both directory and direct file path
        self.keys_path = Path(keys_path)
        if self.keys_path.is_file():
            # If user provided a direct file path, use it for prod.keys
            self.keys_dir = self.keys_path.parent
            self.prod_keys_path = self.keys_path
        else:
            # If user provided a directory, use it as the keys directory
            self.keys_dir = self.keys_path
            self.prod_keys_path = self.keys_dir / "prod.keys"
        
        # Set the title keys path
        self.title_keys_path = self.keys_dir / "title.keys"
        
        # Initialize key dictionaries
        self.prod_keys = {}  # Stores system keys from prod.keys
        self.title_keys = {} # Stores title-specific keys from title.keys
        
        # Flags indicating if keys are loaded
        self.prod_keys_loaded = False
        self.title_keys_loaded = False
    
    def load_prod_keys(self):
        """Load production encryption keys
        
        Returns:
            True if keys were loaded successfully, False otherwise
        """
        if not self.prod_keys_path.exists():
            self.logger.error(f"Production keys file not found: {self.prod_keys_path}")
            # Create a default keys structure with empty keys
            self._set_default_prod_keys()
            self.logger.warning("Using empty placeholder production keys (demo mode)")
            self.prod_keys_loaded = True
            return True
        
        try:
            self.logger.info(f"Loading production keys from: {self.prod_keys_path}")
            with open(self.prod_keys_path, 'r') as f:
                lines = f.readlines()
            
            # Parse keys file (format: key_name = hex_value)
            self._parse_key_file(lines, self.prod_keys)
            
            # Check if we have the required keys
            required_keys = ["header_key", "key_area_key_application_00", "key_area_key_ocean_00", 
                            "titlekek_00", "titlekek_01"]
            
            missing_keys = [key for key in required_keys if key not in self.prod_keys]
            if missing_keys:
                self.logger.warning(f"Missing required production keys: {', '.join(missing_keys)}")
                
                # Create placeholder keys for missing ones (in demo mode)
                self._add_missing_prod_keys(missing_keys)
            
            self.prod_keys_loaded = True
            self.logger.info(f"Loaded {len(self.prod_keys)} production keys successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load production keys: {e}", exc_info=True)
            return False
    
    def load_title_keys(self):
        """Load title-specific keys
        
        Returns:
            True if keys were loaded successfully, False otherwise
        """
        if not self.title_keys_path.exists():
            self.logger.warning(f"Title keys file not found: {self.title_keys_path}")
            # Create an empty dictionary for title keys
            self.title_keys = {}
            self.logger.warning("No title keys available (demo mode)")
            self.title_keys_loaded = True
            return True
        
        try:
            self.logger.info(f"Loading title keys from: {self.title_keys_path}")
            with open(self.title_keys_path, 'r') as f:
                lines = f.readlines()
            
            # Parse keys file (format: titleId = hex_value)
            self._parse_key_file(lines, self.title_keys)
            
            self.title_keys_loaded = True
            self.logger.info(f"Loaded {len(self.title_keys)} title keys successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load title keys: {e}", exc_info=True)
            return False
    
    def load_keys(self):
        """Load all encryption keys
        
        Returns:
            True if keys were loaded successfully, False otherwise
        """
        prod_result = self.load_prod_keys()
        title_result = self.load_title_keys()
        
        return prod_result and title_result
    
    def _parse_key_file(self, lines, keys_dict):
        """Parse a key file with lines in format: key_name = hex_value
        
        Args:
            lines: List of lines from the key file
            keys_dict: Dictionary to store the parsed keys
        """
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
                keys_dict[key_name] = key_bytes
            except ValueError:
                self.logger.warning(f"Invalid key format: {key_name}")
    
    def _set_default_prod_keys(self):
        """Set default empty production keys for demo mode"""
        self.prod_keys = {
            "header_key": b'\x00' * 32,
            "key_area_key_application_00": b'\x00' * 16,
            "key_area_key_ocean_00": b'\x00' * 16,
            "titlekek_00": b'\x00' * 16,
            "titlekek_01": b'\x00' * 16
        }
    
    def _add_missing_prod_keys(self, missing_keys):
        """Add missing production keys with placeholder values
        
        Args:
            missing_keys: List of key names that are missing
        """
        for key in missing_keys:
            # Use appropriate key size based on key name
            if key == "header_key":
                self.prod_keys[key] = b'\x00' * 32  # Header key is 32 bytes
            else:
                self.prod_keys[key] = b'\x00' * 16  # Other keys are 16 bytes
        
        self.logger.warning("Using placeholder keys for missing production keys (demo mode)")
    
    def get_prod_key(self, key_name):
        """Get a specific production key by name
        
        Args:
            key_name: Name of the production key
            
        Returns:
            Key bytes, or None if not found
        """
        return self.prod_keys.get(key_name)
    
    def get_title_key(self, title_id):
        """Get a title key for a specific title ID
        
        Args:
            title_id: Title ID (string or integer)
            
        Returns:
            Title key bytes, or None if not found
        """
        # Convert title_id to string if it's not already
        if not isinstance(title_id, str):
            title_id = f"{title_id:016X}"
        
        return self.title_keys.get(title_id)
    
    def is_keys_loaded(self):
        """Check if keys are loaded
        
        Returns:
            True if keys are loaded, False otherwise
        """
        return self.prod_keys_loaded and self.title_keys_loaded
    
    # Legacy compatibility method
    def get_key(self, key_name):
        """Get a specific key by name (for backwards compatibility)
        
        Args:
            key_name: Name of the key
            
        Returns:
            Key bytes, or None if not found
        """
        return self.get_prod_key(key_name)


class RomLoader:
    """Loads and decrypts game ROMs"""
    
    # ROM header magic values for different formats
    NCA_MAGIC = b'NCA\x00'
    NRO_MAGIC = b'NRO0'
    NSO_MAGIC = b'NSO0'
    NSP_MAGIC = b'PFS0'
    XPI_MAGIC = b'XPI0'
    
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
        
        # ROM format
        self.rom_format = None
    
    def _decrypt_header(self, header_data):
        """Decrypt the NCA/NSP header
        
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
        
        header_key = self.keys_manager.get_prod_key("header_key")
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
        """Parse the ROM header to extract ROM information
        
        Args:
            header_data: Decrypted header data
            
        Returns:
            Dictionary with ROM information
        """
        rom_info = {}
        
        try:
            # Check magic to determine format
            magic = header_data[:4]
            rom_info["magic"] = magic.decode('utf-8', errors='ignore')
            
            # Identify ROM format
            if magic == self.NCA_MAGIC:
                self.rom_format = "NCA"
                return self._parse_nca_header(header_data, rom_info)
            elif magic == self.NRO_MAGIC:
                self.rom_format = "NRO"
                return self._parse_nro_header(header_data, rom_info)
            elif magic == self.NSO_MAGIC:
                self.rom_format = "NSO"
                return self._parse_nso_header(header_data, rom_info)
            elif magic == self.NSP_MAGIC:
                self.rom_format = "NSP"
                return self._parse_nsp_header(header_data, rom_info)
            elif magic == self.XPI_MAGIC:
                self.rom_format = "XPI"
                return self._parse_xpi_header(header_data, rom_info)
            else:
                self.logger.warning(f"Unknown ROM format with magic: {magic}, assuming demo ROM")
                self.rom_format = "UNKNOWN"
                
                # Default ROM info for unknown format
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
                "magic": "UNK?",
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
        
        # Add demo/placeholder indication if this is not a recognized format
        if "magic" in rom_info and rom_info["magic"] not in ["NCA\x00", "NRO0", "NSO0", "PFS0", "XPI0"]:
            rom_info["title"] = f"{rom_info['title']} (Demo)"
        
        return rom_info
    
    def _parse_nca_header(self, header_data, rom_info):
        """Parse NCA format header
        
        Args:
            header_data: Header data
            rom_info: ROM info dictionary to update
            
        Returns:
            Updated ROM info dictionary
        """
        try:
            # Parse NCA-specific fields
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
            self.logger.warning(f"Error parsing NCA header fields: {e}, using defaults")
            rom_info.update({
                "content_type": 0,
                "crypto_type": 0,
                "key_index": 0,
                "size": len(header_data),
                "title_id": 0x0123456789ABCDEF,
                "sdk_version": 0x00000000,
                "section_count": 1
            })
        
        # Set title based on filename
        rom_info["title"] = self.rom_path.stem
        rom_info["format"] = "NCA"
        
        return rom_info
    
    def _parse_nro_header(self, header_data, rom_info):
        """Parse NRO format header
        
        Args:
            header_data: Header data
            rom_info: ROM info dictionary to update
            
        Returns:
            Updated ROM info dictionary
        """
        try:
            # NRO header structure
            # 0x00-0x03: Magic "NRO0"
            # 0x04-0x07: Version
            # 0x08-0x0F: Reserved 
            # 0x10-0x13: Size of .text segment
            # 0x14-0x17: Size of .ro segment
            # 0x18-0x1B: Size of .data segment
            # 0x1C-0x1F: Size of .bss segment
            # ... (more fields follow)
            
            rom_info.update({
                "version": struct.unpack("<I", header_data[0x04:0x08])[0],
                "text_size": struct.unpack("<I", header_data[0x10:0x14])[0],
                "ro_size": struct.unpack("<I", header_data[0x14:0x18])[0],
                "data_size": struct.unpack("<I", header_data[0x18:0x1C])[0],
                "bss_size": struct.unpack("<I", header_data[0x1C:0x20])[0],
                "size": os.path.getsize(self.rom_path),
                "title_id": 0,  # NRO doesn't have title ID in header
                "sdk_version": 0,  # Placeholder
            })
            
            # Try to extract program ID if asset section exists
            if len(header_data) >= 0x40:
                # Parse asset section offset if available (at 0x38)
                asset_offset = struct.unpack("<I", header_data[0x38:0x3C])[0]
                if asset_offset > 0 and asset_offset + 0x18 <= len(header_data):
                    # Check if there's a valid ASET header at the offset
                    aset_magic = header_data[asset_offset:asset_offset+4]
                    if aset_magic == b'ASET':
                        # Extract program ID from asset section
                        prog_id_offset = asset_offset + 0x10
                        if prog_id_offset + 8 <= len(header_data):
                            rom_info["title_id"] = struct.unpack("<Q", header_data[prog_id_offset:prog_id_offset+8])[0]
            
        except (IndexError, struct.error) as e:
            self.logger.warning(f"Error parsing NRO header fields: {e}, using defaults")
            rom_info.update({
                "version": 0,
                "text_size": 0,
                "ro_size": 0, 
                "data_size": 0,
                "bss_size": 0,
                "size": os.path.getsize(self.rom_path),
                "title_id": 0,
                "sdk_version": 0
            })
        
        # Set title based on filename
        rom_info["title"] = self.rom_path.stem
        rom_info["format"] = "NRO"
        
        return rom_info
    
    def _parse_nso_header(self, header_data, rom_info):
        """Parse NSO format header
        
        Args:
            header_data: Header data
            rom_info: ROM info dictionary to update
            
        Returns:
            Updated ROM info dictionary
        """
        try:
            # NSO header structure
            # 0x00-0x03: Magic "NSO0"
            # 0x04-0x07: Version
            # 0x08: Flags
            # 0x0C-0x0F: .text file offset
            # 0x10-0x13: .text memory offset
            # 0x14-0x17: .text section size
            # ... (similar for .rodata and .data sections)
            
            rom_info.update({
                "version": struct.unpack("<I", header_data[0x04:0x08])[0],
                "flags": header_data[0x08],
                "text_offset": struct.unpack("<I", header_data[0x0C:0x10])[0],
                "text_address": struct.unpack("<I", header_data[0x10:0x14])[0],
                "text_size": struct.unpack("<I", header_data[0x14:0x18])[0],
                "ro_offset": struct.unpack("<I", header_data[0x18:0x1C])[0], 
                "ro_address": struct.unpack("<I", header_data[0x1C:0x20])[0],
                "ro_size": struct.unpack("<I", header_data[0x20:0x24])[0],
                "data_offset": struct.unpack("<I", header_data[0x24:0x28])[0],
                "data_address": struct.unpack("<I", header_data[0x28:0x2C])[0],
                "data_size": struct.unpack("<I", header_data[0x2C:0x30])[0], 
                "bss_size": struct.unpack("<I", header_data[0x38:0x3C])[0],
                "size": os.path.getsize(self.rom_path),
                "title_id": 0,  # NSO doesn't contain title ID
                "sdk_version": 0  # Placeholder
            })
            
            # Check for compression flags
            rom_info["text_compressed"] = bool(rom_info["flags"] & 0x01)
            rom_info["ro_compressed"] = bool(rom_info["flags"] & 0x02)
            rom_info["data_compressed"] = bool(rom_info["flags"] & 0x04)
            
        except (IndexError, struct.error) as e:
            self.logger.warning(f"Error parsing NSO header fields: {e}, using defaults")
            rom_info.update({
                "version": 0,
                "flags": 0,
                "text_offset": 0, 
                "text_address": 0,
                "text_size": 0,
                "ro_offset": 0,
                "ro_address": 0,
                "ro_size": 0,
                "data_offset": 0,
                "data_address": 0,
                "data_size": 0,
                "bss_size": 0,
                "size": os.path.getsize(self.rom_path),
                "title_id": 0,
                "sdk_version": 0,
                "text_compressed": False,
                "ro_compressed": False,
                "data_compressed": False
            })
        
        # Set title based on filename
        rom_info["title"] = self.rom_path.stem
        rom_info["format"] = "NSO"
        
        return rom_info
    
    def _parse_nsp_header(self, header_data, rom_info):
        """Parse NSP format header (which is a PFS0 archive)
        
        Args:
            header_data: Header data
            rom_info: ROM info dictionary to update
            
        Returns:
            Updated ROM info dictionary
        """
        try:
            # PFS0 header structure
            # 0x00-0x03: Magic "PFS0"
            # 0x04-0x07: Number of files
            # 0x08-0x0B: String table size
            # 0x0C-0x0F: Reserved
            # 0x10-...: File entry table (each entry is 0x18 bytes)
            # After file entry table: String table
            
            num_files = struct.unpack("<I", header_data[0x04:0x08])[0]
            string_table_size = struct.unpack("<I", header_data[0x08:0x0C])[0]
            
            rom_info.update({
                "num_files": num_files,
                "string_table_size": string_table_size,
                "size": os.path.getsize(self.rom_path),
                "title_id": 0,  # Will try to extract from metadata if available
                "sdk_version": 0  # Placeholder
            })
            
            # Extract filenames from string table
            file_entries_offset = 0x10
            string_table_offset = file_entries_offset + (num_files * 0x18)
            
            if string_table_offset + string_table_size <= len(header_data):
                string_table = header_data[string_table_offset:string_table_offset + string_table_size]
                
                # Extract file entries
                files = []
                for i in range(num_files):
                    entry_offset = file_entries_offset + (i * 0x18)
                    
                    # Ensure we have enough data
                    if entry_offset + 0x18 <= len(header_data):
                        data_offset = struct.unpack("<Q", header_data[entry_offset:entry_offset+8])[0]
                        data_size = struct.unpack("<Q", header_data[entry_offset+8:entry_offset+16])[0]
                        name_offset = struct.unpack("<I", header_data[entry_offset+16:entry_offset+20])[0]
                        
                        # Extract filename from string table
                        if name_offset < string_table_size:
                            # Find null terminator
                            end_pos = string_table.find(b'\x00', name_offset)
                            if end_pos == -1:
                                end_pos = len(string_table)
                            
                            filename = string_table[name_offset:end_pos].decode('utf-8', errors='ignore')
                            
                            files.append({
                                "name": filename,
                                "offset": data_offset,
                                "size": data_size
                            })
                            
                            # Check for CNMT file to extract title ID
                            if filename.endswith('.cnmt.nca'):
                                rom_info["cnmt_nca_index"] = i
                
                rom_info["files"] = files
                
                # If we have files, use the first filename for the title if it's a NCA
                if files and files[0]["name"].endswith('.nca'):
                    rom_info["main_nca"] = files[0]["name"]
            
        except (IndexError, struct.error) as e:
            self.logger.warning(f"Error parsing NSP header fields: {e}, using defaults")
            rom_info.update({
                "num_files": 0,
                "string_table_size": 0,
                "size": os.path.getsize(self.rom_path),
                "title_id": 0,
                "sdk_version": 0,
                "files": []
            })
        
        # Set title based on filename
        rom_info["title"] = self.rom_path.stem
        rom_info["format"] = "NSP"
        
        return rom_info
    
    def _parse_xpi_header(self, header_data, rom_info):
        """Parse XPI format header (fictional format for the ImaginaryConsole)
        
        Args:
            header_data: Header data
            rom_info: ROM info dictionary to update
            
        Returns:
            Updated ROM info dictionary
        """
        try:
            # XPI is a fictional format, so we'll define a plausible structure
            # 0x00-0x03: Magic "XPI0"
            # 0x04-0x07: Version
            # 0x08-0x0F: Application ID
            # 0x10-0x17: Title ID
            # 0x18-0x1B: Flags
            # 0x1C-0x1F: SDK Version
            # 0x20-0x23: Number of sections
            # 0x24-0x27: Header size
            # 0x28-0x2F: Data offset
            # 0x30-0x37: Data size
            # 0x38-0x3F: Reserved
            # 0x40-0x5F: Title name (32 bytes, null-terminated)
            # 0x60-0x7F: Publisher name (32 bytes, null-terminated)
            # 0x80-...: Section table
            
            version = struct.unpack("<I", header_data[0x04:0x08])[0]
            app_id = struct.unpack("<Q", header_data[0x08:0x10])[0]
            title_id = struct.unpack("<Q", header_data[0x10:0x18])[0]
            flags = struct.unpack("<I", header_data[0x18:0x1C])[0]
            sdk_version = struct.unpack("<I", header_data[0x1C:0x20])[0]
            num_sections = struct.unpack("<I", header_data[0x20:0x24])[0]
            header_size = struct.unpack("<I", header_data[0x24:0x28])[0]
            data_offset = struct.unpack("<Q", header_data[0x28:0x30])[0]
            data_size = struct.unpack("<Q", header_data[0x30:0x38])[0]
            
            # Extract title and publisher names (null-terminated strings)
            title_name_bytes = header_data[0x40:0x60]
            publisher_bytes = header_data[0x60:0x80]
            
            # Find terminating null byte
            title_end = title_name_bytes.find(b'\x00')
            if title_end == -1:
                title_end = len(title_name_bytes)
            
            publisher_end = publisher_bytes.find(b'\x00')
            if publisher_end == -1:
                publisher_end = len(publisher_bytes)
            
            title_name = title_name_bytes[:title_end].decode('utf-8', errors='ignore')
            publisher = publisher_bytes[:publisher_end].decode('utf-8', errors='ignore')
            
            rom_info.update({
                "version": version,
                "app_id": app_id,
                "title_id": title_id,
                "flags": flags,
                "sdk_version": sdk_version,
                "num_sections": num_sections,
                "header_size": header_size,
                "data_offset": data_offset,
                "data_size": data_size,
                "title_name": title_name,
                "publisher": publisher,
                "size": os.path.getsize(self.rom_path)
            })
            
            # Parse section table if available
            sections = []
            section_table_offset = 0x80
            
            for i in range(num_sections):
                # Each section entry is 32 bytes
                section_offset = section_table_offset + (i * 32)
                
                if section_offset + 32 <= len(header_data):
                    section_type = struct.unpack("<I", header_data[section_offset:section_offset+4])[0]
                    section_flags = struct.unpack("<I", header_data[section_offset+4:section_offset+8])[0]
                    section_offset_value = struct.unpack("<Q", header_data[section_offset+8:section_offset+16])[0]
                    section_size = struct.unpack("<Q", header_data[section_offset+16:section_offset+24])[0]
                    section_reserved = struct.unpack("<Q", header_data[section_offset+24:section_offset+32])[0]
                    
                    sections.append({
                        "type": section_type,
                        "flags": section_flags,
                        "offset": section_offset_value,
                        "size": section_size
                    })
            
            if sections:
                rom_info["sections"] = sections
            
        except (IndexError, struct.error) as e:
            self.logger.warning(f"Error parsing XPI header fields: {e}, using defaults")
            rom_info.update({
                "version": 0,
                "app_id": 0,
                "title_id": 0,
                "flags": 0,
                "sdk_version": 0,
                "num_sections": 0,
                "header_size": 0x80,
                "data_offset": 0x80,
                "data_size": os.path.getsize(self.rom_path) - 0x80,
                "title_name": "",
                "publisher": "",
                "size": os.path.getsize(self.rom_path)
            })
        
        # Use title from header if available, otherwise from filename
        if rom_info.get("title_name"):
            rom_info["title"] = rom_info["title_name"]
        else:
            rom_info["title"] = self.rom_path.stem
            
        rom_info["format"] = "XPI"
        
        return rom_info
    
    def _decrypt_content(self, encrypted_data, title_id):
        """Decrypt content using a title key
        
        Args:
            encrypted_data: Content to decrypt
            title_id: Title ID for the content
            
        Returns:
            Decrypted content data
        """
        # Ensure keys are loaded
        if not self.keys_manager.is_keys_loaded():
            self.logger.warning("Keys are not loaded, trying to load them now")
            if not self.keys_manager.load_keys():
                self.logger.warning("Still couldn't load keys, using demo mode")
        
        # Get the title key for this title ID
        title_key = self.keys_manager.get_title_key(title_id)
        if not title_key:
            self.logger.warning(f"Title key not found for title ID: {title_id}, using demo mode")
            # Return the original data in demo mode
            return encrypted_data
        
        try:
            # Create a cipher for CBC mode decryption
            # For simplicity, we're using a zero IV; in reality, it would be derived from the title ID
            iv = b'\x00' * 16
            
            cipher = Cipher(
                algorithms.AES(title_key),
                modes.CBC(iv),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            # Decrypt the content
            decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
            
            return decrypted_data
            
        except Exception as e:
            self.logger.warning(f"Error decrypting content: {e}, using demo mode")
            # In demo mode, just return the original data
            return encrypted_data
    
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
            
            # Detect format based on magic
            format_magic = header_data[:4]
            
            # Only decrypt header for NCA and NSP formats
            if format_magic == self.NCA_MAGIC or format_magic == self.NSP_MAGIC:
                decrypted_header = self._decrypt_header(header_data)
            else:
                # NRO, NSO, and XPI formats don't need header decryption
                decrypted_header = header_data
            
            # Parse the header
            self.rom_info = self._parse_header(decrypted_header)
            
            # Add file information
            self.rom_info["file_size"] = os.path.getsize(self.rom_path)
            self.rom_info["file_name"] = self.rom_path.name
            self.rom_info["path"] = str(self.rom_path)
            self.rom_info["sha256"] = hashlib.sha256(rom_data).hexdigest()
            
            # For NSP files, we might need to extract and handle individual NCA files
            # This would be implemented here in a real emulator
            
            self.logger.info(f"Loaded ROM: {self.rom_info['title']} "
                             f"({self.rom_info.get('format', 'Unknown Format')}, "
                             f"Size: {self.rom_info['file_size']/1024/1024:.2f} MB)")
            
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