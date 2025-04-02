"""
ImaginaryConsole Emulator - Storage Module
Emulates the 256GB storage subsystem of the imaginary console.
"""

import logging
import os
import json
import time
from pathlib import Path
import threading


class StorageOperation:
    """Represents a storage I/O operation"""
    
    def __init__(self, op_type, path, size=0):
        """Initialize a storage operation
        
        Args:
            op_type: Type of operation ('read', 'write', 'delete', etc.)
            path: Path being operated on
            size: Size of the data being transferred (bytes)
        """
        self.op_type = op_type
        self.path = path
        self.size = size
        self.start_time = time.time()
        self.end_time = None
        self.success = None
        self.error = None
    
    def complete(self, success, error=None):
        """Mark the operation as completed"""
        self.end_time = time.time()
        self.success = success
        self.error = error
    
    @property
    def duration(self):
        """Get the duration of the operation in seconds"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


class Storage:
    """Emulates the storage subsystem of the imaginary console"""
    
    def __init__(self, size=256*1024*1024*1024):
        """Initialize the storage subsystem
        
        Args:
            size: Storage size in bytes (default 256GB)
        """
        self.logger = logging.getLogger("ImaginaryConsole.Storage")
        
        self.size = size
        
        # Storage is simulated using the local filesystem in the emulator's save directory
        self.storage_root = Path(os.path.expanduser("~/.imaginary_console/storage"))
        self.storage_root.mkdir(parents=True, exist_ok=True)
        
        # Performance characteristics (simplified)
        self.read_speed = 400 * 1024 * 1024  # 400 MB/s
        self.write_speed = 200 * 1024 * 1024  # 200 MB/s
        
        # Access statistics
        self.operations = []
        self.operation_lock = threading.Lock()
        
        # Current usage tracking
        self._update_usage()
        
        self.logger.info(f"Initialized storage system: {size/(1024*1024*1024):.2f} GB")
    
    def _update_usage(self):
        """Update the storage usage statistics"""
        # Calculate total size of files in the storage directory
        total_size = 0
        for dirpath, _, filenames in os.walk(self.storage_root):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        
        self.used_space = total_size
        self.free_space = max(0, self.size - self.used_space)
        
        self.logger.debug(f"Storage usage: {self.used_space/(1024*1024*1024):.2f} GB used, "
                          f"{self.free_space/(1024*1024*1024):.2f} GB free")
    
    def _record_operation(self, op_type, path, size=0):
        """Record a storage operation for statistics tracking"""
        operation = StorageOperation(op_type, path, size)
        
        with self.operation_lock:
            self.operations.append(operation)
            # Keep only the last 1000 operations
            if len(self.operations) > 1000:
                self.operations.pop(0)
        
        return operation
    
    def _get_full_path(self, virtual_path):
        """Convert a virtual path to an actual filesystem path"""
        # Normalize the path and make it relative to ensure it stays within our storage area
        norm_path = os.path.normpath(virtual_path).lstrip('/')
        return self.storage_root / norm_path
    
    def read_file(self, path, binary=False):
        """Read a file from storage
        
        Args:
            path: Virtual path to the file
            binary: Whether to read in binary mode
            
        Returns:
            File contents (bytes or string) and success flag
        """
        full_path = self._get_full_path(path)
        operation = self._record_operation('read', path)
        
        try:
            mode = 'rb' if binary else 'r'
            with open(full_path, mode) as f:
                data = f.read()
            
            # Update operation record
            file_size = len(data) if binary else len(data.encode('utf-8'))
            operation.size = file_size
            operation.complete(True)
            
            # Simulate read delay based on file size and read speed
            delay = file_size / self.read_speed
            if delay > 0.001:  # Only sleep for delays over 1ms
                time.sleep(delay)
            
            return data, True
            
        except Exception as e:
            operation.complete(False, str(e))
            self.logger.error(f"Failed to read file {path}: {e}")
            return None, False
    
    def write_file(self, path, data, binary=False):
        """Write data to a file in storage
        
        Args:
            path: Virtual path to the file
            data: Data to write (bytes or string)
            binary: Whether to write in binary mode
            
        Returns:
            Success flag and bytes written
        """
        full_path = self._get_full_path(path)
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Get data size for operation record
        data_size = len(data) if binary or isinstance(data, bytes) else len(data.encode('utf-8'))
        operation = self._record_operation('write', path, data_size)
        
        # Check if we have enough space
        self._update_usage()
        if data_size > self.free_space:
            self.logger.error(f"Not enough space to write {data_size} bytes to {path}")
            operation.complete(False, "Not enough storage space")
            return False, 0
        
        try:
            mode = 'wb' if binary else 'w'
            with open(full_path, mode) as f:
                f.write(data)
            
            # Simulate write delay based on data size and write speed
            delay = data_size / self.write_speed
            if delay > 0.001:  # Only sleep for delays over 1ms
                time.sleep(delay)
            
            operation.complete(True)
            
            # Update storage usage
            self._update_usage()
            
            return True, data_size
            
        except Exception as e:
            operation.complete(False, str(e))
            self.logger.error(f"Failed to write file {path}: {e}")
            return False, 0
    
    def delete_file(self, path):
        """Delete a file from storage
        
        Args:
            path: Virtual path to the file
            
        Returns:
            Success flag
        """
        full_path = self._get_full_path(path)
        operation = self._record_operation('delete', path)
        
        try:
            if os.path.exists(full_path):
                # Get file size before deleting
                file_size = os.path.getsize(full_path)
                operation.size = file_size
                
                # Delete the file
                os.remove(full_path)
                
                # Update operation record
                operation.complete(True)
                
                # Update storage usage
                self._update_usage()
                
                return True
            else:
                operation.complete(False, "File not found")
                return False
                
        except Exception as e:
            operation.complete(False, str(e))
            self.logger.error(f"Failed to delete file {path}: {e}")
            return False
    
    def list_directory(self, path):
        """List contents of a directory
        
        Args:
            path: Virtual path to the directory
            
        Returns:
            List of filenames and success flag
        """
        full_path = self._get_full_path(path)
        operation = self._record_operation('list', path)
        
        try:
            if os.path.isdir(full_path):
                contents = os.listdir(full_path)
                
                # Add metadata for each item
                result = []
                for item in contents:
                    item_path = os.path.join(full_path, item)
                    is_dir = os.path.isdir(item_path)
                    size = 0 if is_dir else os.path.getsize(item_path)
                    
                    result.append({
                        'name': item,
                        'is_directory': is_dir,
                        'size': size,
                        'modified': os.path.getmtime(item_path)
                    })
                
                operation.complete(True)
                return result, True
            else:
                operation.complete(False, "Not a directory")
                return [], False
                
        except Exception as e:
            operation.complete(False, str(e))
            self.logger.error(f"Failed to list directory {path}: {e}")
            return [], False
    
    def create_directory(self, path):
        """Create a directory in storage
        
        Args:
            path: Virtual path to the directory
            
        Returns:
            Success flag
        """
        full_path = self._get_full_path(path)
        operation = self._record_operation('mkdir', path)
        
        try:
            os.makedirs(full_path, exist_ok=True)
            operation.complete(True)
            return True
            
        except Exception as e:
            operation.complete(False, str(e))
            self.logger.error(f"Failed to create directory {path}: {e}")
            return False
    
    def get_usage_stats(self):
        """Get current storage usage statistics
        
        Returns:
            Dictionary with usage statistics
        """
        self._update_usage()
        
        return {
            "total_size": self.size,
            "used_space": self.used_space,
            "free_space": self.free_space,
            "usage_percent": (self.used_space / self.size) * 100
        }
    
    def get_performance_stats(self):
        """Get storage performance statistics
        
        Returns:
            Dictionary with performance statistics
        """
        with self.operation_lock:
            recent_ops = list(self.operations)
        
        # Count operations by type
        op_counts = {}
        for op in recent_ops:
            op_type = op.op_type
            if op_type not in op_counts:
                op_counts[op_type] = 0
            op_counts[op_type] += 1
        
        # Calculate average read/write speeds
        read_ops = [op for op in recent_ops if op.op_type == 'read' and op.end_time]
        write_ops = [op for op in recent_ops if op.op_type == 'write' and op.end_time]
        
        avg_read_speed = 0
        if read_ops:
            total_read_size = sum(op.size for op in read_ops)
            total_read_time = sum(op.duration for op in read_ops)
            if total_read_time > 0:
                avg_read_speed = total_read_size / total_read_time
        
        avg_write_speed = 0
        if write_ops:
            total_write_size = sum(op.size for op in write_ops)
            total_write_time = sum(op.duration for op in write_ops)
            if total_write_time > 0:
                avg_write_speed = total_write_size / total_write_time
        
        return {
            "operation_counts": op_counts,
            "total_operations": len(recent_ops),
            "average_read_speed": avg_read_speed,
            "average_write_speed": avg_write_speed,
            "max_read_speed": self.read_speed,
            "max_write_speed": self.write_speed
        } 