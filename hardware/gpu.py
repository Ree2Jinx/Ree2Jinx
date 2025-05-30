"""
ImaginaryConsole Emulator - GPU Module
Emulates the Nvidia T239 Ampere GPU with 1 graphics cluster, 12 streaming multiprocessors, 1534 CUDA cores.
"""

import logging
import numpy as np
import time
import threading
from enum import Enum
import struct


class ShaderType(Enum):
    """Types of shaders that the GPU can execute"""
    VERTEX = 0
    FRAGMENT = 1
    COMPUTE = 2
    GEOMETRY = 3
    TESSELLATION = 4


class GpuMode(Enum):
    """GPU operating modes"""
    IDLE = 0
    RENDERING = 1
    COMPUTE = 2
    TRANSFER = 3
    SLEEP = 4


class ShaderProgram:
    """Represents a shader program in the GPU"""
    
    def __init__(self, shader_type, program_id):
        """Initialize a shader program"""
        self.shader_type = shader_type
        self.program_id = program_id
        self.instructions = []
        self.constant_data = []
        self.input_layout = []
        self.output_layout = []
        self.uniforms = {}
        self.attributes = {}
    
    def load_instructions(self, instructions):
        """Load shader instructions (binary)"""
        self.instructions = instructions
        return True
    
    def load_constants(self, constants):
        """Load shader constant data"""
        self.constant_data = constants
        return True

    def set_uniform(self, name, value):
        """Set a uniform value for the shader program.
        
        Args:
            name (str): The name of the uniform.
            value: The value of the uniform (e.g., float, list of floats for vectors/matrices).
        """
        self.uniforms[name] = value

    def define_attribute(self, name, offset, attrib_format, stride):
        """Define a vertex attribute for the shader program.
        
        Args:
            name (str): The name of the attribute (e.g., "position", "texcoord").
            offset (int): Offset in bytes within the vertex data.
            attrib_format (str): Format of the attribute (e.g., "vec3", "vec2", "float").
            stride (int): Stride in bytes between consecutive vertices.
        """
        self.attributes[name] = {"offset": offset, "format": attrib_format, "stride": stride}


class Texture:
    """Represents a GPU Texture"""

    def __init__(self, texture_id, width, height, depth=1, texture_format='RGBA8', data=None):
        """Initialize a Texture
        
        Args:
            texture_id: Unique ID for the texture.
            width (int): Width of the texture.
            height (int): Height of the texture.
            depth (int): Depth of the texture (for 3D textures, default 1 for 2D).
            texture_format (str): Pixel format (e.g., 'RGBA8', 'RGB8', 'R8', 'RGBA16F').
            data (bytes/bytearray, optional): Initial pixel data.
        """
        self.texture_id = texture_id
        self.width = width
        self.height = height
        self.depth = depth
        self.format = texture_format
        
        # Determine bytes per pixel based on format
        if texture_format == 'RGBA8':
            self.bytes_per_pixel = 4
        elif texture_format == 'RGB8':
            self.bytes_per_pixel = 3
        elif texture_format == 'R8':
            self.bytes_per_pixel = 1
        elif texture_format == 'RGBA16F': # 16-bit float per component
            self.bytes_per_pixel = 8 
        else:
            # Default or throw error for unsupported format
            raise ValueError(f"Unsupported texture format: {texture_format}")

        self.data_size = self.width * self.height * self.depth * self.bytes_per_pixel
        if data:
            if len(data) != self.data_size:
                raise ValueError("Provided data size does not match texture dimensions and format.")
            self.data = bytearray(data)
        else:
            self.data = bytearray(self.data_size) # Initialize with zeros if no data provided

    def upload_data(self, data, offset=0, mipmap_level=0):
        """Upload pixel data to the texture.
        
        Args:
            data (bytes/bytearray): The pixel data to upload.
            offset (int): Offset in bytes into the texture data to start writing.
            mipmap_level (int): Mipmap level to upload to (currently ignored, placeholder).
        """
        if mipmap_level != 0:
            # Mipmap handling would go here
            pass
        
        data_len = len(data)
        if offset + data_len > self.data_size:
            raise ValueError("Data size exceeds texture capacity with given offset.")
        
        self.data[offset:offset+data_len] = data

    def sample(self, u, v, w=0, mipmap_level=0):
        """Sample a color from the texture using normalized coordinates.
        (Simplified point sampling without filtering or mipmapping)
        
        Args:
            u (float): U coordinate (0.0 to 1.0).
            v (float): V coordinate (0.0 to 1.0).
            w (float): W coordinate (0.0 to 1.0 for 3D textures).
            mipmap_level (int): Mipmap level to sample from (currently ignored).

        Returns:
            tuple: Color value (e.g., (R, G, B, A) as integers 0-255 for RGBA8).
                   Returns (0,0,0,0) if coordinates are out of bounds or format is unknown.
        """
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0 and 0.0 <= w <= 1.0):
            return (0, 0, 0, 0) # Out of bounds

        x = int(u * (self.width -1))
        y = int(v * (self.height -1))
        z = int(w * (self.depth -1)) # For 3D textures

        if self.depth > 1: # 3D texture
            pixel_offset = (z * self.height * self.width + y * self.width + x) * self.bytes_per_pixel
        else: # 2D texture
            pixel_offset = (y * self.width + x) * self.bytes_per_pixel

        if pixel_offset + self.bytes_per_pixel > self.data_size:
            return (0,0,0,0) # Should not happen with correct coord clamping

        if self.format == 'RGBA8':
            return tuple(self.data[pixel_offset : pixel_offset + 4])
        elif self.format == 'RGB8':
            return tuple(self.data[pixel_offset : pixel_offset + 3]) + (255,) # Add alpha
        elif self.format == 'R8':
            val = self.data[pixel_offset]
            return (val, val, val, 255) # Grayscale to RGBA
        elif self.format == 'RGBA16F':
            # Placeholder: 16F conversion is more complex. Return raw bytes for now.
            # In a real implementation, you'd unpack these floats.
            return tuple(self.data[pixel_offset : pixel_offset + 8]) 
        
        return (0,0,0,0) # Default for unsupported formats


class GraphicsState:
    """Manages various states of the graphics pipeline."""
    def __init__(self):
        self.viewport_x = 0
        self.viewport_y = 0
        self.viewport_width = 0 # Set during framebuffer_size or set_viewport
        self.viewport_height = 0 # Set during framebuffer_size or set_viewport

        self.depth_test_enabled = False
        self.depth_function = 'LESS' # e.g., LESS, LEQUAL, GREATER, ALWAYS

        self.blend_enabled = False
        self.blend_src_factor_rgb = 'SRC_ALPHA'
        self.blend_dst_factor_rgb = 'ONE_MINUS_SRC_ALPHA'
        self.blend_src_factor_alpha = 'ONE'
        self.blend_dst_factor_alpha = 'ZERO'
        self.blend_equation_rgb = 'FUNC_ADD'
        self.blend_equation_alpha = 'FUNC_ADD'

        self.cull_face_enabled = False
        self.cull_face_mode = 'BACK' # FRONT, BACK, FRONT_AND_BACK
        self.front_face_winding = 'CCW' # Counter-Clockwise, CW for Clockwise

        self.scissor_test_enabled = False
        self.scissor_x = 0
        self.scissor_y = 0
        self.scissor_width = 0
        self.scissor_height = 0

        self.color_mask_r = True
        self.color_mask_g = True
        self.color_mask_b = True
        self.color_mask_a = True
        
        self.line_width = 1.0

        # Add other states as needed:
        # - Stencil test
        # - Polygon offset
        # - Dithering
        # - etc.

class RenderTarget:
    """Represents a GPU render target (framebuffer)"""
    
    def __init__(self, width, height, format='RGBA8'):
        """Initialize a render target"""
        self.width = width
        self.height = height
        self.format = format
        
        # Determine bytes per pixel based on format
        if format == 'RGBA8':
            self.bytes_per_pixel = 4
        elif format == 'RGB8':
            self.bytes_per_pixel = 3
        elif format == 'R8':
            self.bytes_per_pixel = 1
        elif format == 'RGBA16F':
            self.bytes_per_pixel = 8
        else:
            self.bytes_per_pixel = 4  # Default
        
        # Allocate memory for the render target
        self.data = bytearray(width * height * self.bytes_per_pixel)
        
        # Clear to black
        self.clear((0, 0, 0, 255))
    
    def resize(self, width, height):
        """Resize the render target"""
        if width == self.width and height == self.height:
            return
        
        self.width = width
        self.height = height
        self.data = bytearray(width * height * self.bytes_per_pixel)
    
    def clear(self, color=(0, 0, 0, 255)):
        """Clear the render target to the specified color"""
        if self.format == 'RGBA8':
            for i in range(0, len(self.data), 4):
                self.data[i] = color[0]      # R
                self.data[i+1] = color[1]    # G
                self.data[i+2] = color[2]    # B
                self.data[i+3] = color[3]    # A
        elif self.format == 'RGB8':
            for i in range(0, len(self.data), 3):
                self.data[i] = color[0]      # R
                self.data[i+1] = color[1]    # G
                self.data[i+2] = color[2]    # B
        elif self.format == 'R8':
            for i in range(len(self.data)):
                self.data[i] = color[0]      # R
    
    def set_pixel(self, x, y, color):
        """Set a pixel at the specified coordinates"""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return
        
        offset = (y * self.width + x) * self.bytes_per_pixel
        
        if self.format == 'RGBA8':
            self.data[offset] = color[0]       # R
            self.data[offset+1] = color[1]     # G
            self.data[offset+2] = color[2]     # B
            self.data[offset+3] = color[3]     # A
        elif self.format == 'RGB8':
            self.data[offset] = color[0]       # R
            self.data[offset+1] = color[1]     # G
            self.data[offset+2] = color[2]     # B
        elif self.format == 'R8':
            self.data[offset] = color[0]       # R
    
    def get_pixel(self, x, y):
        """Get a pixel at the specified coordinates"""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return (0, 0, 0, 0)
        
        offset = (y * self.width + x) * self.bytes_per_pixel
        
        if self.format == 'RGBA8':
            return (self.data[offset], self.data[offset+1], self.data[offset+2], self.data[offset+3])
        elif self.format == 'RGB8':
            return (self.data[offset], self.data[offset+1], self.data[offset+2], 255)
        elif self.format == 'R8':
            return (self.data[offset], 0, 0, 255)
        
        return (0, 0, 0, 0)
    
    def fill_rect(self, x, y, width, height, color):
        """Fill a rectangle with the specified color"""
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(self.width, x + width)
        y_end = min(self.height, y + height)
        
        for py in range(y_start, y_end):
            for px in range(x_start, x_end):
                self.set_pixel(px, py, color)
    
    def get_data(self):
        """Get the raw data of the render target"""
        return bytes(self.data)


class VertexBuffer:
    """Represents a GPU vertex buffer"""
    
    def __init__(self, size_bytes):
        """Initialize a vertex buffer"""
        self.size = size_bytes
        self.data = bytearray(size_bytes)
        self.stride = 0
        self.vertex_count = 0
    
    def upload_data(self, data, offset=0):
        """Upload data to the vertex buffer"""
        if offset + len(data) > self.size:
            return False
        
        # Copy data to the buffer
        for i in range(len(data)):
            self.data[offset + i] = data[i]
        
        return True
    
    def set_layout(self, stride, vertex_count):
        """Set the vertex layout"""
        self.stride = stride
        self.vertex_count = vertex_count


class IndexBuffer:
    """Represents a GPU index buffer"""
    
    def __init__(self, size_bytes, index_type='uint16'):
        """Initialize an index buffer
        
        Args:
            size_bytes: Size of the buffer in bytes
            index_type: Type of indices ('uint16' or 'uint32')
        """
        self.size = size_bytes
        self.data = bytearray(size_bytes)
        self.index_type = index_type
        self.index_count = 0
    
    def upload_data(self, data, offset=0):
        """Upload data to the index buffer"""
        if offset + len(data) > self.size:
            return False
        
        # Copy data to the buffer
        for i in range(len(data)):
            self.data[offset + i] = data[i]
        
        # Update index count based on data size
        element_size = 2 if self.index_type == 'uint16' else 4
        self.index_count = len(data) // element_size
        
        return True
    
    def get_index(self, index):
        """Get the index at the specified position"""
        if index >= self.index_count:
            return 0
        
        if self.index_type == 'uint16':
            offset = index * 2
            return struct.unpack('<H', self.data[offset:offset+2])[0]
        else:  # uint32
            offset = index * 4
            return struct.unpack('<I', self.data[offset:offset+4])[0]


class StreamingMultiprocessor:
    """Emulates a single NVIDIA streaming multiprocessor (SM)"""
    
    def __init__(self, sm_id, cuda_cores_per_sm):
        """Initialize an SM with the given ID and number of CUDA cores"""
        self.sm_id = sm_id
        self.cuda_core_count = cuda_cores_per_sm
        
        # Current operation state
        self.busy = False
        self.current_shader = None
        self.current_workload = None
        
        # Performance counters
        self.operations_completed = 0
        self.cycles_executed = 0
        
        # Simplified representation of CUDA cores utilization (0.0 to 1.0)
        self.utilization = 0.0
        
        # Shader program cache
        self.shader_cache = {}
    
    def execute_workload(self, shader_type, workload_size, cycles):
        """Execute a shader workload on this SM"""
        self.busy = True
        self.current_shader = shader_type
        self.current_workload = workload_size
        
        # Simulate execution - in a real emulator this would perform the actual shader operations
        # Here we just update counters
        self.operations_completed += workload_size
        self.cycles_executed += cycles
        
        # Calculate utilization - simplified model
        self.utilization = min(1.0, workload_size / (self.cuda_core_count * 2))
        
        self.busy = False
        return workload_size
    
    def execute_vertex_shader(self, shader_program, vertex_buffer, start_vertex, vertex_count, output_processed_vertices_list, all_uniforms):
        """Execute a vertex shader on vertices and append to output_processed_vertices_list.
        
        Args:
            shader_program: ShaderProgram object to execute.
            vertex_buffer: Input VertexBuffer object.
            start_vertex: Starting vertex index in the vertex_buffer.
            vertex_count: Number of vertices to process.
            output_processed_vertices_list: A list to append processed vertex data (dictionaries).
            all_uniforms: A dictionary containing all uniforms (VS and FS) relevant to this draw call.
            
        Returns:
            Number of vertices processed.
        """
        if not shader_program or shader_program.shader_type != ShaderType.VERTEX:
            self.logger.warning(f"SM {self.sm_id}: Invalid or non-vertex shader program provided.") # SM needs a logger
            return 0
        if not vertex_buffer:
            self.logger.warning(f"SM {self.sm_id}: No vertex buffer provided.")
            return 0

        # Ensure logger is available (can be passed in or use a default)
        if not hasattr(self, 'logger'): 
            self.logger = logging.getLogger(f"ImaginaryConsole.GPU.SM{self.sm_id}")

        self.busy = True
        self.current_shader = ShaderType.VERTEX
        
        processed_count = 0
        vertex_attributes_layout = shader_program.attributes
        if not vertex_attributes_layout:
            self.logger.warning(f"SM {self.sm_id}: Vertex shader {shader_program.program_id} has no attributes defined.")

        for i in range(vertex_count):
            current_vertex_index_in_buffer = start_vertex + i
            if current_vertex_index_in_buffer >= vertex_buffer.vertex_count: # Boundary check
                self.logger.warning(f"SM {self.sm_id}: Vertex index {current_vertex_index_in_buffer} out of bounds for VB count {vertex_buffer.vertex_count}")
                break

            vertex_base_offset = current_vertex_index_in_buffer * vertex_buffer.stride
            fetched_attributes = {}
            for attr_name, attr_props in vertex_attributes_layout.items():
                attr_offset = vertex_base_offset + attr_props['offset']
                attr_format_str = attr_props['format']
                try:
                    if attr_format_str == 'vec3f': val = struct.unpack_from('<fff', vertex_buffer.data, attr_offset)
                    elif attr_format_str == 'vec4f': val = struct.unpack_from('<ffff', vertex_buffer.data, attr_offset)
                    elif attr_format_str == 'vec2f': val = struct.unpack_from('<ff', vertex_buffer.data, attr_offset)
                    elif attr_format_str == 'float': val = struct.unpack_from('<f', vertex_buffer.data, attr_offset)[0]
                    elif attr_format_str == 'vec4ub': val = struct.unpack_from('<BBBB', vertex_buffer.data, attr_offset)
                    elif attr_format_str == 'vec4ub_norm': raw_val = struct.unpack_from('<BBBB', vertex_buffer.data, attr_offset); val = tuple(x / 255.0 for x in raw_val)
                    elif attr_format_str == 'vec2s': val = struct.unpack_from('<hh', vertex_buffer.data, attr_offset)
                    elif attr_format_str == 'vec4s': val = struct.unpack_from('<hhhh', vertex_buffer.data, attr_offset)
                    elif attr_format_str == 'vec4s_norm': raw_val = struct.unpack_from('<hhhh', vertex_buffer.data, attr_offset); val = tuple(max(-1.0, x / 32767.0) for x in raw_val)
                    elif attr_format_str == 'vec2i': val = struct.unpack_from('<ii', vertex_buffer.data, attr_offset)
                    elif attr_format_str == 'vec4i': val = struct.unpack_from('<iiii', vertex_buffer.data, attr_offset)
                    else: self.logger.warning(f"SM {self.sm_id}: Unsupported attribute format '{attr_format_str}' for {attr_name}"); val = None
                    fetched_attributes[attr_name] = val
                except struct.error as e: self.logger.error(f"SM {self.sm_id}: Error unpacking {attr_name} (fmt: {attr_format_str}) @ off {attr_offset}: {e}"); fetched_attributes[attr_name] = None
            
            current_varyings = {}
            if 'texcoord' in fetched_attributes and fetched_attributes['texcoord'] is not None: current_varyings['texcoord'] = fetched_attributes['texcoord']
            if 'color' in fetched_attributes and fetched_attributes['color'] is not None: current_varyings['color'] = fetched_attributes['color']

            mvp_matrix_uniform_name = "u_mvpMatrix"
            final_transformed_position = list(fetched_attributes.get("position", [0,0,0,1])) # Default if no pos
            # Access uniforms directly from the shader_program object now, or a passed-in merged dict for draw call
            if mvp_matrix_uniform_name in shader_program.uniforms: # Check shader_program specific uniforms
                mvp_matrix = shader_program.uniforms[mvp_matrix_uniform_name]
                # Actual transformation would happen here. Dummy scaling:
                if isinstance(mvp_matrix, list) and len(mvp_matrix) > 0 and isinstance(mvp_matrix[0], (int, float)):
                    if len(final_transformed_position) > 0 : final_transformed_position[0] *= mvp_matrix[0]
            elif mvp_matrix_uniform_name in all_uniforms: # Fallback to check draw call level uniforms if any
                 mvp_matrix = all_uniforms[mvp_matrix_uniform_name]
                 if isinstance(mvp_matrix, list) and len(mvp_matrix) > 0 and isinstance(mvp_matrix[0], (int, float)):
                    if len(final_transformed_position) > 0 : final_transformed_position[0] *= mvp_matrix[0]

            output_processed_vertices_list.append({
                "original_attributes": fetched_attributes, # Could be omitted in real GPU output
                "transformed_position": final_transformed_position,
                "varyings": current_varyings
            })
            processed_count += 1
        
        self.operations_completed += processed_count
        self.cycles_executed += processed_count * 10  # Rough cycle count
        self.utilization = min(1.0, processed_count / (self.cuda_core_count * 0.5)) # Simplified
        
        self.busy = False
        return processed_count
    
    def execute_fragment_shader(self, shader_program, fragment_count, render_target):
        """Execute a fragment shader
        
        Args:
            shader_program: Shader program to execute
            fragment_count: Number of fragments to process
            render_target: Output render target
            
        Returns:
            Number of fragments processed
        """
        if not shader_program or shader_program.shader_type != ShaderType.FRAGMENT:
            return 0
        
        # Mark SM as busy
        self.busy = True
        self.current_shader = ShaderType.FRAGMENT
        
        # Process fragments
        processed_fragments = fragment_count
        
        # Update performance counters
        self.operations_completed += processed_fragments
        self.cycles_executed += processed_fragments * 20  # Assume 20 cycles per fragment
        
        # Calculate utilization
        self.utilization = min(1.0, processed_fragments / (self.cuda_core_count * 2))
        
        self.busy = False
        return processed_fragments
    
    def get_state(self):
        """Get the current state of this SM"""
        return {
            "sm_id": self.sm_id,
            "cuda_cores": self.cuda_core_count,
            "busy": self.busy,
            "shader_type": self.current_shader.name if self.current_shader else "None",
            "utilization": self.utilization,
            "operations": self.operations_completed
        }


class GraphicsCluster:
    """Emulates a NVIDIA graphics cluster containing multiple SMs"""
    
    def __init__(self, cluster_id, sm_count, cuda_cores_per_sm):
        """Initialize a graphics cluster with the given number of SMs"""
        self.cluster_id = cluster_id
        self.sm_count = sm_count
        
        # Create the streaming multiprocessors
        self.sms = [StreamingMultiprocessor(i, cuda_cores_per_sm) for i in range(sm_count)]
        
        # Shared L1 cache (simplified)
        self.l1_cache_hit_rate = 0.8  # 80% hit rate (simplified model)
        
        # Performance tracking
        self.total_operations = 0
    
    def execute_cycle(self, workloads, cycles):
        """Execute a cycle on all SMs in the cluster
        
        Args:
            workloads: List of (shader_type, workload_size) tuples
            cycles: Number of cycles to execute
        
        Returns:
            Total operations processed
        """
        operations = 0
        
        # Distribute workloads across SMs
        sm_index = 0
        for shader_type, workload_size in workloads:
            # Find the next available SM
            attempts = 0
            while attempts < self.sm_count:
                sm = self.sms[sm_index]
                sm_index = (sm_index + 1) % self.sm_count
                
                if not sm.busy:
                    # Execute the workload on this SM
                    ops = sm.execute_workload(shader_type, workload_size, cycles)
                    operations += ops
                    break
                
                attempts += 1
        
        self.total_operations += operations
        return operations
    
    def execute_vertex_shader_batch(self, shader_program, vertex_buffer, start_vertex, vertex_count, output_processed_vertices_list, all_uniforms):
        """Execute a vertex shader on a batch of vertices, distributing across SMs
        
        Args:
            shader_program: Vertex shader program
            vertex_buffer: Input vertex buffer
            start_vertex: Starting vertex index
            vertex_count: Number of vertices to process
            output_processed_vertices_list: A list to append processed vertex data (dictionaries) to.
            all_uniforms: Dictionary of all uniforms for the current draw call.
            
        Returns:
            Number of vertices processed
        """
        if vertex_count <= 0:
            return 0
        
        # Calculate vertices per SM
        vertices_per_sm = max(1, vertex_count // self.sm_count)
        remaining_vertices = vertex_count
        current_vertex = start_vertex
        total_processed = 0
        
        # Distribute workload across SMs
        for sm in self.sms:
            if remaining_vertices <= 0:
                break
            
            # Calculate batch size for this SM
            batch_size = min(vertices_per_sm, remaining_vertices)
            
            # Execute vertex shader on this SM
            processed = sm.execute_vertex_shader(
                shader_program,
                vertex_buffer,
                current_vertex,
                batch_size,
                output_processed_vertices_list,
                all_uniforms
            )
            
            total_processed += processed
            remaining_vertices -= processed
            current_vertex += processed
            
            # If this SM couldn't process all assigned vertices, stop
            if processed < batch_size:
                break
        
        self.total_operations += total_processed
        return total_processed
    
    def execute_fragment_shader_batch(self, shader_program, fragment_count, render_target):
        """Execute a fragment shader on a batch of fragments, distributing across SMs
        
        Args:
            shader_program: Fragment shader program
            fragment_count: Number of fragments to process
            render_target: Output render target
            
        Returns:
            Number of fragments processed
        """
        if fragment_count <= 0:
            return 0
        
        # Calculate fragments per SM
        fragments_per_sm = max(1, fragment_count // self.sm_count)
        remaining_fragments = fragment_count
        total_processed = 0
        
        # Distribute workload across SMs
        for sm in self.sms:
            if remaining_fragments <= 0:
                break
            
            # Calculate batch size for this SM
            batch_size = min(fragments_per_sm, remaining_fragments)
            
            # Execute fragment shader on this SM
            processed = sm.execute_fragment_shader(
                shader_program,
                batch_size,
                render_target
            )
            
            total_processed += processed
            remaining_fragments -= processed
            
            # If this SM couldn't process all assigned fragments, stop
            if processed < batch_size:
                break
        
        self.total_operations += total_processed
        return total_processed
    
    def get_state(self):
        """Get the current state of the graphics cluster"""
        return {
            "cluster_id": self.cluster_id,
            "sm_count": self.sm_count,
            "sms": [sm.get_state() for sm in self.sms],
            "total_operations": self.total_operations
        }


class NvidiaGpu:
    """Emulates the Nvidia T239 Ampere GPU"""
    
    def __init__(self, cuda_cores=1534, sms=12, clusters=1, frequency=561.0):
        """Initialize the GPU with the given specifications"""
        self.logger = logging.getLogger("ImaginaryConsole.GPU")
        
        self.cuda_core_count = cuda_cores
        self.sm_count = sms
        self.cluster_count = clusters
        
        # Frequency in MHz
        self.frequency = frequency
        
        # Calculate CUDA cores per SM
        self.cuda_cores_per_sm = cuda_cores // sms
        
        # Create the graphics clusters
        self.clusters = [
            GraphicsCluster(i, sms // clusters, self.cuda_cores_per_sm) 
            for i in range(clusters)
        ]
        
        # GPU memory (simulated)
        self.vram_size = 0  # Will be shared with system memory
        self.vram_usage = 0
        
        # Current operation mode
        self.mode = GpuMode.IDLE
        
        # Performance tracking
        self.frame_count = 0
        self.frame_time_ms = 16.67  # Default to 60 fps
        self.last_frame_time = time.time()
        
        # Shader pipelines (simplified)
        self.vertex_queue = []
        self.fragment_queue = []
        self.compute_queue = []
        
        # Simulated frame buffer
        self.framebuffer_width = 1920
        self.framebuffer_height = 1080
        
        # Create default render targets
        self.color_buffer = RenderTarget(self.framebuffer_width, self.framebuffer_height, 'RGBA8')
        self.depth_buffer = RenderTarget(self.framebuffer_width, self.framebuffer_height, 'R8')
        
        # Graphics state
        self.current_vertex_buffer = None
        self.current_index_buffer = None
        self.current_vertex_shader = None
        self.current_fragment_shader = None
        
        # Shader programs
        self.shader_programs = {}
        
        # Texture units and currently bound textures (simplified)
        self.textures = {} # Stores Texture objects by ID
        self.texture_units = [None] * 16 # Assume 16 texture units, storing texture_id
        self.active_texture_unit = 0

        # Graphics pipeline state
        self.graphics_state = GraphicsState()
        
        # Memory access reference
        self.memory = None
        
        self.logger.info(f"Initialized NVIDIA GPU with {cuda_cores} CUDA cores, {sms} SMs, {frequency} MHz")
    
    def set_memory(self, memory):
        """Set the memory subsystem reference"""
        self.memory = memory
    
    def set_frequency(self, frequency):
        """Set the GPU frequency in MHz"""
        self.logger.info(f"Setting GPU frequency to {frequency} MHz")
        self.frequency = frequency
    
    def set_framebuffer_size(self, width, height):
        """Set the size of the framebuffer"""
        if width == self.framebuffer_width and height == self.framebuffer_height:
            return
            
        self.framebuffer_width = width
        self.framebuffer_height = height
        
        # Resize render targets
        self.color_buffer.resize(width, height)
        self.depth_buffer.resize(width, height)
        
        # Update graphics state viewport if it was tracking framebuffer size
        if (self.graphics_state.viewport_width == 0 or 
            self.graphics_state.viewport_height == 0 or 
            (self.graphics_state.viewport_width == self.framebuffer_width and 
             self.graphics_state.viewport_height == self.framebuffer_height)):
            self.graphics_state.viewport_width = width
            self.graphics_state.viewport_height = height

        self.logger.info(f"Framebuffer size set to {width}x{height}")
    
    def create_shader_program(self, shader_type, program_id, instructions=None):
        """Create a new shader program
        
        Args:
            shader_type: Type of shader (ShaderType enum)
            program_id: Unique ID for the shader program
            instructions: Optional shader instructions to load
            
        Returns:
            Shader program object
        """
        program = ShaderProgram(shader_type, program_id)
        
        if instructions:
            program.load_instructions(instructions)
        
        # Store the program
        self.shader_programs[program_id] = program
        
        self.logger.debug(f"Created shader program {program_id} of type {shader_type}")
        return program
    
    def create_vertex_buffer(self, size_bytes):
        """Create a new vertex buffer
        
        Args:
            size_bytes: Size of the buffer in bytes
            
        Returns:
            Vertex buffer object
        """
        return VertexBuffer(size_bytes)
    
    def create_index_buffer(self, size_bytes, index_type='uint16'):
        """Create a new index buffer
        
        Args:
            size_bytes: Size of the buffer in bytes
            index_type: Type of indices ('uint16' or 'uint32')
            
        Returns:
            Index buffer object
        """
        return IndexBuffer(size_bytes, index_type)
    
    def bind_vertex_buffer(self, vertex_buffer):
        """Bind a vertex buffer for rendering"""
        self.current_vertex_buffer = vertex_buffer
    
    def bind_index_buffer(self, index_buffer):
        """Bind an index buffer for rendering"""
        self.current_index_buffer = index_buffer
    
    def bind_shader_program(self, program_id):
        """Bind a shader program for rendering
        
        Args:
            program_id: ID of the shader program to bind
            
        Returns:
            True if the program was found and bound, False otherwise
        """
        if program_id not in self.shader_programs:
            return False
        
        program = self.shader_programs[program_id]
        
        if program.shader_type == ShaderType.VERTEX:
            self.current_vertex_shader = program
        elif program.shader_type == ShaderType.FRAGMENT:
            self.current_fragment_shader = program
        
        self.logger.debug(f"Bound shader program {program_id}")
        return True
    
    def create_texture(self, texture_id, width, height, depth=1, texture_format='RGBA8', data=None):
        """Create a new texture and store it.

        Args:
            texture_id: Unique ID for the texture.
            width (int): Width of the texture.
            height (int): Height of the texture.
            depth (int): Depth for 3D textures (default 1 for 2D).
            texture_format (str): Pixel format.
            data (bytes/bytearray, optional): Initial pixel data.

        Returns:
            Texture object or None if ID is already in use.
        """
        if texture_id in self.textures:
            self.logger.warning(f"Texture ID {texture_id} already exists. Cannot create new texture.")
            return None
        try:
            texture = Texture(texture_id, width, height, depth, texture_format, data)
            self.textures[texture_id] = texture
            self.logger.info(f"Created texture {texture_id} ({width}x{height}x{depth}, {texture_format})")
            return texture
        except ValueError as e:
            self.logger.error(f"Failed to create texture {texture_id}: {e}")
            return None

    def active_texture(self, unit_index):
        """Set the active texture unit.

        Args:
            unit_index (int): Index of the texture unit (0-15).
        """
        if 0 <= unit_index < len(self.texture_units):
            self.active_texture_unit = unit_index
            self.logger.debug(f"Active texture unit set to {unit_index}")
        else:
            self.logger.warning(f"Invalid texture unit index: {unit_index}")

    def bind_texture(self, texture_id, target_type='2D'): # target_type for future (e.g., CUBEMAP)
        """Bind a texture to the active texture unit.

        Args:
            texture_id: The ID of the texture to bind. Can be None to unbind.
            target_type (str): Type of texture target (e.g., '2D', '3D', 'CUBEMAP'). Currently placeholder.
        """
        if texture_id is None:
            self.texture_units[self.active_texture_unit] = None
            self.logger.debug(f"Unbound texture from unit {self.active_texture_unit}")
            return

        if texture_id not in self.textures:
            self.logger.warning(f"Texture {texture_id} not found. Cannot bind.")
            return
        
        # TODO: Validate texture target_type if it becomes relevant
        self.texture_units[self.active_texture_unit] = texture_id
        self.logger.debug(f"Bound texture {texture_id} to unit {self.active_texture_unit} (target: {target_type})")

    def set_uniform_value(self, program_id, uniform_name, value):
        """Set a uniform value for a specific shader program.

        Args:
            program_id: The ID of the shader program.
            uniform_name (str): The name of the uniform.
            value: The value to set.
        """
        if program_id not in self.shader_programs:
            self.logger.warning(f"Shader program {program_id} not found. Cannot set uniform.")
            return
        program = self.shader_programs[program_id]
        program.set_uniform(uniform_name, value)
        self.logger.debug(f"Set uniform '{uniform_name}' for shader {program_id}")

    # --- Graphics State Management Methods ---
    def set_viewport(self, x, y, width, height):
        """Set the GPU viewport.
        Args: x, y, width, height: Viewport parameters.
        """
        self.graphics_state.viewport_x = x
        self.graphics_state.viewport_y = y
        self.graphics_state.viewport_width = width
        self.graphics_state.viewport_height = height
        self.logger.debug(f"Viewport set to ({x},{y},{width},{height})")

    def enable_depth_test(self, enabled=True, func='LESS'):
        self.graphics_state.depth_test_enabled = enabled
        if enabled:
            self.graphics_state.depth_function = func
        self.logger.debug(f"Depth test {'enabled' if enabled else 'disabled'} (func: {func})")

    def enable_blend(self, enabled=True):
        self.graphics_state.blend_enabled = enabled
        self.logger.debug(f"Blending {'enabled' if enabled else 'disabled'}")

    def set_blend_function(self, src_rgb, dst_rgb, src_alpha=None, dst_alpha=None):
        self.graphics_state.blend_src_factor_rgb = src_rgb
        self.graphics_state.blend_dst_factor_rgb = dst_rgb
        self.graphics_state.blend_src_factor_alpha = src_alpha if src_alpha is not None else src_rgb
        self.graphics_state.blend_dst_factor_alpha = dst_alpha if dst_alpha is not None else dst_rgb
        self.logger.debug(f"Blend function set (RGB: {src_rgb}->{dst_rgb}, Alpha: {self.graphics_state.blend_src_factor_alpha}->{self.graphics_state.blend_dst_factor_alpha})")

    def set_blend_equation(self, eq_rgb, eq_alpha=None):
        self.graphics_state.blend_equation_rgb = eq_rgb
        self.graphics_state.blend_equation_alpha = eq_alpha if eq_alpha is not None else eq_rgb
        self.logger.debug(f"Blend equation set (RGB: {eq_rgb}, Alpha: {self.graphics_state.blend_equation_alpha})")

    def enable_cull_face(self, enabled=True, mode='BACK', front_face='CCW'):
        self.graphics_state.cull_face_enabled = enabled
        if enabled:
            self.graphics_state.cull_face_mode = mode
            self.graphics_state.front_face_winding = front_face
        self.logger.debug(f"Face culling {'enabled' if enabled else 'disabled'} (mode: {mode}, front: {front_face})")

    def enable_scissor_test(self, enabled=True, x=0, y=0, width=0, height=0):
        self.graphics_state.scissor_test_enabled = enabled
        if enabled:
            self.graphics_state.scissor_x = x
            self.graphics_state.scissor_y = y
            self.graphics_state.scissor_width = width
            self.graphics_state.scissor_height = height
        self.logger.debug(f"Scissor test {'enabled' if enabled else 'disabled'} (rect: {x},{y},{width},{height})")

    def set_color_mask(self, r, g, b, a):
        self.graphics_state.color_mask_r = r
        self.graphics_state.color_mask_g = g
        self.graphics_state.color_mask_b = b
        self.graphics_state.color_mask_a = a
        self.logger.debug(f"Color mask set to (R:{r}, G:{g}, B:{b}, A:{a})")
        
    def set_line_width(self, width):
        self.graphics_state.line_width = width
        self.logger.debug(f"Line width set to {width}")

    def _calculate_blend_factor(self, factor_str, src_color, dst_color, blend_constant_color=None):
        """Calculate a blend factor component based on its string representation.
        Args:
            factor_str (str): e.g., 'ZERO', 'ONE', 'SRC_COLOR', 'DST_COLOR',
                              'SRC_ALPHA', 'DST_ALPHA', 'CONSTANT_COLOR', etc.
            src_color (tuple): (Rs, Gs, Bs, As) current fragment output color.
            dst_color (tuple): (Rd, Gd, Bd, Ad) color currently in framebuffer.
            blend_constant_color (tuple, optional): (Rc, Gc, Bc, Ac) used for CONSTANT_COLOR/ALPHA.
        Returns:
            tuple: (factor_r, factor_g, factor_b, factor_a) numerical factors (0.0-1.0).
        """
        # Colors are assumed to be 0-255 int tuples. Normalize them for calculation.
        sR, sG, sB, sA = [c / 255.0 for c in src_color]
        dR, dG, dB, dA = [c / 255.0 for c in dst_color]
        
        # blend_constant_color is not yet a feature in GraphicsState, so pass None or implement
        # For now, CONSTANT factors will use a default if blend_constant_color is None
        default_const_color_norm = [0.0, 0.0, 0.0, 0.0] # Default to black if not provided
        if blend_constant_color:
            cR, cG, cB, cA = [c / 255.0 for c in blend_constant_color]
        else:
            cR, cG, cB, cA = default_const_color_norm

        if factor_str == 'ZERO': return (0.0, 0.0, 0.0, 0.0)
        if factor_str == 'ONE': return (1.0, 1.0, 1.0, 1.0)
        if factor_str == 'SRC_COLOR': return (sR, sG, sB, sA) # Uses source alpha for all components
        if factor_str == 'ONE_MINUS_SRC_COLOR': return (1.0 - sR, 1.0 - sG, 1.0 - sB, 1.0 - sA)
        if factor_str == 'DST_COLOR': return (dR, dG, dB, dA)
        if factor_str == 'ONE_MINUS_DST_COLOR': return (1.0 - dR, 1.0 - dG, 1.0 - dB, 1.0 - dA)
        if factor_str == 'SRC_ALPHA': return (sA, sA, sA, sA)
        if factor_str == 'ONE_MINUS_SRC_ALPHA': return (1.0 - sA, 1.0 - sA, 1.0 - sA, 1.0 - sA)
        if factor_str == 'DST_ALPHA': return (dA, dA, dA, dA)
        if factor_str == 'ONE_MINUS_DST_ALPHA': return (1.0 - dA, 1.0 - dA, 1.0 - dA, 1.0 - dA)
        if factor_str == 'CONSTANT_COLOR': return (cR, cG, cB, cR) # OpenGL uses Constant R for Alpha here usually
        if factor_str == 'ONE_MINUS_CONSTANT_COLOR': return (1.0 - cR, 1.0 - cG, 1.0 - cB, 1.0 - cR)
        if factor_str == 'CONSTANT_ALPHA': return (cA, cA, cA, cA)
        if factor_str == 'ONE_MINUS_CONSTANT_ALPHA': return (1.0 - cA, 1.0 - cA, 1.0 - cA, 1.0 - cA)
        if factor_str == 'SRC_ALPHA_SATURATE': 
            f = min(sA, 1.0 - dA)
            return (f, f, f, 1.0) # Alpha is 1.0 for this factor
        
        self.logger.warning(f"Unsupported blend factor string: {factor_str}. Defaulting to ONE.")
        return (1.0, 1.0, 1.0, 1.0)

    def clear_render_target(self, color=(0, 0, 0, 255)):
        """Clear the current render targets"""
        self.color_buffer.clear(color)
        self.depth_buffer.clear((255, 0, 0, 0))  # 1.0 for depth buffer
    
    def draw_primitives(self, primitive_type, vertex_count, start_vertex=0):
        """Draw primitives using the current vertex buffer and shaders
        
        Args:
            primitive_type: Type of primitives to draw ('triangles', 'lines', etc.)
            vertex_count: Number of vertices to draw
            start_vertex: Starting vertex index
            
        Returns:
            True if drawing was successful, False otherwise
        """
        if not self.current_vertex_buffer or not self.current_vertex_shader or not self.current_fragment_shader:
            return False
        
        # Execute drawing based on primitive type
        if primitive_type == 'triangles':
            return self._draw_triangles(vertex_count, start_vertex)
        elif primitive_type == 'lines':
            return self._draw_lines(vertex_count, start_vertex)
        else:
            return False
    
    def draw_indexed_primitives(self, primitive_type, index_count, start_index=0):
        """Draw primitives using the current index buffer and shaders
        
        Args:
            primitive_type: Type of primitives to draw ('triangles', 'lines', etc.)
            index_count: Number of indices to draw
            start_index: Starting index
            
        Returns:
            True if drawing was successful, False otherwise
        """
        if (not self.current_vertex_buffer or not self.current_index_buffer or 
            not self.current_vertex_shader or not self.current_fragment_shader):
            return False
        
        # Execute drawing based on primitive type
        if primitive_type == 'triangles':
            return self._draw_indexed_triangles(index_count, start_index)
        elif primitive_type == 'lines':
            return self._draw_indexed_lines(index_count, start_index)
        else:
            return False
    
    def _draw_triangles(self, vertex_count, start_vertex):
        """Draw triangles using the current vertex buffer"""
        valid_count = (vertex_count // 3) * 3
        if valid_count == 0: self.logger.warning("Draw triangles: Invalid vertex count"); return False
        if not self.current_vertex_buffer or not self.current_vertex_shader or not self.current_fragment_shader:
            self.logger.error("Draw triangles: Buffers or shaders not bound."); return False

        self.logger.debug(f"Drawing {valid_count // 3} triangles, {valid_count} vertices from index {start_vertex}.")
        # self.logger.debug(f"  Graphics State: ...") # Keep this if useful

        # Prepare output list for processed vertices from SMs
        sm_processed_vertices = [] 
        
        # Consolidate uniforms for the draw call (VS and FS)
        # This is a simplification. Real GPUs manage uniform buffer objects (UBOs).
        draw_call_uniforms = {}
        if self.current_vertex_shader: draw_call_uniforms.update(self.current_vertex_shader.uniforms)
        if self.current_fragment_shader: draw_call_uniforms.update(self.current_fragment_shader.uniforms) # FS uniforms might also be needed by VS in some archs, or for validation

        # Execute vertex shaders on SMs
        # The SMs will now populate sm_processed_vertices directly.
        for cluster in self.clusters:
            # Note: execute_vertex_shader_batch now takes the list to append to.
            # It should distribute work to its SMs, and each SM appends to this list.
            # The `output_buffer` argument to `execute_vertex_shader_batch` is now this list.
            cluster.execute_vertex_shader_batch(
                self.current_vertex_shader,
                self.current_vertex_buffer, 
                start_vertex, 
                valid_count, 
                sm_processed_vertices, # This list is populated by the SMs
                draw_call_uniforms
            )
        
        if len(sm_processed_vertices) != valid_count:
            self.logger.error(f"Draw triangles: Vertex processing mismatch. Expected {valid_count}, got {len(sm_processed_vertices)}.")
            # This might happen if SMs fail or boundary checks in SM stop processing early.
            return False

        # --- Rasterization and Fragment Shading (uses sm_processed_vertices) ---
        for i in range(0, len(sm_processed_vertices), 3):
            v0 = sm_processed_vertices[i]
            v1 = sm_processed_vertices[i+1]
            v2 = sm_processed_vertices[i+2]

            # ... (rest of the rasterization/fragment shading logic remains largely the same,
            #      but now it uses v0, v1, v2 from sm_processed_vertices) ...
            min_x = int(max(0, self.graphics_state.viewport_x))
            max_x = int(min(self.framebuffer_width -1, self.graphics_state.viewport_x + self.graphics_state.viewport_width -1))
            min_y = int(max(0, self.graphics_state.viewport_y))
            max_y = int(min(self.framebuffer_height -1, self.graphics_state.viewport_y + self.graphics_state.viewport_height -1))

            tri_center_x = (v0['transformed_position'][0] + v1['transformed_position'][0] + v2['transformed_position'][0]) / 3.0
            tri_center_y = (v0['transformed_position'][1] + v1['transformed_position'][1] + v2['transformed_position'][1]) / 3.0

            hack_center_x_vp = self.graphics_state.viewport_x + ( (tri_center_x * 0.5 + 0.5) * self.graphics_state.viewport_width )
            hack_center_y_vp = self.graphics_state.viewport_y + ( (1.0 - (tri_center_y * 0.5 + 0.5)) * self.graphics_state.viewport_height)

            scan_start_x = max(min_x, int(hack_center_x_vp) - 2)
            scan_end_x   = min(max_x, int(hack_center_x_vp) + 2)
            scan_start_y = max(min_y, int(hack_center_y_vp) - 2)
            scan_end_y   = min(max_y, int(hack_center_y_vp) + 2)

            for RASTER_Y in range(scan_start_y, scan_end_y + 1):
                for RASTER_X in range(scan_start_x, scan_end_x + 1):
                    interpolated_varyings = {}
                    if v0['varyings'] and 'texcoord' in v0['varyings']:
                        interpolated_varyings['texcoord'] = v0['varyings']['texcoord']
                    else:
                        interpol_u = (RASTER_X - scan_start_x) / (scan_end_x - scan_start_x + 1e-5)
                        interpol_v = (RASTER_Y - scan_start_y) / (scan_end_y - scan_start_y + 1e-5)
                        interpolated_varyings['texcoord'] = (interpol_u, interpol_v)
                    if v0['varyings'] and 'color' in v0['varyings']:
                         interpolated_varyings['color' ] = v0['varyings']['color']

                    base_color_uniform_name = "u_baseColor"
                    fragment_base_color = list(interpolated_varyings.get('color', (255,255,255,255)))
                    if self.current_fragment_shader and base_color_uniform_name in self.current_fragment_shader.uniforms:
                        uniform_val = self.current_fragment_shader.uniforms[base_color_uniform_name]
                        if isinstance(uniform_val, (list, tuple)) and len(uniform_val) == 4:
                            self.logger.debug(f"  Fragment ({RASTER_X},{RASTER_Y}): Using '{base_color_uniform_name}' {uniform_val}")
                            fragment_base_color[0] = int(fragment_base_color[0] * (uniform_val[0]/255.0))
                            fragment_base_color[1] = int(fragment_base_color[1] * (uniform_val[1]/255.0))
                            fragment_base_color[2] = int(fragment_base_color[2] * (uniform_val[2]/255.0))
                            fragment_base_color[3] = int(fragment_base_color[3] * (uniform_val[3]/255.0))
                    
                    output_color = (0,0,0,255) # Default opaque black before texturing/coloring
                    bound_texture_id = self.texture_units[0]
                    if bound_texture_id and bound_texture_id in self.textures:
                        texture_to_sample = self.textures[bound_texture_id]
                        tex_u, tex_v = interpolated_varyings.get('texcoord', (0.0, 0.0))
                        sampled_color = texture_to_sample.sample(tex_u, tex_v)
                        output_color = (
                            min(255, int(sampled_color[0] * (fragment_base_color[0]/255.0))),
                            min(255, int(sampled_color[1] * (fragment_base_color[1]/255.0))),
                            min(255, int(sampled_color[2] * (fragment_base_color[2]/255.0))),
                            min(255, int(sampled_color[3] * (fragment_base_color[3]/255.0)))
                        )
                        self.logger.debug(f"  Fragment ({RASTER_X},{RASTER_Y}): TexCoords=({tex_u:.2f},{tex_v:.2f}), Sampled={sampled_color}, BaseCol={fragment_base_color}, FinalPreBlend={output_color}")
                    else:
                        output_color = tuple(fragment_base_color) # Use vertex/base color if no texture
                        self.logger.debug(f"  Fragment ({RASTER_X},{RASTER_Y}): No texture bound or found on unit 0. Using base color: {output_color}")

                    # TODO: Depth test (if enabled) using interpolated Z and self.depth_buffer
                    
                    final_pixel_color = output_color # Start with the fragment shader's output

                    if self.graphics_state.blend_enabled:
                        dst_color_int = self.color_buffer.get_pixel(RASTER_X, RASTER_Y)
                        
                        # Factors for RGB components
                        sfactor_rgb_tuple = self._calculate_blend_factor(
                            self.graphics_state.blend_src_factor_rgb, output_color, dst_color_int)
                        dfactor_rgb_tuple = self._calculate_blend_factor(
                            self.graphics_state.blend_dst_factor_rgb, output_color, dst_color_int)
                        
                        # Factors for Alpha component (can be different)
                        sfactor_alpha_tuple = self._calculate_blend_factor(
                            self.graphics_state.blend_src_factor_alpha, output_color, dst_color_int)
                        dfactor_alpha_tuple = self._calculate_blend_factor(
                            self.graphics_state.blend_dst_factor_alpha, output_color, dst_color_int)

                        src_rgba_norm = [c / 255.0 for c in output_color]
                        dst_rgba_norm = [c / 255.0 for c in dst_color_int]
                        blended_rgba_norm = [0.0, 0.0, 0.0, 0.0]

                        if self.graphics_state.blend_equation_rgb == 'FUNC_ADD':
                            for c in range(3):
                                blended_rgba_norm[c] = (src_rgba_norm[c] * sfactor_rgb_tuple[c] + 
                                                      dst_rgba_norm[c] * dfactor_rgb_tuple[c])
                        else:
                            self.logger.warning(f"Unsupported RGB blend equation: {self.graphics_state.blend_equation_rgb}. Defaulting to FUNC_ADD.")
                            for c in range(3):
                                blended_rgba_norm[c] = (src_rgba_norm[c] * sfactor_rgb_tuple[c] + 
                                                      dst_rgba_norm[c] * dfactor_rgb_tuple[c])

                        if self.graphics_state.blend_equation_alpha == 'FUNC_ADD':
                            blended_rgba_norm[3] = (src_rgba_norm[3] * sfactor_alpha_tuple[3] + 
                                                  dst_rgba_norm[3] * dfactor_alpha_tuple[3])
                        else:
                            self.logger.warning(f"Unsupported Alpha blend equation: {self.graphics_state.blend_equation_alpha}. Defaulting to FUNC_ADD.")
                            blended_rgba_norm[3] = (src_rgba_norm[3] * sfactor_alpha_tuple[3] + 
                                                  dst_rgba_norm[3] * dfactor_alpha_tuple[3])
                        
                        final_pixel_color = tuple(
                            max(0, min(255, int(round(c * 255.0)))) for c in blended_rgba_norm
                        )
                        self.logger.debug(f"  Blend @({RASTER_X},{RASTER_Y}): Src={output_color}, Dst={dst_color_int}, sRGBFac={tuple(round(f,2) for f in sfactor_rgb_tuple[:3])}, dRGBFac={tuple(round(f,2) for f in dfactor_rgb_tuple[:3])}, sAFac={sfactor_alpha_tuple[3]:.2f}, dAFac={dfactor_alpha_tuple[3]:.2f} -> Blended={final_pixel_color}")

                    if self.graphics_state.color_mask_r and \
                       self.graphics_state.color_mask_g and \
                       self.graphics_state.color_mask_b and \
                       self.graphics_state.color_mask_a:
                        self.color_buffer.set_pixel(RASTER_X, RASTER_Y, final_pixel_color)

        # The old SM execution batch calls are no longer relevant here as processing is inline.
        return True
    
    def _draw_lines(self, vertex_count, start_vertex):
        """Draw lines using the current vertex buffer"""
        valid_vertex_count = (vertex_count // 2) * 2
        
        if valid_vertex_count == 0:
            self.logger.warning("Draw lines: Invalid vertex count (must be multiple of 2)")
            return False
        
        if not self.current_vertex_buffer or not self.current_vertex_shader or not self.current_fragment_shader:
            self.logger.error("Draw lines: Vertex buffer, vertex shader, or fragment shader not bound.")
            return False

        self.logger.debug(f"Drawing {valid_vertex_count // 2} lines, using {valid_vertex_count} vertices from index {start_vertex}. Line width: {self.graphics_state.line_width}")
        
        sm_processed_vertices = []
        draw_call_uniforms = {}
        if self.current_vertex_shader: draw_call_uniforms.update(self.current_vertex_shader.uniforms)
        if self.current_fragment_shader: draw_call_uniforms.update(self.current_fragment_shader.uniforms)

        for cluster in self.clusters:
            cluster.execute_vertex_shader_batch(
                self.current_vertex_shader,
                self.current_vertex_buffer, 
                start_vertex, 
                valid_vertex_count, 
                sm_processed_vertices,
                draw_call_uniforms
            )
        
        if len(sm_processed_vertices) != valid_vertex_count:
            self.logger.error(f"Draw lines: Vertex processing mismatch. Expected {valid_vertex_count}, got {len(sm_processed_vertices)}.")
            return False

        for i in range(0, len(sm_processed_vertices), 2):
            v0 = sm_processed_vertices[i]
            v1 = sm_processed_vertices[i+1]
            
            min_x_vp = self.graphics_state.viewport_x
            max_x_vp = self.graphics_state.viewport_x + self.graphics_state.viewport_width -1
            min_y_vp = self.graphics_state.viewport_y
            max_y_vp = self.graphics_state.viewport_y + self.graphics_state.viewport_height -1

            x0_vp = min_x_vp + ((v0['transformed_position'][0] * 0.5 + 0.5) * self.graphics_state.viewport_width)
            y0_vp = min_y_vp + ((1.0 - (v0['transformed_position'][1] * 0.5 + 0.5)) * self.graphics_state.viewport_height)
            x1_vp = min_x_vp + ((v1['transformed_position'][0] * 0.5 + 0.5) * self.graphics_state.viewport_width)
            y1_vp = min_y_vp + ((1.0 - (v1['transformed_position'][1] * 0.5 + 0.5)) * self.graphics_state.viewport_height)

            num_steps = 5 
            for step in range(num_steps + 1):
                t = step / num_steps
                RASTER_X = int(x0_vp * (1-t) + x1_vp * t)
                RASTER_Y = int(y0_vp * (1-t) + y1_vp * t)

                if not (min_x_vp <= RASTER_X <= max_x_vp and min_y_vp <= RASTER_Y <= max_y_vp):
                    continue 

                interpolated_varyings = {}
                if v0['varyings'] and 'texcoord' in v0['varyings'] and v1['varyings'] and 'texcoord' in v1['varyings']:
                    tc0, tc1 = v0['varyings']['texcoord'], v1['varyings']['texcoord']
                    interpolated_varyings['texcoord'] = (tc0[0]*(1-t) + tc1[0]*t, tc0[1]*(1-t) + tc1[1]*t) if tc0 and tc1 else (t, 0.0)
                else: interpolated_varyings['texcoord'] = (t, 0.0) 
                
                if v0['varyings'] and 'color' in v0['varyings'] and v1['varyings'] and 'color' in v1['varyings']:
                    col0, col1 = v0['varyings']['color'], v1['varyings']['color']
                    interpolated_varyings['color'] = tuple(int(c0*(1-t) + c1*t) for c0,c1 in zip(col0,col1)) if col0 and col1 else (255,255,255,255)
                else: interpolated_varyings['color'] = (255,255,255,255)

                base_color_uniform_name = "u_baseColor"
                fragment_base_color = list(interpolated_varyings.get('color', (255,255,255,255)))
                if self.current_fragment_shader and base_color_uniform_name in self.current_fragment_shader.uniforms:
                    uniform_val = self.current_fragment_shader.uniforms[base_color_uniform_name]
                    if isinstance(uniform_val, (list, tuple)) and len(uniform_val) == 4:
                        fragment_base_color = [int(fc * (uv/255.0)) for fc, uv in zip(fragment_base_color, uniform_val)]
                
                output_color = tuple(fragment_base_color)
                bound_texture_id = self.texture_units[0]
                if bound_texture_id and bound_texture_id in self.textures:
                    texture_to_sample = self.textures[bound_texture_id]
                    tex_u, tex_v = interpolated_varyings.get('texcoord', (0.0, 0.0))
                    sampled_color = texture_to_sample.sample(tex_u, tex_v)
                    output_color = tuple(min(255, int(sc * (fc/255.0))) for sc, fc in zip(sampled_color, fragment_base_color))
                    self.logger.debug(f"  LineFrag ({RASTER_X},{RASTER_Y}): Tex=({tex_u:.2f},{tex_v:.2f}), Sampled={sampled_color}, Base={fragment_base_color}, FinalPreBlend={output_color}")
                else: 
                    self.logger.debug(f"  LineFrag ({RASTER_X},{RASTER_Y}): No tex bound. Base={fragment_base_color}, FinalPreBlend={output_color}")

                final_pixel_color = output_color
                if self.graphics_state.blend_enabled:
                    dst_color_int = self.color_buffer.get_pixel(RASTER_X, RASTER_Y)
                    sfactor_rgb_tuple = self._calculate_blend_factor(self.graphics_state.blend_src_factor_rgb, output_color, dst_color_int)
                    dfactor_rgb_tuple = self._calculate_blend_factor(self.graphics_state.blend_dst_factor_rgb, output_color, dst_color_int)
                    sfactor_alpha_tuple = self._calculate_blend_factor(self.graphics_state.blend_src_factor_alpha, output_color, dst_color_int)
                    dfactor_alpha_tuple = self._calculate_blend_factor(self.graphics_state.blend_dst_factor_alpha, output_color, dst_color_int)
                    src_rgba_norm = [c / 255.0 for c in output_color]
                    dst_rgba_norm = [c / 255.0 for c in dst_color_int]
                    blended_rgba_norm = [0.0, 0.0, 0.0, 0.0]
                    if self.graphics_state.blend_equation_rgb == 'FUNC_ADD':
                        for c_idx in range(3):
                            blended_rgba_norm[c_idx] = (src_rgba_norm[c_idx] * sfactor_rgb_tuple[c_idx] + dst_rgba_norm[c_idx] * dfactor_rgb_tuple[c_idx])
                    else:
                        self.logger.warning(f"Unsupported RGB blend equation: {self.graphics_state.blend_equation_rgb}. Defaulting to FUNC_ADD.")
                        for c_idx in range(3):
                            blended_rgba_norm[c_idx] = (src_rgba_norm[c_idx] * sfactor_rgb_tuple[c_idx] + dst_rgba_norm[c_idx] * dfactor_rgb_tuple[c_idx])
                    if self.graphics_state.blend_equation_alpha == 'FUNC_ADD':
                        blended_rgba_norm[3] = (src_rgba_norm[3] * sfactor_alpha_tuple[3] + dst_rgba_norm[3] * dfactor_alpha_tuple[3])
                    else:
                        self.logger.warning(f"Unsupported Alpha blend equation: {self.graphics_state.blend_equation_alpha}. Defaulting to FUNC_ADD.")
                        blended_rgba_norm[3] = (src_rgba_norm[3] * sfactor_alpha_tuple[3] + dst_rgba_norm[3] * dfactor_alpha_tuple[3])
                    final_pixel_color = tuple(max(0, min(255, int(round(c * 255.0)))) for c in blended_rgba_norm)
                    self.logger.debug(f"  LineBlend @({RASTER_X},{RASTER_Y}): Src={output_color}, Dst={dst_color_int}, sRGBFac={tuple(round(f,2) for f in sfactor_rgb_tuple[:3])}, dRGBFac={tuple(round(f,2) for f in dfactor_rgb_tuple[:3])}, sAFac={sfactor_alpha_tuple[3]:.2f}, dAFac={dfactor_alpha_tuple[3]:.2f} -> Blended={final_pixel_color}")

                if self.graphics_state.color_mask_r and self.graphics_state.color_mask_g and \
                   self.graphics_state.color_mask_b and self.graphics_state.color_mask_a:
                    self.color_buffer.set_pixel(RASTER_X, RASTER_Y, final_pixel_color)
        return True
    
    def _draw_indexed_triangles(self, index_count, start_index):
        """Draw triangles using the current index buffer"""
        valid_index_count = (index_count // 3) * 3
        if valid_index_count == 0: self.logger.warning("Draw indexed triangles: Invalid index count"); return False
        if (not self.current_vertex_buffer or not self.current_index_buffer or 
            not self.current_vertex_shader or not self.current_fragment_shader):
            self.logger.error("Draw indexed triangles: Buffers or shaders not bound."); return False

        self.logger.debug(f"Drawing {valid_index_count // 3} indexed triangles, using {valid_index_count} indices from IB start {start_index}.")

        unique_vertex_indices_from_ib = sorted(list(set(self.current_index_buffer.get_index(start_index + i) for i in range(valid_index_count))))
        
        processed_unique_vertices = {} 
        draw_call_uniforms = {}
        if self.current_vertex_shader: draw_call_uniforms.update(self.current_vertex_shader.uniforms)
        if self.current_fragment_shader: draw_call_uniforms.update(self.current_fragment_shader.uniforms)

        temp_processed_map = {} 
        for vb_idx in unique_vertex_indices_from_ib:
            if self.clusters and self.clusters[0].sms:
                sm_to_use = self.clusters[0].sms[0] 
                single_vertex_processed_list = []
                sm_to_use.execute_vertex_shader(
                    self.current_vertex_shader,
                    self.current_vertex_buffer,
                    vb_idx, 1,
                    single_vertex_processed_list,
                    draw_call_uniforms
                )
                if single_vertex_processed_list:
                    temp_processed_map[vb_idx] = single_vertex_processed_list[0]
                else:
                    self.logger.error(f"Failed to process unique vertex index {vb_idx} via SM.")
                    return False 
            else:
                self.logger.error("No GPU clusters/SMs configured for indexed draw processing.")
                return False
        
        processed_unique_vertices = temp_processed_map

        if len(processed_unique_vertices) != len(unique_vertex_indices_from_ib):
            self.logger.error(f"Draw indexed triangles: Unique vertex processing failed. Expected {len(unique_vertex_indices_from_ib)}, Got {len(processed_unique_vertices)}")
            return False

        for i in range(0, valid_index_count, 3):
            idx0 = self.current_index_buffer.get_index(start_index + i)
            idx1 = self.current_index_buffer.get_index(start_index + i + 1)
            idx2 = self.current_index_buffer.get_index(start_index + i + 2)
            
            if not all(idx in processed_unique_vertices for idx in [idx0, idx1, idx2]):
                self.logger.error(f"Draw indexed triangles: Vertex index not found in processed map. Indices: {idx0},{idx1},{idx2}.")
                return False

            v0 = processed_unique_vertices[idx0]
            v1 = processed_unique_vertices[idx1]
            v2 = processed_unique_vertices[idx2]

            min_x = int(max(0, self.graphics_state.viewport_x))
            max_x = int(min(self.framebuffer_width -1, self.graphics_state.viewport_x + self.graphics_state.viewport_width -1))
            min_y = int(max(0, self.graphics_state.viewport_y))
            max_y = int(min(self.framebuffer_height -1, self.graphics_state.viewport_y + self.graphics_state.viewport_height -1))

            tri_center_x = (v0['transformed_position'][0] + v1['transformed_position'][0] + v2['transformed_position'][0]) / 3.0
            tri_center_y = (v0['transformed_position'][1] + v1['transformed_position'][1] + v2['transformed_position'][1]) / 3.0
            hack_center_x_vp = self.graphics_state.viewport_x + ( (tri_center_x * 0.5 + 0.5) * self.graphics_state.viewport_width )
            hack_center_y_vp = self.graphics_state.viewport_y + ( (1.0 - (tri_center_y * 0.5 + 0.5)) * self.graphics_state.viewport_height)
            scan_start_x = max(min_x, int(hack_center_x_vp) - 2)
            scan_end_x   = min(max_x, int(hack_center_x_vp) + 2)
            scan_start_y = max(min_y, int(hack_center_y_vp) - 2)
            scan_end_y   = min(max_y, int(hack_center_y_vp) + 2)

            for RASTER_Y in range(scan_start_y, scan_end_y + 1):
                for RASTER_X in range(scan_start_x, scan_end_x + 1):
                    interpolated_varyings = {}
                    if v0['varyings'] and 'texcoord' in v0['varyings']:
                        interpolated_varyings['texcoord'] = v0['varyings']['texcoord']
                    else:
                        interpol_u = (RASTER_X - scan_start_x) / (scan_end_x - scan_start_x + 1e-5)
                        interpol_v = (RASTER_Y - scan_start_y) / (scan_end_y - scan_start_y + 1e-5)
                        interpolated_varyings['texcoord'] = (interpol_u, interpol_v)
                    if v0['varyings'] and 'color' in v0['varyings']:
                         interpolated_varyings['color' ] = v0['varyings']['color']

                    base_color_uniform_name = "u_baseColor"
                    fragment_base_color = list(interpolated_varyings.get('color', (255,255,255,255)))
                    if self.current_fragment_shader and base_color_uniform_name in self.current_fragment_shader.uniforms:
                        uniform_val = self.current_fragment_shader.uniforms[base_color_uniform_name]
                        if isinstance(uniform_val, (list, tuple)) and len(uniform_val) == 4:
                            fragment_base_color[0] = int(fragment_base_color[0] * (uniform_val[0]/255.0))
                            fragment_base_color[1] = int(fragment_base_color[1] * (uniform_val[1]/255.0))
                            fragment_base_color[2] = int(fragment_base_color[2] * (uniform_val[2]/255.0))
                            fragment_base_color[3] = int(fragment_base_color[3] * (uniform_val[3]/255.0))
                    
                    output_color = tuple(fragment_base_color) # Start with base color
                    bound_texture_id = self.texture_units[0]
                    if bound_texture_id and bound_texture_id in self.textures:
                        texture_to_sample = self.textures[bound_texture_id]
                        tex_u, tex_v = interpolated_varyings.get('texcoord', (0.0, 0.0))
                        sampled_color = texture_to_sample.sample(tex_u, tex_v)
                        output_color = (
                            min(255, int(sampled_color[0] * (fragment_base_color[0]/255.0))),\
                            min(255, int(sampled_color[1] * (fragment_base_color[1]/255.0))),\
                            min(255, int(sampled_color[2] * (fragment_base_color[2]/255.0))),\
                            min(255, int(sampled_color[3] * (fragment_base_color[3]/255.0)))\
                        )
                        self.logger.debug(f"  IdxTriFrag ({RASTER_X},{RASTER_Y}): Tex=({tex_u:.2f},{tex_v:.2f}), Sampled={sampled_color}, Base={fragment_base_color}, FinalPreBlend={output_color}")
                    else:
                        self.logger.debug(f"  IdxTriFrag ({RASTER_X},{RASTER_Y}): No tex. Base={fragment_base_color}, FinalPreBlend={output_color}")

                    final_pixel_color = output_color
                    if self.graphics_state.blend_enabled:
                        dst_color_int = self.color_buffer.get_pixel(RASTER_X, RASTER_Y)
                        sfactor_rgb_tuple = self._calculate_blend_factor(self.graphics_state.blend_src_factor_rgb, output_color, dst_color_int)
                        dfactor_rgb_tuple = self._calculate_blend_factor(self.graphics_state.blend_dst_factor_rgb, output_color, dst_color_int)
                        sfactor_alpha_tuple = self._calculate_blend_factor(self.graphics_state.blend_src_factor_alpha, output_color, dst_color_int)
                        dfactor_alpha_tuple = self._calculate_blend_factor(self.graphics_state.blend_dst_factor_alpha, output_color, dst_color_int)
                        src_rgba_norm = [c / 255.0 for c in output_color]
                        dst_rgba_norm = [c / 255.0 for c in dst_color_int]
                        blended_rgba_norm = [0.0, 0.0, 0.0, 0.0]
                        if self.graphics_state.blend_equation_rgb == 'FUNC_ADD':
                            for c_idx in range(3):
                                blended_rgba_norm[c_idx] = (src_rgba_norm[c_idx] * sfactor_rgb_tuple[c_idx] + dst_rgba_norm[c_idx] * dfactor_rgb_tuple[c_idx])
                        else:
                            self.logger.warning(f"Unsupported RGB blend equation: {self.graphics_state.blend_equation_rgb}. Defaulting to FUNC_ADD.")
                            for c_idx in range(3):
                                blended_rgba_norm[c_idx] = (src_rgba_norm[c_idx] * sfactor_rgb_tuple[c_idx] + dst_rgba_norm[c_idx] * dfactor_rgb_tuple[c_idx])
                        if self.graphics_state.blend_equation_alpha == 'FUNC_ADD':
                            blended_rgba_norm[3] = (src_rgba_norm[3] * sfactor_alpha_tuple[3] + dst_rgba_norm[3] * dfactor_alpha_tuple[3])
                        else:
                            self.logger.warning(f"Unsupported Alpha blend equation: {self.graphics_state.blend_equation_alpha}. Defaulting to FUNC_ADD.")
                            blended_rgba_norm[3] = (src_rgba_norm[3] * sfactor_alpha_tuple[3] + dst_rgba_norm[3] * dfactor_alpha_tuple[3])
                        final_pixel_color = tuple(max(0, min(255, int(round(c * 255.0)))) for c in blended_rgba_norm)
                        self.logger.debug(f"  IdxTriBlend @({RASTER_X},{RASTER_Y}): Src={output_color}, Dst={dst_color_int}, sRGBFac={tuple(round(f,2) for f in sfactor_rgb_tuple[:3])}, dRGBFac={tuple(round(f,2) for f in dfactor_rgb_tuple[:3])}, sAFac={sfactor_alpha_tuple[3]:.2f}, dAFac={dfactor_alpha_tuple[3]:.2f} -> Blended={final_pixel_color}")

                    if self.graphics_state.color_mask_r and \
                       self.graphics_state.color_mask_g and \
                       self.graphics_state.color_mask_b and \
                       self.graphics_state.color_mask_a:
                        self.color_buffer.set_pixel(RASTER_X, RASTER_Y, final_pixel_color)
        return True
    
    def _draw_indexed_lines(self, index_count, start_index):
        """Draw lines using the current index buffer"""
        valid_index_count = (index_count // 2) * 2
        if valid_index_count == 0: self.logger.warning("Draw indexed lines: Invalid index count"); return False
        if (not self.current_vertex_buffer or not self.current_index_buffer or
            not self.current_vertex_shader or not self.current_fragment_shader):
            self.logger.error("Draw indexed lines: Buffers or shaders not bound."); return False

        self.logger.debug(f"Drawing {valid_index_count // 2} indexed lines, using {valid_index_count} indices from IB start {start_index}.")

        unique_vertex_indices_from_ib = sorted(list(set(self.current_index_buffer.get_index(start_index + i) for i in range(valid_index_count))))
        
        processed_unique_vertices = {} 
        draw_call_uniforms = {}
        if self.current_vertex_shader: draw_call_uniforms.update(self.current_vertex_shader.uniforms)
        if self.current_fragment_shader: draw_call_uniforms.update(self.current_fragment_shader.uniforms)

        for vb_idx in unique_vertex_indices_from_ib:
            if self.clusters and self.clusters[0].sms:
                sm_to_use = self.clusters[0].sms[0]
                single_vertex_processed_list = []
                sm_to_use.execute_vertex_shader(
                    self.current_vertex_shader,
                    self.current_vertex_buffer,
                    vb_idx, 1,
                    single_vertex_processed_list,
                    draw_call_uniforms
                )
                if single_vertex_processed_list:
                    processed_unique_vertices[vb_idx] = single_vertex_processed_list[0]
                else:
                    self.logger.error(f"Failed to process unique vertex index {vb_idx} for indexed lines.")
                    return False
            else:
                self.logger.error("No GPU clusters/SMs for indexed line processing."); return False
        
        if len(processed_unique_vertices) != len(unique_vertex_indices_from_ib):
            self.logger.error(f"Draw indexed lines: Unique vertex processing failed. Expected {len(unique_vertex_indices_from_ib)}, Got {len(processed_unique_vertices)}")
            return False

        for i in range(0, valid_index_count, 2):
            idx0 = self.current_index_buffer.get_index(start_index + i)
            idx1 = self.current_index_buffer.get_index(start_index + i + 1)

            if not all(idx in processed_unique_vertices for idx in [idx0, idx1]):
                self.logger.error(f"Draw indexed lines: Vertex index not found in processed map. Indices: {idx0},{idx1}.")
                return False

            v0 = processed_unique_vertices[idx0]
            v1 = processed_unique_vertices[idx1]

            min_x_vp = self.graphics_state.viewport_x
            max_x_vp = self.graphics_state.viewport_x + self.graphics_state.viewport_width -1
            min_y_vp = self.graphics_state.viewport_y
            max_y_vp = self.graphics_state.viewport_y + self.graphics_state.viewport_height -1

            x0_vp = min_x_vp + ((v0['transformed_position'][0] * 0.5 + 0.5) * self.graphics_state.viewport_width)
            y0_vp = min_y_vp + ((1.0 - (v0['transformed_position'][1] * 0.5 + 0.5)) * self.graphics_state.viewport_height)
            x1_vp = min_x_vp + ((v1['transformed_position'][0] * 0.5 + 0.5) * self.graphics_state.viewport_width)
            y1_vp = min_y_vp + ((1.0 - (v1['transformed_position'][1] * 0.5 + 0.5)) * self.graphics_state.viewport_height)

            num_steps = 5 
            for step in range(num_steps + 1):
                t = step / num_steps
                RASTER_X = int(x0_vp * (1-t) + x1_vp * t)
                RASTER_Y = int(y0_vp * (1-t) + y1_vp * t)

                if not (min_x_vp <= RASTER_X <= max_x_vp and min_y_vp <= RASTER_Y <= max_y_vp):
                    continue 

                interpolated_varyings = {}
                if v0['varyings'] and 'texcoord' in v0['varyings'] and v1['varyings'] and 'texcoord' in v1['varyings']:
                    tc0, tc1 = v0['varyings']['texcoord'], v1['varyings']['texcoord']
                    interpolated_varyings['texcoord'] = (tc0[0]*(1-t) + tc1[0]*t, tc0[1]*(1-t) + tc1[1]*t) if tc0 and tc1 else (t, 0.0)
                else: interpolated_varyings['texcoord'] = (t, 0.0) 
                
                if v0['varyings'] and 'color' in v0['varyings'] and v1['varyings'] and 'color' in v1['varyings']:
                    col0, col1 = v0['varyings']['color'], v1['varyings']['color']
                    interpolated_varyings['color'] = tuple(int(c0*(1-t) + c1*t) for c0,c1 in zip(col0,col1)) if col0 and col1 else (255,255,255,255)
                else: interpolated_varyings['color'] = (255,255,255,255)

                base_color_uniform_name = "u_baseColor"
                fragment_base_color = list(interpolated_varyings.get('color', (255,255,255,255)))
                if self.current_fragment_shader and base_color_uniform_name in self.current_fragment_shader.uniforms:
                    uniform_val = self.current_fragment_shader.uniforms[base_color_uniform_name]
                    if isinstance(uniform_val, (list, tuple)) and len(uniform_val) == 4:
                        fragment_base_color = [int(fc * (uv/255.0)) for fc, uv in zip(fragment_base_color, uniform_val)]
                
                output_color = tuple(fragment_base_color)
                bound_texture_id = self.texture_units[0]
                if bound_texture_id and bound_texture_id in self.textures:
                    texture_to_sample = self.textures[bound_texture_id]
                    tex_u, tex_v = interpolated_varyings.get('texcoord', (0.0, 0.0))
                    sampled_color = texture_to_sample.sample(tex_u, tex_v)
                    output_color = tuple(min(255, int(sc * (fc/255.0))) for sc, fc in zip(sampled_color, fragment_base_color))
                    self.logger.debug(f"  IdxLineFrag ({RASTER_X},{RASTER_Y}): Tex=({tex_u:.2f},{tex_v:.2f}), Sampled={sampled_color}, Base={fragment_base_color}, FinalPreBlend={output_color}")
                else: 
                    self.logger.debug(f"  IdxLineFrag ({RASTER_X},{RASTER_Y}): No tex bound. Base={fragment_base_color}, FinalPreBlend={output_color}")

                final_pixel_color = output_color
                if self.graphics_state.blend_enabled:
                    dst_color_int = self.color_buffer.get_pixel(RASTER_X, RASTER_Y)
                    sfactor_rgb_tuple = self._calculate_blend_factor(self.graphics_state.blend_src_factor_rgb, output_color, dst_color_int)
                    dfactor_rgb_tuple = self._calculate_blend_factor(self.graphics_state.blend_dst_factor_rgb, output_color, dst_color_int)
                    sfactor_alpha_tuple = self._calculate_blend_factor(self.graphics_state.blend_src_factor_alpha, output_color, dst_color_int)
                    dfactor_alpha_tuple = self._calculate_blend_factor(self.graphics_state.blend_dst_factor_alpha, output_color, dst_color_int)
                    src_rgba_norm = [c / 255.0 for c in output_color]
                    dst_rgba_norm = [c / 255.0 for c in dst_color_int]
                    blended_rgba_norm = [0.0, 0.0, 0.0, 0.0]
                    if self.graphics_state.blend_equation_rgb == 'FUNC_ADD':
                        for c_idx in range(3):
                            blended_rgba_norm[c_idx] = (src_rgba_norm[c_idx] * sfactor_rgb_tuple[c_idx] + dst_rgba_norm[c_idx] * dfactor_rgb_tuple[c_idx])
                    else:
                        self.logger.warning(f"Unsupported RGB blend equation: {self.graphics_state.blend_equation_rgb}. Defaulting to FUNC_ADD.")
                        for c_idx in range(3):
                            blended_rgba_norm[c_idx] = (src_rgba_norm[c_idx] * sfactor_rgb_tuple[c_idx] + dst_rgba_norm[c_idx] * dfactor_rgb_tuple[c_idx])
                    if self.graphics_state.blend_equation_alpha == 'FUNC_ADD':
                        blended_rgba_norm[3] = (src_rgba_norm[3] * sfactor_alpha_tuple[3] + dst_rgba_norm[3] * dfactor_alpha_tuple[3])
                    else:
                        self.logger.warning(f"Unsupported Alpha blend equation: {self.graphics_state.blend_equation_alpha}. Defaulting to FUNC_ADD.")
                        blended_rgba_norm[3] = (src_rgba_norm[3] * sfactor_alpha_tuple[3] + dst_rgba_norm[3] * dfactor_alpha_tuple[3])
                    final_pixel_color = tuple(max(0, min(255, int(round(c * 255.0)))) for c in blended_rgba_norm)
                    self.logger.debug(f"  IdxLineBlend @({RASTER_X},{RASTER_Y}): Src={output_color}, Dst={dst_color_int}, sRGBFac={tuple(round(f,2) for f in sfactor_rgb_tuple[:3])}, dRGBFac={tuple(round(f,2) for f in dfactor_rgb_tuple[:3])}, sAFac={sfactor_alpha_tuple[3]:.2f}, dAFac={dfactor_alpha_tuple[3]:.2f} -> Blended={final_pixel_color}")

                if self.graphics_state.color_mask_r and self.graphics_state.color_mask_g and \
                   self.graphics_state.color_mask_b and self.graphics_state.color_mask_a:
                    self.color_buffer.set_pixel(RASTER_X, RASTER_Y, final_pixel_color)
        return True
    
    def draw_rect(self, x, y, width, height, color=(255, 255, 255, 255)):
        """Draw a filled rectangle directly to the framebuffer, respecting viewport, scissor, color mask, and blending.
        
        Args:
            x, y: Top-left corner coordinates (framebuffer space).
            width, height: Rectangle dimensions.
            color: RGBA color tuple.
            
        Returns:
            True if successful.
        """
        self.logger.debug(f"draw_rect: Input rect ({x},{y},{width},{height}), Color: {color}")

        # Apply Viewport clipping
        # Viewport defines the drawable area on the framebuffer
        vp_x, vp_y = self.graphics_state.viewport_x, self.graphics_state.viewport_y
        vp_w, vp_h = self.graphics_state.viewport_width, self.graphics_state.viewport_height

        # Calculate intersection of input rect with viewport
        draw_x = max(x, vp_x)
        draw_y = max(y, vp_y)
        rect_right = x + width
        rect_bottom = y + height
        vp_right = vp_x + vp_w
        vp_bottom = vp_y + vp_h
        
        draw_right = min(rect_right, vp_right)
        draw_bottom = min(rect_bottom, vp_bottom)

        draw_width = draw_right - draw_x
        draw_height = draw_bottom - draw_y

        if draw_width <= 0 or draw_height <= 0:
            self.logger.debug("draw_rect: Rect is entirely outside viewport.")
            return True # Nothing to draw

        self.logger.debug(f"draw_rect: Viewport clipped rect ({draw_x},{draw_y},{draw_width},{draw_height})")

        # Apply Scissor Test if enabled
        if self.graphics_state.scissor_test_enabled:
            sc_x, sc_y = self.graphics_state.scissor_x, self.graphics_state.scissor_y
            sc_w, sc_h = self.graphics_state.scissor_width, self.graphics_state.scissor_height

            # Further clip the draw_rect with scissor_rect
            prev_draw_x, prev_draw_y = draw_x, draw_y
            draw_x = max(draw_x, sc_x)
            draw_y = max(draw_y, sc_y)
            
            draw_right = min(prev_draw_x + draw_width, sc_x + sc_w)
            draw_bottom = min(prev_draw_y + draw_height, sc_y + sc_h)

            draw_width = draw_right - draw_x
            draw_height = draw_bottom - draw_y

            if draw_width <= 0 or draw_height <= 0:
                self.logger.debug("draw_rect: Rect is entirely outside scissor box.")
                return True # Nothing to draw
            self.logger.debug(f"draw_rect: Scissor clipped rect ({draw_x},{draw_y},{draw_width},{draw_height})")

        # Iterate and set pixels, respecting color mask and blending
        # Ensure coordinates are within framebuffer bounds
        final_x_start = max(0, draw_x)
        final_y_start = max(0, draw_y)
        final_x_end = min(self.framebuffer_width, draw_x + draw_width)
        final_y_end = min(self.framebuffer_height, draw_y + draw_height)

        for RASTER_Y in range(final_y_start, final_y_end):
            for RASTER_X in range(final_x_start, final_x_end):
                # Current color to draw (source color for blending)
                src_color_for_pixel = list(color)

                # Apply Color Mask to the source color
                if not self.graphics_state.color_mask_r: src_color_for_pixel[0] = 0 # Effectively, a masked component from src won't contribute
                if not self.graphics_state.color_mask_g: src_color_for_pixel[1] = 0
                if not self.graphics_state.color_mask_b: src_color_for_pixel[2] = 0
                if not self.graphics_state.color_mask_a: src_color_for_pixel[3] = 0
                
                # If all components are masked out from the source, and no blending, nothing to do.
                # If blending, a masked source might still affect dest via some blend factors.
                # For simplicity here, if source is fully masked by color_mask (becomes 0,0,0,0),
                # and no blending, we can skip. With blending, it depends on factors.

                final_pixel_color_tuple = tuple(src_color_for_pixel)

                if self.graphics_state.blend_enabled:
                    dst_color_int = self.color_buffer.get_pixel(RASTER_X, RASTER_Y)
                    
                    # Use the masked source color for blending calculations
                    sfactor_rgb_tuple = self._calculate_blend_factor(self.graphics_state.blend_src_factor_rgb, final_pixel_color_tuple, dst_color_int)
                    dfactor_rgb_tuple = self._calculate_blend_factor(self.graphics_state.blend_dst_factor_rgb, final_pixel_color_tuple, dst_color_int)
                    sfactor_alpha_tuple = self._calculate_blend_factor(self.graphics_state.blend_src_factor_alpha, final_pixel_color_tuple, dst_color_int)
                    dfactor_alpha_tuple = self._calculate_blend_factor(self.graphics_state.blend_dst_factor_alpha, final_pixel_color_tuple, dst_color_int)

                    src_rgba_norm = [c / 255.0 for c in final_pixel_color_tuple] # Use potentially masked source
                    dst_rgba_norm = [c / 255.0 for c in dst_color_int]
                    blended_rgba_norm = [0.0, 0.0, 0.0, 0.0]

                    if self.graphics_state.blend_equation_rgb == 'FUNC_ADD':
                        for c in range(3):
                            # If color mask for this component on framebuffer is false, dest component is not read/used as per some GL specs.
                            # However, simpler to apply color mask at the very end of writing.
                            # Here, we use the calculated factors as is.
                            blended_rgba_norm[c] = (src_rgba_norm[c] * sfactor_rgb_tuple[c] + dst_rgba_norm[c] * dfactor_rgb_tuple[c])
                    else:
                        self.logger.warning(f"draw_rect: Unsupported RGB blend equation {self.graphics_state.blend_equation_rgb}. Defaulting to FUNC_ADD.")
                        for c in range(3):
                            blended_rgba_norm[c] = (src_rgba_norm[c] * sfactor_rgb_tuple[c] + dst_rgba_norm[c] * dfactor_rgb_tuple[c])
                    
                    if self.graphics_state.blend_equation_alpha == 'FUNC_ADD':
                         blended_rgba_norm[3] = (src_rgba_norm[3] * sfactor_alpha_tuple[3] + dst_rgba_norm[3] * dfactor_alpha_tuple[3])
                    else:
                        self.logger.warning(f"draw_rect: Unsupported Alpha blend equation {self.graphics_state.blend_equation_alpha}. Defaulting to FUNC_ADD.")
                        blended_rgba_norm[3] = (src_rgba_norm[3] * sfactor_alpha_tuple[3] + dst_rgba_norm[3] * dfactor_alpha_tuple[3])
                    
                    final_pixel_color_tuple = tuple(max(0, min(255, int(round(c * 255.0)))) for c in blended_rgba_norm)
                
                # Get original pixel if a component is masked for write
                # Color mask applies to the *write* operation.
                current_fb_color = self.color_buffer.get_pixel(RASTER_X, RASTER_Y)
                write_color_r = final_pixel_color_tuple[0] if self.graphics_state.color_mask_r else current_fb_color[0]
                write_color_g = final_pixel_color_tuple[1] if self.graphics_state.color_mask_g else current_fb_color[1]
                write_color_b = final_pixel_color_tuple[2] if self.graphics_state.color_mask_b else current_fb_color[2]
                write_color_a = final_pixel_color_tuple[3] if self.graphics_state.color_mask_a else current_fb_color[3]
                
                self.color_buffer.set_pixel(RASTER_X, RASTER_Y, (write_color_r, write_color_g, write_color_b, write_color_a))
        
        self.logger.debug(f"draw_rect: Completed drawing to region ({final_x_start},{final_y_start}) to ({final_x_end},{final_y_end}).")
        return True
    
    def get_framebuffer_data(self):
        """Get the current framebuffer data
        
        Returns:
            Bytes object containing framebuffer data
        """
        return self.color_buffer.get_data()
    
    def execute_cycle(self, delta_time):
        """Execute a GPU cycle for the given time delta
        
        Args:
            delta_time: Time in seconds for this execution step
        """
        # Skip execution if the GPU is in sleep mode
        if self.mode == GpuMode.SLEEP:
            return
        
        # Calculate cycles to execute based on frequency and time delta
        cycles = int(self.frequency * 1000000 * delta_time)
        
        # Generate simulated workloads based on the current mode
        workloads = []
        
        if self.mode == GpuMode.RENDERING:
            # Simulate a rendering workload
            vertex_count = int(self.framebuffer_width * self.framebuffer_height * 0.01)  # Simplified model
            fragment_count = int(self.framebuffer_width * self.framebuffer_height * 0.1)  # Simplified model
            
            workloads.append((ShaderType.VERTEX, vertex_count))
            workloads.append((ShaderType.FRAGMENT, fragment_count))
            
        elif self.mode == GpuMode.COMPUTE:
            # Simulate a compute workload
            compute_size = int(self.cuda_core_count * 0.8)  # Simplified model
            workloads.append((ShaderType.COMPUTE, compute_size))
        
        # Execute workloads on all clusters
        total_operations = 0
        for cluster in self.clusters:
            operations = cluster.execute_cycle(workloads, cycles)
            total_operations += operations
        
        # Update frame timing if we're in rendering mode
        if self.mode == GpuMode.RENDERING:
            current_time = time.time()
            frame_time = current_time - self.last_frame_time
            
            # Only count as a new frame if enough time has passed (simplified)
            if frame_time >= (1.0 / 60):  # Capped at 60 fps for simplicity
                self.frame_count += 1
                self.frame_time_ms = frame_time * 1000
                self.last_frame_time = current_time
    
    def set_mode(self, mode):
        """Set the GPU operating mode"""
        self.mode = mode
        self.logger.info(f"GPU mode set to {mode.name}")
    
    def start_rendering(self):
        """Start the rendering pipeline"""
        self.mode = GpuMode.RENDERING
        self.logger.info("GPU rendering started")
    
    def stop_rendering(self):
        """Stop the rendering pipeline"""
        self.mode = GpuMode.IDLE
        self.logger.info("GPU rendering stopped")
    
    def calculate_tflops(self):
        """Calculate the theoretical TFLOPS based on current frequency"""
        # Simplified TFLOPS calculation: CUDA cores * 2 operations * frequency (GHz)
        tflops = (self.cuda_core_count * 2 * (self.frequency / 1000)) / 1000
        return tflops
    
    def get_performance_stats(self):
        """Get current GPU performance statistics"""
        return {
            "frequency": self.frequency,
            "tflops": self.calculate_tflops(),
            "frame_rate": 1000 / self.frame_time_ms if self.frame_time_ms > 0 else 0,
            "frame_time_ms": self.frame_time_ms,
            "mode": self.mode.name,
            "vram_usage": self.vram_usage
        }
    
    def get_state(self):
        """Get the current state of the GPU"""
        return {
            "frequency": self.frequency,
            "mode": self.mode.name,
            "framebuffer": {
                "width": self.framebuffer_width,
                "height": self.framebuffer_height
            },
            "frame_count": self.frame_count,
            "frame_time_ms": self.frame_time_ms,
            "clusters": [cluster.get_state() for cluster in self.clusters]
        } 