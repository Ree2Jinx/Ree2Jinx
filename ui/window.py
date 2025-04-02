"""
ImaginaryConsole Emulator - Window Module
Handles the main window, rendering, and user interface for the emulator.
"""

import logging
import time
import os
import pygame
import OpenGL.GL as gl
from pathlib import Path
import numpy as np
import platform
import subprocess

from ui.gui import FileDialog, MessageDialog, MouseSettingsPanel


class Window:
    """Main window for the emulator"""
    
    # Default window properties
    DEFAULT_WIDTH = 1280
    DEFAULT_HEIGHT = 720
    DEFAULT_TITLE = "ImaginaryConsole Emulator"
    
    def __init__(self, config):
        """Initialize the window with the given configuration
        
        Args:
            config: Emulator configuration object
        """
        self.logger = logging.getLogger("ImaginaryConsole.Window")
        self.config = config
        
        # Initialize pygame
        pygame.init()
        
        # Disable key repeat to avoid multiple triggers
        pygame.key.set_repeat(0)  # Disable key repeat completely
        
        # Set window properties from config
        self.width, self.height = config.window_size
        self.fullscreen = config.fullscreen
        self.vsync = config.vsync
        
        # Calculate the scaling factor relative to the native resolution
        self.scale_x = self.width / self.DEFAULT_WIDTH
        self.scale_y = self.height / self.DEFAULT_HEIGHT
        
        # UI state
        self.running = False
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.frame_count = 0
        self.frame_time = 0
        self.title = self.DEFAULT_TITLE
        
        # Assets
        self.assets_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "assets"
        self.logo = None
        self.font = None
        self.load_assets()
        
        # Input manager
        self.input_manager = None
        
        # UI panels
        self.mouse_settings_panel = None
        self.show_mouse_settings = False
        
        # Create window and renderer
        self._create_window()
        
        # Dialog state tracking
        self.dialog_active = False
        self.last_dialog_time = 0
        
        # Key tracking to handle repeats
        self.ctrl_o_pressed = False
        self.ctrl_o_handled = False
        
        # Add event filter to catch key events early
        pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.KEYUP, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP])
        
        self.logger.info(f"Window initialized: {self.width}x{self.height}, fullscreen={self.fullscreen}")
    
    def load_assets(self):
        """Load UI assets"""
        try:
            # Create assets directory if it doesn't exist
            self.assets_path.mkdir(parents=True, exist_ok=True)
            
            # Load logo if available
            logo_path = self.assets_path / "logo.png"
            if logo_path.exists():
                self.logo = pygame.image.load(str(logo_path))
                self.logger.debug(f"Loaded logo: {logo_path}")
            
            # Initialize font
            pygame.font.init()
            font_path = self.assets_path / "font.ttf"
            if font_path.exists():
                self.font = pygame.font.Font(str(font_path), 16)
            else:
                self.font = pygame.font.SysFont("Arial", 16)
            
            self.logger.debug("Assets loaded")
            
        except Exception as e:
            self.logger.error(f"Failed to load assets: {e}")
            # Fall back to system font
            self.font = pygame.font.SysFont("Arial", 16)
    
    def _create_window(self):
        """Create the pygame window"""
        pygame.display.set_caption(self.title)
        
        # Set graphics flags
        flags = pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.OPENGL
        if self.fullscreen:
            flags |= pygame.FULLSCREEN
        
        # Create the window
        self.surface = pygame.display.set_mode((self.width, self.height), flags, vsync=self.vsync)
        
        # Set icon if available
        if self.logo:
            pygame.display.set_icon(self.logo)
        
        # Initialize OpenGL context
        self._init_gl()
    
    def _init_gl(self):
        """Initialize OpenGL settings"""
        # Set clear color to black
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        
        # Set up orthographic projection for 2D rendering
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, self.width, self.height, 0, -1, 1)
        
        # Set matrix mode to model-view
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        
        # Enable alpha blending
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        
        # Additional OpenGL settings
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glDisable(gl.GL_LIGHTING)
        
        self.logger.debug("OpenGL initialized")
    
    def register_input_manager(self, input_manager):
        """Register the input manager for the window
        
        Args:
            input_manager: ControllerManager instance
        """
        self.input_manager = input_manager
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        self.fullscreen = not self.fullscreen
        self._create_window()
        self.logger.info(f"Fullscreen mode: {self.fullscreen}")
    
    def resize(self, width, height):
        """Resize the window
        
        Args:
            width: New window width
            height: New window height
        """
        if width == self.width and height == self.height:
            return
        
        self.width = width
        self.height = height
        self.scale_x = self.width / self.DEFAULT_WIDTH
        self.scale_y = self.height / self.DEFAULT_HEIGHT
        
        self._create_window()
        self.logger.info(f"Window resized to {width}x{height}")
    
    def handle_events(self):
        """Handle window events
        
        Returns:
            True if the window should continue running, False otherwise
        """
        events = pygame.event.get()
        
        # Handle Ctrl+O specially to prevent duplicates
        keys = pygame.key.get_pressed()
        ctrl_pressed = keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL] or keys[pygame.K_LMETA] or keys[pygame.K_RMETA]
        o_pressed = keys[pygame.K_o]
        
        # Detect Ctrl+O state change (pressed but not yet handled)
        if ctrl_pressed and o_pressed:
            if not self.ctrl_o_pressed:
                self.ctrl_o_pressed = True
                self.ctrl_o_handled = False
        else:
            self.ctrl_o_pressed = False
        
        # Handle Ctrl+O only once per press and not during dialogs
        if self.ctrl_o_pressed and not self.ctrl_o_handled and not self.dialog_active:
            self.ctrl_o_handled = True
            self._handle_open_rom()
        
        # Process regular events
        for event in events:
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                # Alt+Enter to toggle fullscreen
                if event.key == pygame.K_RETURN and (pygame.key.get_mods() & pygame.KMOD_ALT):
                    self.toggle_fullscreen()
                # Escape to exit
                elif event.key == pygame.K_ESCAPE:
                    # If mouse settings are shown, close the panel
                    if self.show_mouse_settings:
                        self.show_mouse_settings = False
                        return True
                    return False
                # F2 to toggle mouse settings panel
                elif event.key == pygame.K_F2:
                    self.show_mouse_settings = not self.show_mouse_settings
                    if self.show_mouse_settings and not self.mouse_settings_panel:
                        # Create the panel if it doesn't exist
                        self._create_mouse_settings_panel()
            
            # Mouse click to lock/unlock mouse for pointer control
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Only lock mouse if mouse settings panel is not shown
                if self.input_manager and not self.show_mouse_settings:
                    self.input_manager.lock_mouse(self.surface)
            
            # Escape key to unlock mouse if locked
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                if self.input_manager and self.input_manager.mouse_locked:
                    self.input_manager.unlock_mouse()
                    # Don't exit if just unlocking the mouse
                    continue
        
        # Update UI elements if active
        if self.show_mouse_settings and self.mouse_settings_panel:
            self.mouse_settings_panel.update(events)
        
        return True
    
    def _handle_open_rom(self):
        """Safely handle opening ROM dialog to prevent multiple calls"""
        # Check cooldown timer
        current_time = time.time()
        if current_time - self.last_dialog_time < 1.0:
            self.logger.debug("Ignoring Ctrl+O due to cooldown")
            return
            
        self.last_dialog_time = current_time
        self.dialog_active = True
        try:
            # Log only once
            self.logger.info("Opening ROM file dialog")
            self._open_rom_file_direct()
        except Exception as e:
            self.logger.error(f"Error handling open ROM dialog: {e}", exc_info=True)
        finally:
            self.dialog_active = False
    
    def _open_rom_file_direct(self):
        """Open a ROM file using direct commands to avoid duplicate triggers"""
        # Import locally to avoid circular imports
        from system.rom_loader import RomLoader
        
        # On macOS, use direct AppleScript instead of going through FileDialog
        is_macos = platform.system() == 'Darwin'
        rom_path = None
        
        if is_macos:
            try:
                # Construct a direct AppleScript command
                apple_script = '''
                tell application "System Events"
                    activate
                    set theFile to POSIX path of (choose file with prompt "Open ROM File")
                    return theFile
                end tell
                '''
                
                # Execute the command
                result = subprocess.run(['osascript', '-e', apple_script], 
                                       capture_output=True, text=True)
                
                if result.returncode == 0 and result.stdout.strip():
                    rom_path = result.stdout.strip()
            except Exception as e:
                self.logger.error(f"Error with AppleScript dialog: {e}")
        else:
            # For non-macOS, use the regular FileDialog
            from ui.gui import FileDialog
            
            # Define ROM file types
            rom_filetypes = [
                ("ROM Files", [".rom", ".bin", ".iso"]),
                ("All Files", [".*"])
            ]
            
            # Show file dialog
            rom_path = FileDialog.open_file(
                title="Open ROM File",
                filetypes=rom_filetypes,
                initial_dir=str(self.config.roms_path)
            )
        
        # If a file was selected, load it
        if rom_path:
            self.logger.info(f"Selected ROM file: {rom_path}")
            self._load_selected_rom(rom_path)
    
    def _load_selected_rom(self, rom_path):
        """Load a selected ROM file"""
        # Import locally to avoid circular imports
        from ui.gui import MessageDialog
        from system.rom_loader import RomLoader
        
        try:
            # Access the system instance
            if not hasattr(self, "_running_system") or self._running_system is None:
                self.logger.error("No system instance available")
                MessageDialog.show_error("Error", "Cannot load ROM: system not initialized")
                return
            
            system = self._running_system
            
            # Create ROM loader and load the ROM
            rom_loader = RomLoader(rom_path, self.config.keys_path)
            if system.load_rom(rom_loader):
                self.logger.info(f"ROM loaded successfully: {rom_path}")
            else:
                self.logger.error(f"Failed to load ROM: {rom_path}")
                MessageDialog.show_error("Error", f"Failed to load ROM: {rom_path}")
        except Exception as e:
            self.logger.error(f"Error loading ROM: {e}", exc_info=True)
            MessageDialog.show_error("Error", f"Error loading ROM: {e}")
    
    def _open_rom_file(self):
        """Legacy method for opening ROM files via FileDialog"""
        # This method is kept for backward compatibility
        # For new code, use _open_rom_file_direct instead
        self._open_rom_file_direct()
    
    def update(self, system):
        """Update the window state
        
        Args:
            system: System instance to update
        """
        # Update system state
        if system:
            system.step()
        
        # Update FPS counter
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.frame_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.frame_time = now
            
            # Update window title with FPS
            if system and system.rom_info:
                title = f"{self.DEFAULT_TITLE} - {system.rom_info['title']} - {self.fps:.1f} FPS"
            else:
                title = f"{self.DEFAULT_TITLE} - {self.fps:.1f} FPS"
            
            pygame.display.set_caption(title)
    
    def render(self, system):
        """Render the emulator display
        
        Args:
            system: System instance to render
        """
        # Clear the screen
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glLoadIdentity()
        
        # Render the emulator display (simplified)
        if system and system.rom_loaded:
            # In a real emulator, we would render the actual framebuffer from the GPU
            # Here we just draw a placeholder rectangle
            self._render_game_screen()
        else:
            # If no ROM is loaded, show the startup screen
            self._render_startup_screen()
        
        # Render UI overlay (FPS, etc.)
        self._render_ui_overlay(system)
        
        # Render mouse settings panel if shown
        if self.show_mouse_settings and self.mouse_settings_panel:
            # Create a surface for the panel
            panel_surface = pygame.Surface((self.mouse_settings_panel.rect.width, 
                                           self.mouse_settings_panel.rect.height))
            # Draw the panel to this surface
            self.mouse_settings_panel.draw(panel_surface, 0, 0)
            
            # Convert to texture and draw
            panel_texture = self._create_texture_from_surface(panel_surface)
            self._draw_texture(panel_texture, self.mouse_settings_panel.rect.x, 
                              self.mouse_settings_panel.rect.y,
                              self.mouse_settings_panel.rect.width,
                              self.mouse_settings_panel.rect.height)
            gl.glDeleteTextures(1, [panel_texture])
        
        # Swap buffers
        pygame.display.flip()
    
    def _render_game_screen(self):
        """Render the game screen"""
        # Clear to black first
        gl.glColor3f(0.0, 0.0, 0.0)
        self._draw_rect(0, 0, self.width, self.height)
        
        # Draw a placeholder game screen (gradient background)
        gl.glBegin(gl.GL_QUADS)
        gl.glColor3f(0.0, 0.0, 0.2)  # Dark blue at top
        gl.glVertex2f(0, 0)
        gl.glVertex2f(self.width, 0)
        gl.glColor3f(0.0, 0.2, 0.4)  # Lighter blue at bottom
        gl.glVertex2f(self.width, self.height)
        gl.glVertex2f(0, self.height)
        gl.glEnd()
        
        # Draw a sample sprite
        self._draw_sprite()
    
    def _render_startup_screen(self):
        """Render the startup screen (when no ROM is loaded)"""
        # Draw background
        gl.glColor3f(0.05, 0.05, 0.1)  # Dark blue background
        self._draw_rect(0, 0, self.width, self.height)
        
        # Draw logo if available
        if self.logo:
            logo_texture = self._create_texture_from_surface(self.logo)
            logo_width = self.logo.get_width() * self.scale_x
            logo_height = self.logo.get_height() * self.scale_y
            
            # Center the logo
            x = (self.width - logo_width) / 2
            y = (self.height - logo_height) / 3
            
            self._draw_texture(logo_texture, x, y, logo_width, logo_height)
            gl.glDeleteTextures(1, [logo_texture])
        
        # Draw welcome text
        text = "ImaginaryConsole Emulator"
        text_surface = self.font.render(text, True, (255, 255, 255))
        text_texture = self._create_texture_from_surface(text_surface)
        text_width = text_surface.get_width() * self.scale_x
        text_height = text_surface.get_height() * self.scale_y
        
        # Position below logo
        x = (self.width - text_width) / 2
        y = self.height / 2
        
        self._draw_texture(text_texture, x, y, text_width, text_height)
        gl.glDeleteTextures(1, [text_texture])
        
        # Draw instructions
        instructions = "Press Ctrl+O to open a ROM file"
        inst_surface = self.font.render(instructions, True, (200, 200, 200))
        inst_texture = self._create_texture_from_surface(inst_surface)
        inst_width = inst_surface.get_width() * self.scale_x
        inst_height = inst_surface.get_height() * self.scale_y
        
        # Position below welcome text
        x = (self.width - inst_width) / 2
        y = self.height / 2 + text_height + 20
        
        self._draw_texture(inst_texture, x, y, inst_width, inst_height)
        gl.glDeleteTextures(1, [inst_texture])
    
    def _render_ui_overlay(self, system):
        """Render UI overlay with system information
        
        Args:
            system: System instance
        """
        # Draw FPS counter in the top-right corner
        fps_text = f"FPS: {self.fps:.1f}"
        fps_surface = self.font.render(fps_text, True, (255, 255, 255))
        fps_texture = self._create_texture_from_surface(fps_surface)
        fps_width = fps_surface.get_width() * self.scale_x
        fps_height = fps_surface.get_height() * self.scale_y
        
        x = self.width - fps_width - 10
        y = 10
        
        self._draw_texture(fps_texture, x, y, fps_width, fps_height)
        gl.glDeleteTextures(1, [fps_texture])
        
        # If system is running, draw system info
        if system and system.rom_loaded:
            # Draw ROM title
            title = system.rom_info.get("title", "Unknown ROM")
            title_surface = self.font.render(title, True, (255, 255, 255))
            title_texture = self._create_texture_from_surface(title_surface)
            title_width = title_surface.get_width() * self.scale_x
            title_height = title_surface.get_height() * self.scale_y
            
            x = 10
            y = 10
            
            self._draw_texture(title_texture, x, y, title_width, title_height)
            gl.glDeleteTextures(1, [title_texture])
            
            # Draw mode (docked/handheld)
            mode = "Docked" if self.config.docked_mode else "Handheld"
            mode_surface = self.font.render(mode, True, (200, 200, 200))
            mode_texture = self._create_texture_from_surface(mode_surface)
            mode_width = mode_surface.get_width() * self.scale_x
            mode_height = mode_surface.get_height() * self.scale_y
            
            x = 10
            y = 10 + title_height + 5
            
            self._draw_texture(mode_texture, x, y, mode_width, mode_height)
            gl.glDeleteTextures(1, [mode_texture])
    
    def _draw_sprite(self):
        """Draw a sample sprite for demonstration"""
        # Create a simple geometry
        size = min(self.width, self.height) / 8
        x = self.width / 2 - size / 2
        y = self.height / 2 - size / 2
        
        # Animate position based on time
        t = time.time()
        x += np.sin(t) * size
        y += np.cos(t) * size
        
        # Draw the sprite
        gl.glColor3f(1.0, 0.8, 0.2)  # Yellow
        self._draw_rect(x, y, size, size)
        
        # Draw a smaller inner rect
        gl.glColor3f(0.8, 0.3, 0.1)  # Orange-red
        inner_size = size * 0.7
        inner_x = x + (size - inner_size) / 2
        inner_y = y + (size - inner_size) / 2
        self._draw_rect(inner_x, inner_y, inner_size, inner_size)
    
    def _draw_rect(self, x, y, width, height):
        """Draw a rectangle with the current color"""
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(x, y)
        gl.glVertex2f(x + width, y)
        gl.glVertex2f(x + width, y + height)
        gl.glVertex2f(x, y + height)
        gl.glEnd()
    
    def _create_texture_from_surface(self, surface):
        """Create an OpenGL texture from a pygame surface
        
        Args:
            surface: Pygame surface to convert
            
        Returns:
            OpenGL texture ID
        """
        # Get the size and data from the surface
        width, height = surface.get_size()
        data = pygame.image.tostring(surface, "RGBA", 1)
        
        # Create a new texture
        texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        
        # Set texture parameters
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        
        # Upload the texture data
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0, 
                       gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)
        
        return texture
    
    def _draw_texture(self, texture, x, y, width, height):
        """Draw a texture at the specified position
        
        Args:
            texture: OpenGL texture ID
            x, y: Position to draw
            width, height: Size to draw
        """
        # Enable texturing
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        
        # Set color to white for unmodified texture
        gl.glColor4f(1.0, 1.0, 1.0, 1.0)
        
        # Draw the quad with texture coordinates
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0, 1)
        gl.glVertex2f(x, y)
        gl.glTexCoord2f(1, 1)
        gl.glVertex2f(x + width, y)
        gl.glTexCoord2f(1, 0)
        gl.glVertex2f(x + width, y + height)
        gl.glTexCoord2f(0, 0)
        gl.glVertex2f(x, y + height)
        gl.glEnd()
        
        # Disable texturing
        gl.glDisable(gl.GL_TEXTURE_2D)
    
    def _create_mouse_settings_panel(self):
        """Create the mouse settings panel"""
        if not self.input_manager:
            return
        
        panel_width = 400
        panel_height = 300
        
        # Center the panel in the window
        x = (self.width - panel_width) // 2
        y = (self.height - panel_height) // 2
        
        self.mouse_settings_panel = MouseSettingsPanel(
            x, y, panel_width, panel_height, self.input_manager, self.font
        )
        
        # Set the close callback
        self.mouse_settings_panel.on_close = self._close_mouse_settings_panel
    
    def _close_mouse_settings_panel(self):
        """Close the mouse settings panel"""
        self.show_mouse_settings = False
    
    def run(self, system):
        """Run the main window loop
        
        Args:
            system: System instance to run
        """
        self.running = True
        self.frame_time = time.time()
        self._running_system = system  # Store reference to the system for use in other methods
        
        # Start controller polling if input manager is available and we're not on macOS
        is_macos = platform.system() == 'Darwin'
        
        if self.input_manager:
            # On macOS, we'll call update() directly in the main loop
            # instead of using a background thread
            if not is_macos:
                self.input_manager.start_polling()
        
        try:
            while self.running:
                # Handle events
                self.running = self.handle_events()
                if not self.running:
                    break
                
                # On macOS, manually update the controller state in the main thread
                if is_macos and self.input_manager:
                    self.input_manager.update()
                
                # Update state
                self.update(system)
                
                # Render
                self.render(system)
                
                # Cap to target framerate
                self.clock.tick(60)
                
        except Exception as e:
            self.logger.error(f"Error in window loop: {e}", exc_info=True)
        
        finally:
            # Stop controller polling
            if self.input_manager and not is_macos:
                self.input_manager.stop_polling()
            
            # Clean up pygame
            self.close()
            pygame.quit()
            
            self.logger.info("Window closed")
    
    def close(self):
        """Close the window"""
        self.running = False 