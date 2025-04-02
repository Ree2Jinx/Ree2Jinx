"""
ImaginaryConsole Emulator - GUI Module
Provides GUI utilities for the emulator, including file dialogs and interface elements.
"""

import logging
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import pygame
import pygame.freetype


class FileDialog:
    """Provides file dialog functionality"""
    
    @staticmethod
    def open_file(title="Open File", filetypes=None, initial_dir=None):
        """Show a file open dialog
        
        Args:
            title: Dialog title
            filetypes: List of tuples (description, extensions)
            initial_dir: Initial directory to show
            
        Returns:
            Selected file path, or None if cancelled
        """
        # Hide the pygame window temporarily
        pygame_display_mode = None
        try:
            pygame_display_mode = pygame.display.get_surface().get_size()
            pygame.display.iconify()
        except:
            pass
        
        # Create root window
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        # Convert filetypes to tkinter format
        tk_filetypes = []
        if filetypes:
            for desc, exts in filetypes:
                if isinstance(exts, str):
                    exts = [exts]
                tk_filetypes.append((desc, " ".join(f"*{ext}" for ext in exts)))
        
        # Show the file dialog
        try:
            filepath = filedialog.askopenfilename(
                title=title,
                filetypes=tk_filetypes,
                initialdir=initial_dir
            )
        except Exception as e:
            logging.error(f"Error showing file dialog: {e}")
            filepath = None
        
        # Destroy the root window
        root.destroy()
        
        # Restore the pygame window
        if pygame_display_mode:
            try:
                pygame.display.set_mode(pygame_display_mode, pygame.OPENGL | pygame.DOUBLEBUF)
            except:
                pass
        
        return filepath if filepath else None
    
    @staticmethod
    def save_file(title="Save File", filetypes=None, initial_dir=None, default_ext=None):
        """Show a file save dialog
        
        Args:
            title: Dialog title
            filetypes: List of tuples (description, extensions)
            initial_dir: Initial directory to show
            default_ext: Default extension to append
            
        Returns:
            Selected file path, or None if cancelled
        """
        # Hide the pygame window temporarily
        pygame_display_mode = None
        try:
            pygame_display_mode = pygame.display.get_surface().get_size()
            pygame.display.iconify()
        except:
            pass
        
        # Create root window
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        # Convert filetypes to tkinter format
        tk_filetypes = []
        if filetypes:
            for desc, exts in filetypes:
                if isinstance(exts, str):
                    exts = [exts]
                tk_filetypes.append((desc, " ".join(f"*{ext}" for ext in exts)))
        
        # Show the file dialog
        try:
            filepath = filedialog.asksaveasfilename(
                title=title,
                filetypes=tk_filetypes,
                initialdir=initial_dir,
                defaultextension=default_ext
            )
        except Exception as e:
            logging.error(f"Error showing file dialog: {e}")
            filepath = None
        
        # Destroy the root window
        root.destroy()
        
        # Restore the pygame window
        if pygame_display_mode:
            try:
                pygame.display.set_mode(pygame_display_mode, pygame.OPENGL | pygame.DOUBLEBUF)
            except:
                pass
        
        return filepath if filepath else None
    
    @staticmethod
    def select_directory(title="Select Directory", initial_dir=None):
        """Show a directory selection dialog
        
        Args:
            title: Dialog title
            initial_dir: Initial directory to show
            
        Returns:
            Selected directory path, or None if cancelled
        """
        # Hide the pygame window temporarily
        pygame_display_mode = None
        try:
            pygame_display_mode = pygame.display.get_surface().get_size()
            pygame.display.iconify()
        except:
            pass
        
        # Create root window
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        # Show the directory dialog
        try:
            dirpath = filedialog.askdirectory(
                title=title,
                initialdir=initial_dir
            )
        except Exception as e:
            logging.error(f"Error showing directory dialog: {e}")
            dirpath = None
        
        # Destroy the root window
        root.destroy()
        
        # Restore the pygame window
        if pygame_display_mode:
            try:
                pygame.display.set_mode(pygame_display_mode, pygame.OPENGL | pygame.DOUBLEBUF)
            except:
                pass
        
        return dirpath if dirpath else None


class MessageDialog:
    """Provides message dialog functionality"""
    
    @staticmethod
    def show_info(title, message):
        """Show an information dialog
        
        Args:
            title: Dialog title
            message: Dialog message
        """
        # Hide the pygame window temporarily
        pygame_display_mode = None
        try:
            pygame_display_mode = pygame.display.get_surface().get_size()
            pygame.display.iconify()
        except:
            pass
        
        # Create root window
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        # Show the message dialog
        try:
            messagebox.showinfo(title, message)
        except Exception as e:
            logging.error(f"Error showing message dialog: {e}")
        
        # Destroy the root window
        root.destroy()
        
        # Restore the pygame window
        if pygame_display_mode:
            try:
                pygame.display.set_mode(pygame_display_mode, pygame.OPENGL | pygame.DOUBLEBUF)
            except:
                pass
    
    @staticmethod
    def show_error(title, message):
        """Show an error dialog
        
        Args:
            title: Dialog title
            message: Dialog message
        """
        # Hide the pygame window temporarily
        pygame_display_mode = None
        try:
            pygame_display_mode = pygame.display.get_surface().get_size()
            pygame.display.iconify()
        except:
            pass
        
        # Create root window
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        # Show the message dialog
        try:
            messagebox.showerror(title, message)
        except Exception as e:
            logging.error(f"Error showing error dialog: {e}")
        
        # Destroy the root window
        root.destroy()
        
        # Restore the pygame window
        if pygame_display_mode:
            try:
                pygame.display.set_mode(pygame_display_mode, pygame.OPENGL | pygame.DOUBLEBUF)
            except:
                pass
    
    @staticmethod
    def show_warning(title, message):
        """Show a warning dialog
        
        Args:
            title: Dialog title
            message: Dialog message
        """
        # Hide the pygame window temporarily
        pygame_display_mode = None
        try:
            pygame_display_mode = pygame.display.get_surface().get_size()
            pygame.display.iconify()
        except:
            pass
        
        # Create root window
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        # Show the message dialog
        try:
            messagebox.showwarning(title, message)
        except Exception as e:
            logging.error(f"Error showing warning dialog: {e}")
        
        # Destroy the root window
        root.destroy()
        
        # Restore the pygame window
        if pygame_display_mode:
            try:
                pygame.display.set_mode(pygame_display_mode, pygame.OPENGL | pygame.DOUBLEBUF)
            except:
                pass
    
    @staticmethod
    def ask_yes_no(title, message):
        """Show a yes/no question dialog
        
        Args:
            title: Dialog title
            message: Dialog message
            
        Returns:
            True if yes was selected, False otherwise
        """
        # Hide the pygame window temporarily
        pygame_display_mode = None
        try:
            pygame_display_mode = pygame.display.get_surface().get_size()
            pygame.display.iconify()
        except:
            pass
        
        # Create root window
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        # Show the question dialog
        try:
            result = messagebox.askyesno(title, message)
        except Exception as e:
            logging.error(f"Error showing question dialog: {e}")
            result = False
        
        # Destroy the root window
        root.destroy()
        
        # Restore the pygame window
        if pygame_display_mode:
            try:
                pygame.display.set_mode(pygame_display_mode, pygame.OPENGL | pygame.DOUBLEBUF)
            except:
                pass
        
        return result


class Button:
    """Simple button class for pygame UI"""
    
    def __init__(self, x, y, width, height, text, font=None, color=(100, 100, 200), 
                 hover_color=(150, 150, 255), text_color=(255, 255, 255)):
        """Initialize a button
        
        Args:
            x, y: Position of the button
            width, height: Size of the button
            text: Button text
            font: Pygame font for the text
            color: Normal button color
            hover_color: Color when mouse is hovering over the button
            text_color: Text color
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font if font else pygame.font.SysFont("Arial", 16)
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.hovered = False
        self.clicked = False
        
        # Pre-render the text surfaces
        self.text_surface = self.font.render(text, True, text_color)
        self.text_rect = self.text_surface.get_rect(center=self.rect.center)
    
    def update(self, events):
        """Update the button state
        
        Args:
            events: List of pygame events
            
        Returns:
            True if the button was clicked, False otherwise
        """
        # Check if mouse is hovering over the button
        mouse_pos = pygame.mouse.get_pos()
        self.hovered = self.rect.collidepoint(mouse_pos)
        
        # Reset clicked state
        self.clicked = False
        
        # Check for mouse clicks
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.hovered:
                    self.clicked = True
        
        return self.clicked
    
    def draw(self, surface):
        """Draw the button
        
        Args:
            surface: Pygame surface to draw on
        """
        # Draw the button rectangle
        color = self.hover_color if self.hovered else self.color
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, (50, 50, 50), self.rect, 2)  # Border
        
        # Draw the text
        surface.blit(self.text_surface, self.text_rect)


class Menu:
    """Simple menu class for pygame UI"""
    
    def __init__(self, width, height, font=None, bg_color=(30, 30, 40)):
        """Initialize a menu
        
        Args:
            width, height: Size of the menu
            font: Pygame font for the text
            bg_color: Background color
        """
        self.width = width
        self.height = height
        self.font = font if font else pygame.font.SysFont("Arial", 16)
        self.bg_color = bg_color
        
        # Create a surface for the menu
        self.surface = pygame.Surface((width, height))
        
        # List of buttons in the menu
        self.buttons = []
    
    def add_button(self, x, y, width, height, text, callback, 
                 color=(100, 100, 200), hover_color=(150, 150, 255), text_color=(255, 255, 255)):
        """Add a button to the menu
        
        Args:
            x, y: Position of the button relative to the menu
            width, height: Size of the button
            text: Button text
            callback: Function to call when the button is clicked
            color: Normal button color
            hover_color: Color when mouse is hovering over the button
            text_color: Text color
            
        Returns:
            The created button
        """
        button = Button(x, y, width, height, text, self.font, color, hover_color, text_color)
        self.buttons.append((button, callback))
        return button
    
    def update(self, events):
        """Update the menu state
        
        Args:
            events: List of pygame events
        """
        for button, callback in self.buttons:
            if button.update(events):
                if callback:
                    callback()
    
    def draw(self, surface, x, y):
        """Draw the menu
        
        Args:
            surface: Pygame surface to draw on
            x, y: Position to draw the menu
        """
        # Clear the menu surface
        self.surface.fill(self.bg_color)
        
        # Draw all buttons
        for button, _ in self.buttons:
            button.draw(self.surface)
        
        # Draw the menu to the target surface
        surface.blit(self.surface, (x, y))


class Slider:
    """Simple slider control for pygame UI"""
    
    def __init__(self, x, y, width, height, min_value, max_value, initial_value, 
                 color=(100, 100, 200), handle_color=(150, 150, 255), text=None, font=None):
        """Initialize a slider
        
        Args:
            x, y: Position of the slider
            width, height: Size of the slider
            min_value, max_value: Range of values
            initial_value: Starting value
            color: Slider bar color
            handle_color: Slider handle color
            text: Optional label text
            font: Pygame font for the text
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.min_value = min_value
        self.max_value = max_value
        self.value = max(min_value, min(max_value, initial_value))  # Clamp to range
        self.color = color
        self.handle_color = handle_color
        self.text = text
        self.font = font if font else pygame.font.SysFont("Arial", 16)
        self.dragging = False
        
        # Handle size
        self.handle_width = 16
        self.handle_height = height + 6
        
        # Calculate handle position
        self._update_handle_position()
        
        # Callback
        self.on_change = None
        
        # Pre-render the text surface if text is provided
        if text:
            self.text_surface = self.font.render(text, True, (255, 255, 255))
            self.text_rect = self.text_surface.get_rect(midright=(x - 10, y + height // 2))
            
            # Value text
            self.value_text = f"{self.value:.1f}"
            self.value_surface = self.font.render(self.value_text, True, (255, 255, 255))
            self.value_rect = self.value_surface.get_rect(midleft=(x + width + 10, y + height // 2))
    
    def _update_handle_position(self):
        """Calculate the slider handle position based on current value"""
        value_range = self.max_value - self.min_value
        if value_range == 0:
            value_range = 1  # Avoid division by zero
            
        value_percent = (self.value - self.min_value) / value_range
        handle_x = self.rect.x + (self.rect.width * value_percent) - (self.handle_width // 2)
        self.handle_rect = pygame.Rect(handle_x, self.rect.y - 3, self.handle_width, self.handle_height)
    
    def update_from_position(self, x_pos):
        """Update the slider value from a position"""
        relative_x = max(0, min(x_pos - self.rect.x, self.rect.width))
        value_percent = relative_x / self.rect.width
        new_value = self.min_value + (value_percent * (self.max_value - self.min_value))
        
        # Only update if value changed significantly
        if abs(new_value - self.value) > 0.01:
            self.value = new_value
            self._update_handle_position()
            
            # Update the value text
            if self.text:
                self.value_text = f"{self.value:.1f}"
                self.value_surface = self.font.render(self.value_text, True, (255, 255, 255))
            
            # Call the callback if set
            if self.on_change:
                self.on_change(self.value)
            
            return True
        return False
    
    def update(self, events):
        """Update the slider state
        
        Args:
            events: List of pygame events
            
        Returns:
            True if the value changed, False otherwise
        """
        changed = False
        mouse_pos = pygame.mouse.get_pos()
        
        # Process events
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Check if the mouse is over the handle or slider bar
                if self.handle_rect.collidepoint(mouse_pos) or self.rect.collidepoint(mouse_pos):
                    self.dragging = True
                    changed = self.update_from_position(mouse_pos[0])
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging = False
            elif event.type == pygame.MOUSEMOTION and self.dragging:
                changed = self.update_from_position(mouse_pos[0])
        
        return changed
    
    def draw(self, surface):
        """Draw the slider
        
        Args:
            surface: Pygame surface to draw on
        """
        # Draw the slider bar
        pygame.draw.rect(surface, self.color, self.rect)
        pygame.draw.rect(surface, (50, 50, 50), self.rect, 1)  # Border
        
        # Draw the filled portion
        fill_width = self.handle_rect.centerx - self.rect.x
        fill_rect = pygame.Rect(self.rect.x, self.rect.y, fill_width, self.rect.height)
        pygame.draw.rect(surface, (80, 80, 180), fill_rect)
        
        # Draw the handle
        pygame.draw.rect(surface, self.handle_color, self.handle_rect)
        pygame.draw.rect(surface, (50, 50, 50), self.handle_rect, 1)  # Border
        
        # Draw the label text if present
        if self.text:
            surface.blit(self.text_surface, self.text_rect)
            surface.blit(self.value_surface, self.value_rect)


class ToggleSwitch:
    """Simple toggle switch for pygame UI"""
    
    def __init__(self, x, y, width, height, text, initial_state=False, 
                 color_off=(100, 100, 100), color_on=(100, 200, 100), font=None):
        """Initialize a toggle switch
        
        Args:
            x, y: Position of the switch
            width, height: Size of the switch
            text: Label text
            initial_state: Starting state (True=on, False=off)
            color_off: Color when off
            color_on: Color when on
            font: Pygame font for the text
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.state = initial_state
        self.color_off = color_off
        self.color_on = color_on
        self.font = font if font else pygame.font.SysFont("Arial", 16)
        
        # Handle size
        self.handle_width = width // 2
        self.handle_height = height - 4
        
        # Calculate handle position
        self._update_handle_position()
        
        # Callback
        self.on_change = None
        
        # Pre-render the text surface
        self.text_surface = self.font.render(text, True, (255, 255, 255))
        self.text_rect = self.text_surface.get_rect(midright=(x - 10, y + height // 2))
    
    def _update_handle_position(self):
        """Calculate the handle position based on current state"""
        if self.state:
            handle_x = self.rect.x + self.rect.width - self.handle_width - 2
        else:
            handle_x = self.rect.x + 2
        
        self.handle_rect = pygame.Rect(handle_x, self.rect.y + 2, self.handle_width, self.handle_height)
    
    def toggle(self):
        """Toggle the switch state
        
        Returns:
            The new state
        """
        self.state = not self.state
        self._update_handle_position()
        
        # Call the callback if set
        if self.on_change:
            self.on_change(self.state)
        
        return self.state
    
    def update(self, events):
        """Update the switch state
        
        Args:
            events: List of pygame events
            
        Returns:
            True if the state changed, False otherwise
        """
        changed = False
        
        # Process events
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Check if the mouse is over the switch
                if self.rect.collidepoint(event.pos):
                    self.toggle()
                    changed = True
        
        return changed
    
    def draw(self, surface):
        """Draw the toggle switch
        
        Args:
            surface: Pygame surface to draw on
        """
        # Draw the switch background
        color = self.color_on if self.state else self.color_off
        pygame.draw.rect(surface, color, self.rect, border_radius=self.rect.height // 2)
        pygame.draw.rect(surface, (50, 50, 50), self.rect, 1, border_radius=self.rect.height // 2)  # Border
        
        # Draw the handle
        pygame.draw.rect(surface, (240, 240, 240), self.handle_rect, border_radius=self.handle_height // 2)
        pygame.draw.rect(surface, (100, 100, 100), self.handle_rect, 1, border_radius=self.handle_height // 2)  # Border
        
        # Draw the label text
        surface.blit(self.text_surface, self.text_rect)


class MouseSettingsPanel:
    """Panel for mouse settings in the UI"""
    
    def __init__(self, x, y, width, height, controller_manager, font=None, bg_color=(30, 30, 40)):
        """Initialize the mouse settings panel
        
        Args:
            x, y: Position of the panel
            width, height: Size of the panel
            controller_manager: Reference to the controller manager
            font: Pygame font for the text
            bg_color: Background color
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.controller_manager = controller_manager
        self.font = font if font else pygame.font.SysFont("Arial", 16)
        self.bg_color = bg_color
        
        # Create a surface for the panel
        self.surface = pygame.Surface((width, height))
        
        # Title
        self.title = "Mouse Settings"
        self.title_font = pygame.font.SysFont("Arial", 20, bold=True)
        self.title_surface = self.title_font.render(self.title, True, (255, 255, 255))
        self.title_rect = self.title_surface.get_rect(midtop=(width // 2, 10))
        
        # Mouse sensitivity slider
        self.sensitivity_slider = Slider(
            150, 60, 200, 20, 
            0.1, 3.0, self.controller_manager.mouse_sensitivity,
            text="Sensitivity:",
            font=self.font
        )
        
        # Set callback
        self.sensitivity_slider.on_change = self.on_sensitivity_changed
        
        # Mouse acceleration toggle
        self.acceleration_toggle = ToggleSwitch(
            150, 100, 60, 30, 
            "Acceleration:",
            initial_state=self.controller_manager.mouse_acceleration,
            font=self.font
        )
        
        # Set callback
        self.acceleration_toggle.on_change = self.on_acceleration_changed
        
        # New Joycon mouse support toggle
        self.joycon_mouse_toggle = ToggleSwitch(
            150, 140, 60, 30, 
            "Enable for Joycons:",
            initial_state=self.controller_manager.new_joycon_mouse_enabled,
            font=self.font
        )
        
        # Set callback
        self.joycon_mouse_toggle.on_change = self.on_joycon_mouse_changed
        
        # Direct mouse input toggle
        self.direct_mouse_toggle = ToggleSwitch(
            150, 180, 60, 30, 
            "Enable as System Mouse:",
            initial_state=self.controller_manager.direct_mouse_enabled,
            font=self.font
        )
        
        # Set callback
        self.direct_mouse_toggle.on_change = self.on_direct_mouse_changed
        
        # Close button
        self.close_button = Button(
            width // 2 - 50, height - 40, 100, 30, 
            "Close", self.font, color=(150, 100, 100), hover_color=(200, 120, 120)
        )
        
        # Close callback - set by the caller
        self.on_close = None
    
    def on_sensitivity_changed(self, value):
        """Callback for when the sensitivity slider changes"""
        self.controller_manager.set_mouse_sensitivity(value)
    
    def on_acceleration_changed(self, state):
        """Callback for when the acceleration toggle changes"""
        self.controller_manager.toggle_mouse_acceleration()
    
    def on_joycon_mouse_changed(self, state):
        """Callback for when the Joycon mouse toggle changes"""
        self.controller_manager.toggle_new_joycon_mouse()
        
    def on_direct_mouse_changed(self, state):
        """Callback for when the direct mouse toggle changes"""
        self.controller_manager.toggle_direct_mouse()
    
    def update(self, events):
        """Update the panel state
        
        Args:
            events: List of pygame events
        """
        self.sensitivity_slider.update(events)
        self.acceleration_toggle.update(events)
        self.joycon_mouse_toggle.update(events)
        self.direct_mouse_toggle.update(events)
        
        # Check if close button was clicked
        if self.close_button.update(events) and self.on_close:
            self.on_close()
    
    def draw(self, surface, x, y):
        """Draw the panel
        
        Args:
            surface: Pygame surface to draw on
            x, y: Position to draw the panel
        """
        # Clear the panel surface
        self.surface.fill(self.bg_color)
        
        # Draw a border
        pygame.draw.rect(self.surface, (60, 60, 80), (0, 0, self.rect.width, self.rect.height), 2)
        
        # Draw the title
        self.surface.blit(self.title_surface, self.title_rect)
        
        # Draw the controls
        self.sensitivity_slider.draw(self.surface)
        self.acceleration_toggle.draw(self.surface)
        self.joycon_mouse_toggle.draw(self.surface)
        self.direct_mouse_toggle.draw(self.surface)
        self.close_button.draw(self.surface)
        
        # Draw the panel to the target surface
        surface.blit(self.surface, (x, y)) 