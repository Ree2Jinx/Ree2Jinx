"""
ImaginaryConsole Emulator - Controller Manager Module
Handles the detection, configuration, and input processing for controllers including joycons and pro controllers.
"""

import logging
import time
import json
import os
import threading
from enum import Enum
from pathlib import Path

import pygame


class ControllerType(Enum):
    """Enum for different types of controllers"""
    UNKNOWN = 0
    JOYCON_LEFT = 1
    JOYCON_RIGHT = 2
    JOYCON_PAIR = 3
    PRO_CONTROLLER = 4
    XBOX_CONTROLLER = 5
    MOUSE_KEYBOARD = 6
    MOUSE = 7  # Direct mouse input device


class AxisMapping:
    """Maps controller axis to emulated controller axis"""
    
    def __init__(self, source_axis, target_axis, invert=False, deadzone=0.1, scale=1.0):
        """Initialize axis mapping
        
        Args:
            source_axis: Source axis ID on the physical controller
            target_axis: Target axis ID on the emulated controller
            invert: Whether to invert the axis values
            deadzone: Deadzone to apply (0.0 to 1.0)
            scale: Scaling factor
        """
        self.source_axis = source_axis
        self.target_axis = target_axis
        self.invert = invert
        self.deadzone = deadzone
        self.scale = scale
    
    def process(self, value):
        """Process the input value through the mapping
        
        Args:
            value: Input axis value (-1.0 to 1.0)
            
        Returns:
            Processed axis value
        """
        # Apply deadzone
        if abs(value) < self.deadzone:
            return 0.0
        
        # Adjust for deadzone so we still use the full range
        if value > 0:
            adjusted = (value - self.deadzone) / (1.0 - self.deadzone)
        else:
            adjusted = (value + self.deadzone) / (1.0 - self.deadzone)
        
        # Apply inversion if needed
        if self.invert:
            adjusted = -adjusted
        
        # Apply scaling
        return adjusted * self.scale


class ButtonMapping:
    """Maps controller button to emulated controller button"""
    
    def __init__(self, source_button, target_button):
        """Initialize button mapping
        
        Args:
            source_button: Source button ID on the physical controller
            target_button: Target button ID on the emulated controller
        """
        self.source_button = source_button
        self.target_button = target_button


class ControllerProfile:
    """Represents a controller mapping profile"""
    
    def __init__(self, name, controller_type):
        """Initialize a controller profile
        
        Args:
            name: Name of the profile
            controller_type: Type of controller this profile is for
        """
        self.name = name
        self.controller_type = controller_type
        self.button_mappings = []
        self.axis_mappings = []
    
    def add_button_mapping(self, source_button, target_button):
        """Add a button mapping to the profile"""
        mapping = ButtonMapping(source_button, target_button)
        self.button_mappings.append(mapping)
    
    def add_axis_mapping(self, source_axis, target_axis, invert=False, deadzone=0.1, scale=1.0):
        """Add an axis mapping to the profile"""
        mapping = AxisMapping(source_axis, target_axis, invert, deadzone, scale)
        self.axis_mappings.append(mapping)
    
    def get_button_mapping(self, source_button):
        """Get the target button for a source button"""
        for mapping in self.button_mappings:
            if mapping.source_button == source_button:
                return mapping.target_button
        return None
    
    def get_axis_mapping(self, source_axis):
        """Get the axis mapping for a source axis"""
        for mapping in self.axis_mappings:
            if mapping.source_axis == source_axis:
                return mapping
        return None


class Controller:
    """Represents a connected controller"""
    
    def __init__(self, controller_id, controller_type, name):
        """Initialize a controller
        
        Args:
            controller_id: ID of the controller in pygame
            controller_type: Type of controller
            name: Name of the controller
        """
        self.id = controller_id
        self.type = controller_type
        self.name = name
        self.profile = None
        
        # Button and axis states
        self.buttons = {}
        self.axes = {}
        
        # Motion data (for joycons)
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        
        # Connection info
        self.connected = True
        self.connection_time = time.time()
        self.last_input_time = self.connection_time
    
    def update_button(self, button_id, pressed):
        """Update the state of a button
        
        Args:
            button_id: Button ID
            pressed: Whether the button is pressed
        
        Returns:
            True if the state changed, False otherwise
        """
        # Check if state changed
        changed = button_id not in self.buttons or self.buttons[button_id] != pressed
        
        # Update state
        self.buttons[button_id] = pressed
        
        if changed:
            self.last_input_time = time.time()
        
        return changed
    
    def update_axis(self, axis_id, value):
        """Update the state of an axis
        
        Args:
            axis_id: Axis ID
            value: Axis value (-1.0 to 1.0)
        
        Returns:
            True if the state changed significantly, False otherwise
        """
        # Check if the axis changed significantly (to avoid noise)
        changed = False
        if axis_id not in self.axes:
            changed = True
        elif abs(self.axes[axis_id] - value) > 0.05:  # Sensitivity threshold
            changed = True
        
        # Update state
        self.axes[axis_id] = value
        
        if changed:
            self.last_input_time = time.time()
        
        return changed
    
    def update_motion(self, accelerometer, gyroscope):
        """Update motion sensor data
        
        Args:
            accelerometer: (x, y, z) accelerometer values
            gyroscope: (x, y, z) gyroscope values
        """
        self.accelerometer = accelerometer
        self.gyroscope = gyroscope
        self.last_input_time = time.time()
    
    def get_target_button_state(self, target_button):
        """Get the state of a target button based on mappings
        
        Args:
            target_button: Target button ID
            
        Returns:
            True if the button is pressed, False otherwise
        """
        if not self.profile:
            return False
        
        # Check all mappings for this target button
        for mapping in self.profile.button_mappings:
            if mapping.target_button == target_button:
                source_button = mapping.source_button
                if source_button in self.buttons and self.buttons[source_button]:
                    return True
        
        return False
    
    def get_target_axis_value(self, target_axis):
        """Get the value of a target axis based on mappings
        
        Args:
            target_axis: Target axis ID
            
        Returns:
            Axis value (-1.0 to 1.0)
        """
        if not self.profile:
            return 0.0
        
        # Find all mappings for this target axis
        for mapping in self.profile.axis_mappings:
            if mapping.target_axis == target_axis:
                source_axis = mapping.source_axis
                if source_axis in self.axes:
                    return mapping.process(self.axes[source_axis])
        
        return 0.0


class ControllerManager:
    """Manages controller input for the emulator"""
    
    # Button definitions for the imaginary console
    class Buttons:
        A = 0
        B = 1
        X = 2
        Y = 3
        L = 4
        R = 5
        ZL = 6
        ZR = 7
        MINUS = 8
        PLUS = 9
        LEFT_STICK = 10
        RIGHT_STICK = 11
        HOME = 12
        SHARE = 13
        DPAD_UP = 14
        DPAD_RIGHT = 15
        DPAD_DOWN = 16
        DPAD_LEFT = 17
    
    # Axis definitions for the imaginary console
    class Axes:
        LEFT_X = 0
        LEFT_Y = 1
        RIGHT_X = 2
        RIGHT_Y = 3
        GYRO_X = 4
        GYRO_Y = 5
        GYRO_Z = 6
        ACCEL_X = 7
        ACCEL_Y = 8
        ACCEL_Z = 9
    
    def __init__(self, config):
        """Initialize the controller manager
        
        Args:
            config: Emulator configuration object
        """
        self.logger = logging.getLogger("ImaginaryConsole.ControllerManager")
        self.config = config
        
        # Initialize pygame for controller support
        pygame.init()
        pygame.joystick.init()
        
        # Active controllers
        self.controllers = {}
        
        # Controller profiles
        self.profiles = {}
        self.load_default_profiles()
        self.load_custom_profiles()
        
        # Mouse state
        self.mouse_locked = False
        self.mouse_position = (0, 0)
        self.mouse_buttons = [False, False, False]  # Left, middle, right
        self.mouse_absolute_position = (0, 0)       # Absolute screen position
        self.mouse_motion_accumulator = [0, 0]      # For tracking small movements
        self.mouse_sensitivity = 1.0                # Adjustable sensitivity
        self.mouse_acceleration = False             # Mouse acceleration toggle
        self.new_joycon_mouse_enabled = True        # Whether to use mouse with New Joycons
        self.direct_mouse_enabled = True            # Whether to use mouse as direct input device
        
        # Keyboard state
        self.keys = {}
        
        # Thread for background controller polling
        self.polling_thread = None
        self.stop_event = threading.Event()
        
        # Scan for controllers
        self.scan_controllers()
        
        self.logger.info(f"Controller manager initialized, found {len(self.controllers)} controllers")
    
    def load_default_profiles(self):
        """Load the default controller profiles"""
        # Default profile for Joycon (Left)
        joycon_left = ControllerProfile("Default Joycon Left", ControllerType.JOYCON_LEFT)
        # Add button mappings...
        joycon_left.add_button_mapping(0, self.Buttons.MINUS)
        joycon_left.add_button_mapping(1, self.Buttons.SHARE)
        joycon_left.add_button_mapping(2, self.Buttons.DPAD_UP)
        joycon_left.add_button_mapping(3, self.Buttons.DPAD_RIGHT)
        joycon_left.add_button_mapping(4, self.Buttons.DPAD_DOWN)
        joycon_left.add_button_mapping(5, self.Buttons.DPAD_LEFT)
        joycon_left.add_button_mapping(6, self.Buttons.L)
        joycon_left.add_button_mapping(7, self.Buttons.ZL)
        joycon_left.add_button_mapping(8, self.Buttons.LEFT_STICK)
        # Add axis mappings...
        joycon_left.add_axis_mapping(0, self.Axes.LEFT_X)
        joycon_left.add_axis_mapping(1, self.Axes.LEFT_Y, invert=True)
        self.profiles["default_joycon_left"] = joycon_left
        
        # Default profile for Joycon (Right)
        joycon_right = ControllerProfile("Default Joycon Right", ControllerType.JOYCON_RIGHT)
        # Add button mappings...
        joycon_right.add_button_mapping(0, self.Buttons.PLUS)
        joycon_right.add_button_mapping(1, self.Buttons.HOME)
        joycon_right.add_button_mapping(2, self.Buttons.A)
        joycon_right.add_button_mapping(3, self.Buttons.X)
        joycon_right.add_button_mapping(4, self.Buttons.B)
        joycon_right.add_button_mapping(5, self.Buttons.Y)
        joycon_right.add_button_mapping(6, self.Buttons.R)
        joycon_right.add_button_mapping(7, self.Buttons.ZR)
        joycon_right.add_button_mapping(8, self.Buttons.RIGHT_STICK)
        # Add axis mappings...
        joycon_right.add_axis_mapping(0, self.Axes.RIGHT_X)
        joycon_right.add_axis_mapping(1, self.Axes.RIGHT_Y, invert=True)
        # Add mouse support to right Joycon
        joycon_right.add_axis_mapping(100, self.Axes.GYRO_X)  # Virtual axis for mouse X movement
        joycon_right.add_axis_mapping(101, self.Axes.GYRO_Y)  # Virtual axis for mouse Y movement
        self.profiles["default_joycon_right"] = joycon_right
        
        # Default profile for Pro Controller
        pro = ControllerProfile("Default Pro Controller", ControllerType.PRO_CONTROLLER)
        # Add button mappings...
        pro.add_button_mapping(0, self.Buttons.A)
        pro.add_button_mapping(1, self.Buttons.B)
        pro.add_button_mapping(2, self.Buttons.X)
        pro.add_button_mapping(3, self.Buttons.Y)
        pro.add_button_mapping(4, self.Buttons.L)
        pro.add_button_mapping(5, self.Buttons.R)
        pro.add_button_mapping(6, self.Buttons.ZL)
        pro.add_button_mapping(7, self.Buttons.ZR)
        pro.add_button_mapping(8, self.Buttons.MINUS)
        pro.add_button_mapping(9, self.Buttons.PLUS)
        pro.add_button_mapping(10, self.Buttons.LEFT_STICK)
        pro.add_button_mapping(11, self.Buttons.RIGHT_STICK)
        pro.add_button_mapping(12, self.Buttons.HOME)
        pro.add_button_mapping(13, self.Buttons.SHARE)
        # D-pad handled via hat in update_controller
        # Add axis mappings...
        pro.add_axis_mapping(0, self.Axes.LEFT_X)
        pro.add_axis_mapping(1, self.Axes.LEFT_Y, invert=True)
        pro.add_axis_mapping(2, self.Axes.RIGHT_X)
        pro.add_axis_mapping(3, self.Axes.RIGHT_Y, invert=True)
        self.profiles["default_pro_controller"] = pro
        
        # Default profile for Xbox Controller (emulating Pro Controller)
        xbox = ControllerProfile("Default Xbox Controller", ControllerType.XBOX_CONTROLLER)
        # Add button mappings...
        xbox.add_button_mapping(0, self.Buttons.B)  # Xbox A = B
        xbox.add_button_mapping(1, self.Buttons.A)  # Xbox B = A
        xbox.add_button_mapping(2, self.Buttons.Y)  # Xbox X = Y
        xbox.add_button_mapping(3, self.Buttons.X)  # Xbox Y = X
        xbox.add_button_mapping(4, self.Buttons.L)
        xbox.add_button_mapping(5, self.Buttons.R)
        xbox.add_button_mapping(6, self.Buttons.MINUS)  # Xbox Back/View = Minus
        xbox.add_button_mapping(7, self.Buttons.PLUS)   # Xbox Start/Menu = Plus
        xbox.add_button_mapping(8, self.Buttons.LEFT_STICK)
        xbox.add_button_mapping(9, self.Buttons.RIGHT_STICK)
        xbox.add_button_mapping(10, self.Buttons.HOME)  # Xbox Guide = Home
        # D-pad handled via hat in update_controller
        # Add axis mappings...
        xbox.add_axis_mapping(0, self.Axes.LEFT_X)
        xbox.add_axis_mapping(1, self.Axes.LEFT_Y, invert=True)
        xbox.add_axis_mapping(2, self.Axes.ZL, scale=0.5, deadzone=0.05)  # Left Trigger
        xbox.add_axis_mapping(3, self.Axes.RIGHT_X)
        xbox.add_axis_mapping(4, self.Axes.RIGHT_Y, invert=True)
        xbox.add_axis_mapping(5, self.Axes.ZR, scale=0.5, deadzone=0.05)  # Right Trigger
        self.profiles["default_xbox_controller"] = xbox
        
        # Default profile for keyboard and mouse
        km = ControllerProfile("Default Keyboard and Mouse", ControllerType.MOUSE_KEYBOARD)
        # Key mappings set up in update_keyboard
        self.profiles["default_keyboard_mouse"] = km
        
        # Default profile for direct mouse input
        mouse = ControllerProfile("Direct Mouse Input", ControllerType.MOUSE)
        # Mouse position and buttons handled directly
        self.profiles["default_mouse"] = mouse
    
    def load_custom_profiles(self):
        """Load custom controller profiles from configuration directory"""
        profiles_dir = self.config.config_dir / "controller_profiles"
        profiles_dir.mkdir(exist_ok=True)
        
        for profile_file in profiles_dir.glob("*.json"):
            try:
                with open(profile_file, 'r') as f:
                    profile_data = json.load(f)
                
                # Create profile
                profile_name = profile_data.get("name", f"Custom {profile_file.stem}")
                controller_type = ControllerType[profile_data.get("type", "UNKNOWN")]
                profile = ControllerProfile(profile_name, controller_type)
                
                # Load button mappings
                for button_map in profile_data.get("buttons", []):
                    source = button_map.get("source")
                    target = button_map.get("target")
                    if source is not None and target is not None:
                        profile.add_button_mapping(source, target)
                
                # Load axis mappings
                for axis_map in profile_data.get("axes", []):
                    source = axis_map.get("source")
                    target = axis_map.get("target")
                    if source is not None and target is not None:
                        invert = axis_map.get("invert", False)
                        deadzone = axis_map.get("deadzone", 0.1)
                        scale = axis_map.get("scale", 1.0)
                        profile.add_axis_mapping(source, target, invert, deadzone, scale)
                
                # Store the profile
                self.profiles[profile_file.stem] = profile
                self.logger.info(f"Loaded custom controller profile: {profile_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to load controller profile {profile_file}: {e}")
    
    def save_custom_profile(self, profile):
        """Save a custom controller profile
        
        Args:
            profile: ControllerProfile object to save
        """
        profiles_dir = self.config.config_dir / "controller_profiles"
        profiles_dir.mkdir(exist_ok=True)
        
        filename = f"{profile.name.lower().replace(' ', '_')}.json"
        filepath = profiles_dir / filename
        
        try:
            # Build profile data
            profile_data = {
                "name": profile.name,
                "type": profile.controller_type.name,
                "buttons": [
                    {"source": m.source_button, "target": m.target_button}
                    for m in profile.button_mappings
                ],
                "axes": [
                    {
                        "source": m.source_axis,
                        "target": m.target_axis,
                        "invert": m.invert,
                        "deadzone": m.deadzone,
                        "scale": m.scale
                    }
                    for m in profile.axis_mappings
                ]
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(profile_data, f, indent=4)
            
            # Add to profiles
            self.profiles[filepath.stem] = profile
            
            self.logger.info(f"Saved custom controller profile: {profile.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save controller profile {profile.name}: {e}")
            return False
    
    def detect_controller_type(self, name):
        """Detect the type of controller based on its name
        
        Args:
            name: Controller name string
            
        Returns:
            Detected controller type
        """
        name_lower = name.lower()
        
        if "joy-con (l)" in name_lower:
            return ControllerType.JOYCON_LEFT
        elif "joy-con (r)" in name_lower:
            return ControllerType.JOYCON_RIGHT
        elif "pro controller" in name_lower:
            return ControllerType.PRO_CONTROLLER
        elif any(x in name_lower for x in ["xbox", "x-box", "microsoft"]):
            return ControllerType.XBOX_CONTROLLER
        else:
            # Generic fallback - assume pro controller compatible
            return ControllerType.PRO_CONTROLLER
    
    def get_default_profile(self, controller_type):
        """Get the default profile for a controller type
        
        Args:
            controller_type: Type of controller
            
        Returns:
            Default profile for the controller type
        """
        if controller_type == ControllerType.JOYCON_LEFT:
            return self.profiles.get("default_joycon_left")
        elif controller_type == ControllerType.JOYCON_RIGHT:
            return self.profiles.get("default_joycon_right")
        elif controller_type == ControllerType.PRO_CONTROLLER:
            return self.profiles.get("default_pro_controller")
        elif controller_type == ControllerType.XBOX_CONTROLLER:
            return self.profiles.get("default_xbox_controller")
        elif controller_type == ControllerType.MOUSE_KEYBOARD:
            return self.profiles.get("default_keyboard_mouse")
        elif controller_type == ControllerType.MOUSE:
            return self.profiles.get("default_mouse")
        return None
    
    def scan_controllers(self):
        """Scan for connected controllers and initialize them"""
        # Rescan pygame joysticks
        pygame.joystick.quit()
        pygame.joystick.init()
        
        # Track which controllers are still connected
        still_connected = set()
        
        # Check each joystick
        for i in range(pygame.joystick.get_count()):
            try:
                joystick = pygame.joystick.Joystick(i)
                joystick.init()
                
                # Get controller info
                controller_id = joystick.get_instance_id()
                name = joystick.get_name()
                
                if controller_id in self.controllers:
                    # Controller already known
                    still_connected.add(controller_id)
                    continue
                
                # Detect type and create controller
                controller_type = self.detect_controller_type(name)
                controller = Controller(controller_id, controller_type, name)
                
                # Assign default profile
                controller.profile = self.get_default_profile(controller_type)
                
                # Add to active controllers
                self.controllers[controller_id] = controller
                still_connected.add(controller_id)
                
                self.logger.info(f"Controller connected: {name} (Type: {controller_type.name})")
                
            except Exception as e:
                self.logger.error(f"Error initializing controller {i}: {e}")
        
        # Mark disconnected controllers
        for controller_id in list(self.controllers.keys()):
            if controller_id not in still_connected:
                controller = self.controllers[controller_id]
                controller.connected = False
                self.logger.info(f"Controller disconnected: {controller.name}")
                
                # Remove from active controllers
                del self.controllers[controller_id]
        
        # Always add keyboard/mouse as a controller
        if "keyboard_mouse" not in self.controllers:
            controller = Controller("keyboard_mouse", ControllerType.MOUSE_KEYBOARD, "Keyboard and Mouse")
            controller.profile = self.get_default_profile(ControllerType.MOUSE_KEYBOARD)
            self.controllers["keyboard_mouse"] = controller
            
        # Add direct mouse as a controller when enabled
        if self.direct_mouse_enabled and "direct_mouse" not in self.controllers:
            controller = Controller("direct_mouse", ControllerType.MOUSE, "Direct Mouse Input")
            controller.profile = self.get_default_profile(ControllerType.MOUSE)
            self.controllers["direct_mouse"] = controller
    
    def start_polling(self):
        """Start the controller polling thread"""
        if self.polling_thread is None or not self.polling_thread.is_alive():
            self.stop_event.clear()
            self.polling_thread = threading.Thread(target=self._polling_thread, daemon=True)
            self.polling_thread.start()
            self.logger.debug("Controller polling thread started")
    
    def stop_polling(self):
        """Stop the controller polling thread"""
        if self.polling_thread and self.polling_thread.is_alive():
            self.stop_event.set()
            self.polling_thread.join(timeout=1.0)
            self.logger.debug("Controller polling thread stopped")
    
    def _polling_thread(self):
        """Background thread for polling controller inputs"""
        while not self.stop_event.is_set():
            try:
                # Process pygame events
                self.update()
                
                # Sleep to avoid high CPU usage
                time.sleep(0.005)  # 5ms sleep, gives approximately 200Hz polling rate
                
            except Exception as e:
                self.logger.error(f"Error in controller polling thread: {e}")
                time.sleep(0.1)  # Sleep longer on error
    
    def update(self):
        """Update the state of all controllers"""
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.JOYDEVICEADDED or event.type == pygame.JOYDEVICEREMOVED:
                self.scan_controllers()
            elif event.type == pygame.JOYBUTTONDOWN or event.type == pygame.JOYBUTTONUP:
                self.update_controller_button(event)
            elif event.type == pygame.JOYAXISMOTION:
                self.update_controller_axis(event)
            elif event.type == pygame.JOYHATMOTION:
                self.update_controller_hat(event)
            elif event.type == pygame.MOUSEMOTION:
                self.update_mouse_motion(event)
            elif event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.MOUSEBUTTONUP:
                self.update_mouse_button(event)
            elif event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
                self.update_keyboard(event)
    
    def update_controller_button(self, event):
        """Update controller button state from pygame event"""
        controller_id = event.instance_id
        if controller_id not in self.controllers:
            return
        
        controller = self.controllers[controller_id]
        button_id = event.button
        pressed = (event.type == pygame.JOYBUTTONDOWN)
        
        controller.update_button(button_id, pressed)
    
    def update_controller_axis(self, event):
        """Update controller axis state from pygame event"""
        controller_id = event.instance_id
        if controller_id not in self.controllers:
            return
        
        controller = self.controllers[controller_id]
        axis_id = event.axis
        value = event.value
        
        controller.update_axis(axis_id, value)
    
    def update_controller_hat(self, event):
        """Update controller hat (d-pad) state from pygame event"""
        controller_id = event.instance_id
        if controller_id not in self.controllers:
            return
        
        controller = self.controllers[controller_id]
        hat_id = event.hat
        hat_value = event.value
        
        # Convert hat to virtual buttons for d-pad
        if hat_id == 0:  # Usually the only hat
            # D-pad up
            controller.update_button(100 + self.Buttons.DPAD_UP, hat_value[1] > 0)
            # D-pad right
            controller.update_button(100 + self.Buttons.DPAD_RIGHT, hat_value[0] > 0)
            # D-pad down
            controller.update_button(100 + self.Buttons.DPAD_DOWN, hat_value[1] < 0)
            # D-pad left
            controller.update_button(100 + self.Buttons.DPAD_LEFT, hat_value[0] < 0)
    
    def update_mouse_motion(self, event):
        """Update mouse motion state"""
        if "keyboard_mouse" not in self.controllers:
            return
        
        controller = self.controllers["keyboard_mouse"]
        
        # Update absolute mouse position
        self.mouse_absolute_position = event.pos
        
        if self.mouse_locked:
            # Calculate relative motion with sensitivity
            rel_x, rel_y = event.rel
            
            # Apply sensitivity
            rel_x *= self.mouse_sensitivity
            rel_y *= self.mouse_sensitivity
            
            # Accumulate motion for small movements
            self.mouse_motion_accumulator[0] += rel_x
            self.mouse_motion_accumulator[1] += rel_y
            
            # Update mouse position with clamping
            x = max(-1.0, min(1.0, self.mouse_position[0] + rel_x / 100.0))
            y = max(-1.0, min(1.0, self.mouse_position[1] + rel_y / 100.0))
            
            self.mouse_position = (x, y)
            
            # Map to right stick for keyboard/mouse controller
            controller.update_axis(self.Axes.RIGHT_X, x)
            controller.update_axis(self.Axes.RIGHT_Y, y)
            
            # For New Joycons with mouse support
            if self.new_joycon_mouse_enabled:
                # Find all controllers that are Joycon Right type
                for ctrl_id, ctrl in self.controllers.items():
                    if (ctrl.type == ControllerType.JOYCON_RIGHT or 
                        ctrl.type == ControllerType.JOYCON_PAIR):
                        
                        # Map mouse movement to gyro axes
                        # Scale the movement appropriately for gyro simulation
                        gyro_scale = 0.01
                        ctrl.update_axis(100, rel_x * gyro_scale)  # Virtual axis for mouse X
                        ctrl.update_axis(101, rel_y * gyro_scale)  # Virtual axis for mouse Y
                        
                        # We also update motion data to simulate gyro movement
                        current_accel = ctrl.accelerometer
                        current_gyro = ctrl.gyroscope
                        
                        # Create new gyro values based on mouse movement
                        # We're simulating rotation around X and Y axes
                        new_gyro = (
                            current_gyro[0] + (rel_y * gyro_scale), 
                            current_gyro[1] + (rel_x * gyro_scale),
                            current_gyro[2]
                        )
                        
                        # Update the motion data
                        ctrl.update_motion(current_accel, new_gyro)
            
            # For direct mouse input
            if self.direct_mouse_enabled and "direct_mouse" in self.controllers:
                mouse_ctrl = self.controllers["direct_mouse"]
                
                # Set raw mouse position data (normalized from -1 to 1)
                # X value - horizontal position
                mouse_ctrl.update_axis(0, self.mouse_position[0])
                # Y value - vertical position
                mouse_ctrl.update_axis(1, self.mouse_position[1])
                
                # Set relative motion values (used for cursor movement)
                mouse_ctrl.update_axis(2, rel_x * 0.01)  # X movement scaled down
                mouse_ctrl.update_axis(3, rel_y * 0.01)  # Y movement scaled down
        else:
            # Just update stored position when not locked
            x, y = event.pos
            
            # For direct mouse input in unlocked mode, update absolute screen position
            if self.direct_mouse_enabled and "direct_mouse" in self.controllers:
                mouse_ctrl = self.controllers["direct_mouse"]
                
                # Convert screen coordinates to normalized -1 to 1 values
                screen_width, screen_height = pygame.display.get_surface().get_size()
                norm_x = (x / screen_width) * 2 - 1
                norm_y = (y / screen_height) * 2 - 1
                
                # Update absolute position
                mouse_ctrl.update_axis(0, norm_x)
                mouse_ctrl.update_axis(1, norm_y)
                
                # Clear relative motion
                mouse_ctrl.update_axis(2, 0)
                mouse_ctrl.update_axis(3, 0)
    
    def update_mouse_button(self, event):
        """Update mouse button state"""
        if "keyboard_mouse" not in self.controllers:
            return
        
        controller = self.controllers["keyboard_mouse"]
        button_id = event.button - 1  # Pygame starts at 1
        pressed = (event.type == pygame.MOUSEBUTTONDOWN)
        
        if 0 <= button_id <= 2:
            self.mouse_buttons[button_id] = pressed
            
            # Map mouse buttons to controller buttons for keyboard/mouse controller
            if button_id == 0:  # Left mouse button
                controller.update_button(self.Buttons.ZR, pressed)
            elif button_id == 1:  # Middle mouse button
                controller.update_button(self.Buttons.RIGHT_STICK, pressed)
            elif button_id == 2:  # Right mouse button
                controller.update_button(self.Buttons.ZL, pressed)
            
            # For New Joycons with mouse support
            if self.new_joycon_mouse_enabled and self.mouse_locked:
                # Find all controllers that are Joycon Right type
                for ctrl_id, ctrl in self.controllers.items():
                    if (ctrl.type == ControllerType.JOYCON_RIGHT or 
                        ctrl.type == ControllerType.JOYCON_PAIR):
                        
                        # Map mouse buttons to Joycon buttons
                        if button_id == 0:  # Left mouse button
                            ctrl.update_button(2, pressed)  # A button
                        elif button_id == 1:  # Middle mouse button
                            ctrl.update_button(8, pressed)  # Right stick button
                        elif button_id == 2:  # Right mouse button
                            ctrl.update_button(4, pressed)  # B button
            
            # For direct mouse input, map directly to mouse buttons
            if self.direct_mouse_enabled and "direct_mouse" in self.controllers:
                mouse_ctrl = self.controllers["direct_mouse"]
                mouse_ctrl.update_button(button_id, pressed)
            
            # Handle mouse wheel events (buttons 3 and 4)
            if button_id == 3:  # Mouse wheel up
                if self.new_joycon_mouse_enabled and self.mouse_locked:
                    for ctrl_id, ctrl in self.controllers.items():
                        if (ctrl.type == ControllerType.JOYCON_RIGHT or 
                            ctrl.type == ControllerType.JOYCON_PAIR):
                            ctrl.update_button(3, pressed)  # X button
                
                # For direct mouse wheel
                if self.direct_mouse_enabled and "direct_mouse" in self.controllers:
                    mouse_ctrl = self.controllers["direct_mouse"]
                    mouse_ctrl.update_button(3, pressed)  # Wheel up button
            
            elif button_id == 4:  # Mouse wheel down
                if self.new_joycon_mouse_enabled and self.mouse_locked:
                    for ctrl_id, ctrl in self.controllers.items():
                        if (ctrl.type == ControllerType.JOYCON_RIGHT or 
                            ctrl.type == ControllerType.JOYCON_PAIR):
                            ctrl.update_button(5, pressed)  # Y button
                
                # For direct mouse wheel
                if self.direct_mouse_enabled and "direct_mouse" in self.controllers:
                    mouse_ctrl = self.controllers["direct_mouse"]
                    mouse_ctrl.update_button(4, pressed)  # Wheel down button
    
    def update_keyboard(self, event):
        """Update keyboard state"""
        if "keyboard_mouse" not in self.controllers:
            return
        
        controller = self.controllers["keyboard_mouse"]
        key = event.key
        pressed = (event.type == pygame.KEYDOWN)
        
        # Update internal key state
        self.keys[key] = pressed
        
        # Map keyboard keys to controller buttons
        key_mappings = {
            pygame.K_z: self.Buttons.B,
            pygame.K_x: self.Buttons.A,
            pygame.K_a: self.Buttons.Y,
            pygame.K_s: self.Buttons.X,
            pygame.K_q: self.Buttons.L,
            pygame.K_w: self.Buttons.R,
            pygame.K_e: self.Buttons.ZL,
            pygame.K_r: self.Buttons.ZR,
            pygame.K_BACKSPACE: self.Buttons.MINUS,
            pygame.K_RETURN: self.Buttons.PLUS,
            pygame.K_ESCAPE: self.Buttons.HOME,
            pygame.K_F1: self.Buttons.SHARE,
            pygame.K_SPACE: self.Buttons.LEFT_STICK,
            pygame.K_UP: self.Buttons.DPAD_UP,
            pygame.K_RIGHT: self.Buttons.DPAD_RIGHT,
            pygame.K_DOWN: self.Buttons.DPAD_DOWN,
            pygame.K_LEFT: self.Buttons.DPAD_LEFT
        }
        
        # Update buttons based on key mappings
        if key in key_mappings:
            controller.update_button(key_mappings[key], pressed)
        
        # Update axes based on WASD keys
        left_x = 0.0
        left_y = 0.0
        
        if self.keys.get(pygame.K_d, False):
            left_x += 1.0
        if self.keys.get(pygame.K_a, False):
            left_x -= 1.0
        if self.keys.get(pygame.K_w, False):
            left_y -= 1.0
        if self.keys.get(pygame.K_s, False):
            left_y += 1.0
        
        # Normalize diagonal movement
        if left_x != 0.0 and left_y != 0.0:
            magnitude = (left_x**2 + left_y**2)**0.5
            left_x /= magnitude
            left_y /= magnitude
        
        controller.update_axis(self.Axes.LEFT_X, left_x)
        controller.update_axis(self.Axes.LEFT_Y, left_y)
    
    def lock_mouse(self, window_surface):
        """Lock the mouse to the window for pointer control
        
        Args:
            window_surface: Pygame surface of the window
            
        Returns:
            True if the mouse was locked, False otherwise
        """
        if not self.mouse_locked:
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
            self.mouse_locked = True
            self.mouse_position = (0.0, 0.0)  # Center position
            self.mouse_motion_accumulator = [0, 0]  # Reset accumulator
            
            # Reset controller axis
            if "keyboard_mouse" in self.controllers:
                controller = self.controllers["keyboard_mouse"]
                controller.update_axis(self.Axes.RIGHT_X, 0.0)
                controller.update_axis(self.Axes.RIGHT_Y, 0.0)
            
            # Also reset any Joycon right stick positions
            if self.new_joycon_mouse_enabled:
                for ctrl_id, ctrl in self.controllers.items():
                    if (ctrl.type == ControllerType.JOYCON_RIGHT or 
                        ctrl.type == ControllerType.JOYCON_PAIR):
                        ctrl.update_axis(0, 0.0)  # RIGHT_X
                        ctrl.update_axis(1, 0.0)  # RIGHT_Y
                        # Reset virtual mouse axes
                        ctrl.update_axis(100, 0.0)
                        ctrl.update_axis(101, 0.0)
            
            self.logger.debug("Mouse locked for pointer control")
            return True
        
        return False
    
    def unlock_mouse(self):
        """Unlock the mouse from pointer control
        
        Returns:
            True if the mouse was unlocked, False otherwise
        """
        if self.mouse_locked:
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)
            self.mouse_locked = False
            self.mouse_motion_accumulator = [0, 0]  # Reset accumulator
            
            # Reset controller axis
            if "keyboard_mouse" in self.controllers:
                controller = self.controllers["keyboard_mouse"]
                controller.update_axis(self.Axes.RIGHT_X, 0.0)
                controller.update_axis(self.Axes.RIGHT_Y, 0.0)
            
            # Also reset any Joycon right stick positions
            if self.new_joycon_mouse_enabled:
                for ctrl_id, ctrl in self.controllers.items():
                    if (ctrl.type == ControllerType.JOYCON_RIGHT or 
                        ctrl.type == ControllerType.JOYCON_PAIR):
                        ctrl.update_axis(0, 0.0)  # RIGHT_X
                        ctrl.update_axis(1, 0.0)  # RIGHT_Y
                        # Reset virtual mouse axes
                        ctrl.update_axis(100, 0.0)
                        ctrl.update_axis(101, 0.0)
            
            self.logger.debug("Mouse unlocked from pointer control")
            return True
        
        return False
    
    def set_mouse_sensitivity(self, sensitivity):
        """Set the mouse sensitivity
        
        Args:
            sensitivity: Sensitivity value (0.1 to 3.0)
            
        Returns:
            The new sensitivity value
        """
        self.mouse_sensitivity = max(0.1, min(3.0, sensitivity))
        self.logger.debug(f"Mouse sensitivity set to {self.mouse_sensitivity}")
        return self.mouse_sensitivity
    
    def toggle_mouse_acceleration(self):
        """Toggle mouse acceleration
        
        Returns:
            The new state of mouse acceleration
        """
        self.mouse_acceleration = not self.mouse_acceleration
        self.logger.debug(f"Mouse acceleration {'enabled' if self.mouse_acceleration else 'disabled'}")
        return self.mouse_acceleration
    
    def toggle_new_joycon_mouse(self):
        """Toggle mouse support for New Joycons
        
        Returns:
            The new state of New Joycon mouse support
        """
        self.new_joycon_mouse_enabled = not self.new_joycon_mouse_enabled
        self.logger.debug(f"New Joycon mouse support {'enabled' if self.new_joycon_mouse_enabled else 'disabled'}")
        return self.new_joycon_mouse_enabled
    
    def toggle_direct_mouse(self):
        """Toggle direct mouse input
        
        Returns:
            The new state of direct mouse support
        """
        self.direct_mouse_enabled = not self.direct_mouse_enabled
        
        # Add or remove the direct mouse controller
        if self.direct_mouse_enabled:
            if "direct_mouse" not in self.controllers:
                controller = Controller("direct_mouse", ControllerType.MOUSE, "Direct Mouse Input")
                controller.profile = self.get_default_profile(ControllerType.MOUSE)
                self.controllers["direct_mouse"] = controller
        else:
            if "direct_mouse" in self.controllers:
                del self.controllers["direct_mouse"]
        
        self.logger.debug(f"Direct mouse input {'enabled' if self.direct_mouse_enabled else 'disabled'}")
        return self.direct_mouse_enabled
    
    def get_button_state(self, button):
        """Get the state of a controller button
        
        Args:
            button: Button ID from Buttons enum
            
        Returns:
            True if the button is pressed on any controller, False otherwise
        """
        for controller in self.controllers.values():
            if controller.get_target_button_state(button):
                return True
        
        return False
    
    def get_axis_value(self, axis):
        """Get the value of a controller axis
        
        Args:
            axis: Axis ID from Axes enum
            
        Returns:
            Axis value from -1.0 to 1.0
        """
        # For multiple controllers, we want to use the largest absolute value
        max_value = 0.0
        
        for controller in self.controllers.values():
            value = controller.get_target_axis_value(axis)
            
            # Keep the value with the largest magnitude
            if abs(value) > abs(max_value):
                max_value = value
        
        return max_value 