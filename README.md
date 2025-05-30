# Ree2Jinx (Codename ImaginaryConsoleEmulator)

Attempting to make an emulator of an "imaginary" console. I say "imaginary" because I'm pretty much pulling the specs out of my rear. It is 100% not a certain console by a specific red company with a plumber mascot! No, not at all!

This project is for educational purposes and demonstrates various emulation concepts.

## Console Specifications

The emulated console has the following hardware specifications:

- **CPU**: ARM Cortex-A78c (8 cores, ARM64, ARMv8.2-A, 8MB L3 cache support)
- **GPU**: Nvidia T239 Ampere (1 graphics cluster, 12 streaming multiprocesseors, 1534 CUDA cores)
- **RAM**: 12GB LPDDR5
- **Storage**: 256GB
- **Handheld Mode**: 
  - CPU: 998.4MHz
  - GPU: 561MHz (~1.72 TFLOPS)
  - Memory Frequency: 4266MHz
  - Memory Bandwidth: 68.256 GB/s
- **Docked Mode**: 
  - CPU: 1100.8MHz
  - GPU: 1007.25MHz (~3.09 TFLOPS)
  - Memory Frequency: 6400MHz
  - Memory Bandwidth: 102.4 GB/s

## Will you improve it after Ninento releases the Switch 2?

***Nintendo? Nintendo who?***

Just kidding! We will continue to improve this emulator after Nintendo releases their console because we are:

1. Confident that the specs will be different than that of this imaginary emulator.
2. Confident that Nintendo learned their security lessons with Switch 1.
3. Making an educational project. Even if the big N makes a VERY similar console, say perhaps a second Switch, we can say "oooh we predicted that" and carry on!
4. Not harming Nintendo in any way. We don't include the keys or ROMs, and we purposefully nerfed the emulator by writing it in Python so it will run so slow nobody will want to play it until about when the Switch 3 comes out! (~10 years from now.)

Note it is NOT our goal to make it slow beyond writing it in Python, and we even added bare-minimum CUDA support for all you NVIDIA users!

## Features

- Emulates the core hardware components of the imaginary console
- Supports both handheld and docked operational modes
- Provides joycon and pro controller emulation
- Supports Xbox controllers as pro controllers
- Mouse and keyboard input support
- Advanced controller input management with support for multiple controller types, including Joy-Cons, Pro Controllers, and Xbox controllers.
- Customizable controller mapping profiles.
- Firmware and ROM loading with decryption
- Save state functionality
- OpenGL-accelerated rendering

## Requirements

The emulator requires the following dependencies:

- Python 3.8 or newer
- NumPy
- PyGame
- PyOpenGL
- CuPy (for CUDA acceleration)
- PyUSB
- Cryptography
- Pillow
- Keyboard and Mouse libraries

You can install the dependencies using the provided `requirements.txt` file:

```
pip install -r requirements.txt
```

## Usage

### Running the Emulator

To start the emulator with default settings:

```
python main.py
```

### Command Line Options

The following command-line options are available:

- `--rom PATH`: Path to a ROM file to load
- `--keys PATH`: Path to a keys file for ROM decryption
- `--firmware PATH`: Path to firmware directory
- `--docked`: Start in docked mode instead of handheld mode
- `--fullscreen`: Start in fullscreen mode

### Controls

#### Keyboard Controls

- Arrow keys: D-pad
- Z/X/A/S: B/A/Y/X buttons
- Q/W/E/R: L/R/ZL/ZR buttons
- Backspace/Enter: Minus/Plus buttons
- Escape: Home button
- F1: Share button
- Space: Left stick button
- WASD: Left analog stick
- Mouse movement (when locked): Primarily controls the right analog stick. See 'New Joycons Mouse Support' for additional gyro mapping.

#### Mouse Controls

- Left click inside the window: Lock/unlock mouse for console pointer control
- Left/Right/Middle click (when mouse is locked): ZR/ZL/Right stick buttons
- Mouse wheel up/down: Wheel events
- F2: Open Mouse Settings panel to adjust sensitivity and toggle mouse options

#### Controller Support

The emulator supports various controllers through SDL:

- Joycons (both left and right)
  - New Joycons support mouse input for pointer control
- Pro controllers
- Xbox controllers (mapped as pro controllers)
- Other SDL-compatible controllers
- Supports customizable controller profiles (JSON format) for button and axis mappings, typically loaded from a `controller_profiles` subdirectory within your configuration path.

### New Joycons Mouse Support

The "New Joycons" feature mouse input integration:

- When mouse is locked **and 'Enable for Joycons' is active in Mouse Settings**, mouse movement is *additionally* mapped to simulate gyro controls for Right Joy-Cons or Joy-Con Pairs. This is alongside its primary function of controlling the right analog stick.
- Mouse buttons are mapped to Joycon buttons:
  - Left click: A button
  - Right click: B button
  - Middle click: Right stick button
  - Mouse wheel up/down: X/Y buttons
- Sensitivity can be adjusted in the settings menu

## File Structure

- `main.py`: Main entry point for the emulator
- `config.py`: Configuration and hardware specifications
- `hardware/`: Hardware emulation components
  - `cpu.py`: ARM CPU emulation
  - `gpu.py`: NVIDIA GPU emulation
  - `memory.py`: Memory subsystem
  - `storage.py`: Storage subsystem
  - `system.py`: Overall hardware system
- `input/`: Input handling
  - `controller_manager.py`: Controller detection and input management
- `system/`: System-level functionality
  - `firmware_manager.py`: Firmware loading and verification
  - `rom_loader.py`: ROM loading and decryption
- `ui/`: User interface
  - `window.py`: Main window and rendering
  - `gui.py`: UI utilities and file dialogs
- `assets/`: Graphics and other assets

## Project Status

This is an educational project intended to demonstrate emulation concepts. The emulator does not run 
actual commercial games, nor is it intended to do so.

## License

This project is released under the MIT License. See the LICENSE file for details.

## Disclaimer

This project is released as an educational project. You should not attempt to use any copyrighted materials with this emulator. To the big red company: Don't sue please! 