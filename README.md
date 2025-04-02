# Ree2Jinx (Codename ImaginaryConsoleEmulator)

Attempting to make a NS2 emulator pre-release.

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

## Features

- Emulates the core hardware components of the imaginary console
- Supports both handheld and docked operational modes
- Provides joycon and pro controller emulation
- Supports Xbox controllers as pro controllers
- Mouse and keyboard input support
- Firmware loading with NCA file support
- ROM loading with decryption
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
- `--firmware PATH`: Path to firmware directory containing .nca files
- `--docked`: Start in docked mode instead of handheld mode
- `--fullscreen`: Start in fullscreen mode

### Firmware Structure

The emulator now uses .nca files for firmware. These files should be placed in the firmware directory:

- The firmware directory should contain .nca files with the following types:
  - PROGRAM (required): Contains the main system program code
  - CONTROL (required): Contains control data for the system
  - DATA (required): Contains system data
  - META (required): Contains metadata including version information
  - PUBLIC_DATA (optional): Contains additional public data

Example firmware directory structure:
```
firmware/
  ├── system_program.nca (PROGRAM type)
  ├── system_control.nca (CONTROL type)
  ├── system_data.nca (DATA type)
  ├── system_meta.nca (META type)
  ├── system_public.nca (PUBLIC_DATA type, optional)
  └── firmware_info.json (automatically generated)
```

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
- Mouse: Right analog stick (when locked)

#### Mouse Controls

- Left click inside the window: Lock/unlock mouse for console pointer control
- Left/Right/Middle click (when mouse is locked): ZR/ZL/Right stick buttons
- Mouse wheel up/down: Wheel events
- F2: Open Mouse Settings panel to adjust sensitivity and toggle mouse options

#### Direct Mouse Support

The emulator supports the mouse as a direct input device for the imaginary console:

- When "Enable as System Mouse" is enabled (toggled in Mouse Settings), the mouse becomes a dedicated input device for the system
- Mouse movements are sent directly to the system as raw mouse input rather than being mapped to controller buttons
- Mouse buttons are recognized as their own discrete inputs separate from controller buttons
- Both absolute position (cursor location) and relative motion (movement) data are tracked

#### Controller Support

The emulator supports various controllers through SDL:

- Joycons (both left and right)
  - New Joycons support mouse input for pointer control
- Pro controllers
- Xbox controllers (mapped as pro controllers)
- Other SDL-compatible controllers

### New Joycons Mouse Support

The "New Joycons" feature mouse input integration:

- When mouse is locked, its movement is mapped to gyro controls
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
  - `firmware_manager.py`: Firmware loading and verification (supports .nca files)
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

This is an educational project for learning about hardware emulation concepts. The console it 
emulates is imaginary (i.e. not public) and does not actually exist. No real console firmware or ROMs will work 
with this emulator (sadly), nor should you attempt to use any copyrighted materials you don't own with it. This is all built on rumored specs - so expect inaccuracies and bugs, even with homebrew.