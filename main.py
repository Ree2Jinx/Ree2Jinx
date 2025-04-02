#!/usr/bin/env python3
"""
ImaginaryConsole Emulator - Main Entry Point
This module serves as the main entry point for the emulator, initializing all components
and managing the main execution loop.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Internal modules
from config import Config
from hardware.system import System
from ui.window import Window
from input.controller_manager import ControllerManager
from system.rom_loader import RomLoader
from system.firmware_manager import FirmwareManager


def setup_logging():
    """Configure the logging system for the emulator"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "emulator.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("ImaginaryConsole")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="ImaginaryConsole Emulator")
    parser.add_argument("--rom", type=str, help="Path to ROM file to load")
    parser.add_argument("--keys", type=str, help="Path to keys file")
    parser.add_argument("--firmware", type=str, help="Path to firmware directory")
    parser.add_argument("--docked", action="store_true", help="Start in docked mode instead of handheld mode")
    parser.add_argument("--fullscreen", action="store_true", help="Start in fullscreen mode")
    return parser.parse_args()


def main():
    """Main entry point for the emulator"""
    logger = setup_logging()
    logger.info("Starting ImaginaryConsole Emulator")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize configuration
    config = Config()
    config.load_config()
    
    # Override config with command line arguments if provided
    if args.docked:
        config.set_docked_mode(True)
    
    # Initialize system components
    try:
        # Setup firmware and keys
        firmware_manager = FirmwareManager(args.firmware or config.firmware_path)
        firmware_manager.load_firmware()
        
        # Initialize hardware
        system = System(config)
        
        # Initialize input management
        controller_manager = ControllerManager(config)
        
        # Create main window
        window = Window(config)
        window.register_input_manager(controller_manager)
        
        # Load ROM if specified
        if args.rom:
            rom_loader = RomLoader(args.rom, args.keys or config.keys_path)
            system.load_rom(rom_loader)
        
        # Start the emulation loop
        logger.info("Entering main emulation loop")
        window.run(system)
        
    except Exception as e:
        logger.error(f"Error during emulation: {e}", exc_info=True)
        return 1
    
    logger.info("Emulator shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 