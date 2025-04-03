import base64
import os
from pathlib import Path

# A small base64-encoded PNG file (16x16 blue square with white border)
ICON_DATA = """
iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAA5ElEQVQ4T82TPQ6CQBCFZwikpoLEQE3D
DSg9gdFwIUkJsfQMXoBQaIgJNYUVJcEY34OsCbhofgqdbN7O2817O6vUsgUiCKlhZl+ZJtKHWWWw64Y7
gCa3XAKOqRXoug45iiI4jgNxHENRFPF0XZcW2LatsizD9QBAF3DOMU1TsixLtm0bfd9nWZYBAFiWJU3T
xLIs0TRNLIoCu66DIAiQc05SQgiS53nKfiuu60aMMYqiSPq+L23bghACm6ZBXdfSey/8L8Hx54sLQHM1
TcPhcKj6HpyBHxEAfNcg46XqG9dM5ER8AHLL+SyKIrnbAAAAAElFTkSuQmCC
"""

def create_simple_icon():
    # Ensure assets directory exists
    assets_path = Path(os.path.dirname(os.path.abspath(__file__))) / "assets"
    assets_path.mkdir(parents=True, exist_ok=True)
    
    # Decode the base64 data
    icon_data = base64.b64decode(ICON_DATA.strip())
    
    # Save the icon
    icon_path = assets_path / "icon.png"
    with open(icon_path, "wb") as f:
        f.write(icon_data)
    
    print(f"Icon created at {icon_path}")

if __name__ == "__main__":
    create_simple_icon() 