#!/usr/bin/env python3
"""
Startup script for KisanAI frontend
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

def activate_virtual_environment():
    """Activate virtual environment if it exists"""
    venv_path = Path('.venv')
    if venv_path.exists():
        if platform.system() == "Windows":
            activate_script = venv_path / "Scripts" / "activate.bat"
            if activate_script.exists():
                print("✓ Virtual environment found, activating...")
                # Set environment variables for Windows
                os.environ['VIRTUAL_ENV'] = str(venv_path.absolute())
                python_exe = venv_path / "Scripts" / "python.exe"
                if python_exe.exists():
                    sys.executable = str(python_exe)
                    print(f"✓ Using Python from virtual environment: {python_exe}")
                    # Add Scripts directory to PATH for npm and other tools
                    scripts_path = str(venv_path / "Scripts")
                    current_path = os.environ.get('PATH', '')
                    if scripts_path not in current_path:
                        os.environ['PATH'] = scripts_path + os.pathsep + current_path
                return True
        else:
            activate_script = venv_path / "bin" / "activate"
            if activate_script.exists():
                print("✓ Virtual environment found, activating...")
                # Set environment variables for Unix-like systems
                os.environ['VIRTUAL_ENV'] = str(venv_path.absolute())
                python_exe = venv_path / "bin" / "python"
                if python_exe.exists():
                    sys.executable = str(python_exe)
                    print(f"✓ Using Python from virtual environment: {python_exe}")
                    # Add bin directory to PATH for npm and other tools
                    bin_path = str(venv_path / "bin")
                    current_path = os.environ.get('PATH', '')
                    if bin_path not in current_path:
                        os.environ['PATH'] = bin_path + os.pathsep + current_path
                return True
    else:
        print("⚠️  No virtual environment found, using system Python")
    return False

def check_node():
    """Check if Node.js is installed"""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Node.js version: {result.stdout.strip()}")
            return True
        else:
            print("✗ Node.js not found")
            return False
    except FileNotFoundError:
        print("✗ Node.js not found")
        return False

def install_dependencies():
    """Install npm dependencies"""
    print("Installing frontend dependencies...")
    try:
        subprocess.check_call(['npm', 'install'], cwd='frontend')
        print("✓ Frontend dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        return False

def start_frontend():
    """Start the React development server"""
    print("🚀 Starting KisanAI frontend...")
    os.chdir('frontend')
    try:
        print("✓ Frontend server starting on http://localhost:3000")
        # Use shell=True on Windows to properly handle npm commands
        if platform.system() == "Windows":
            subprocess.run(['npm', 'start'], shell=True)
        else:
            subprocess.run(['npm', 'start'])
    except KeyboardInterrupt:
        print("\n🛑 Frontend stopped by user")
    except Exception as e:
        print(f"✗ Error starting frontend: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🌾 KisanAI Frontend Startup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists('frontend'):
        print("✗ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Activate virtual environment
    activate_virtual_environment()
    
    # Skip Node.js check and dependency installation
    print("⚠️  Skipping Node.js check and dependency installation...")
    print("Make sure Node.js and all dependencies are already installed.")
    
    # Start the frontend
    start_frontend()
