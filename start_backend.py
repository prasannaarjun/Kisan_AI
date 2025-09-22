#!/usr/bin/env python3
"""
Startup script for KisanAI backend
"""
import os
import sys
import subprocess
import logging
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

def setup_environment():
    """Set up environment variables"""
    os.environ.setdefault('SERVER_HOST', '0.0.0.0')
    os.environ.setdefault('SERVER_PORT', '8000')
    os.environ.setdefault('LOG_LEVEL', 'INFO')
    os.environ.setdefault('ASR_MODEL', 'ai4bharat/indic-conformer-600m-multilingual')
    os.environ.setdefault('LLM_MODEL', 'google/gemma-2-9b-it')
    os.environ.setdefault('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

def install_dependencies():
    """Install Python dependencies"""
    print("Installing dependencies...")
    
    try:
        print("Installing from requirements.txt...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✓ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        print("Please try installing manually:")
        print("pip install -r requirements.txt")
        return False

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting KisanAI backend server...")
    try:
        # Add current directory to Python path for module imports
        current_dir = os.getcwd()
        sys.path.insert(0, current_dir)
        
        print("✓ Backend server starting on http://localhost:8000")
        print("✓ API documentation available at http://localhost:8000/docs")
        
        # Use module approach with proper environment
        env = os.environ.copy()
        env['PYTHONPATH'] = current_dir
        
        subprocess.run([sys.executable, '-m', 'backend.app'], env=env)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"✗ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🌾 KisanAI Backend Startup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists('backend'):
        print("✗ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Activate virtual environment
    activate_virtual_environment()
    
    # Set up environment variables
    setup_environment()
    
    # Skip dependency installation
    print("⚠️  Skipping dependency installation...")
    print("Make sure all dependencies are already installed.")
    
    # Start the server
    start_server()
