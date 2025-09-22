#!/usr/bin/env python3
"""
KisanAI - Complete Application Launcher
Runs both backend and frontend with proper process management
"""
import os
import sys
import subprocess
import time
import signal
import platform
from pathlib import Path

class KisanAILauncher:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.running = False
        self.venv_activated = False
        
    def activate_virtual_environment(self):
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
                        self.venv_activated = True
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
                        self.venv_activated = True
                        return True
        else:
            print("⚠️  No virtual environment found, using system Python")
        return False
        
    def check_dependencies(self):
        """Check if all dependencies are available"""
        print("🔍 Checking dependencies...")
        
        # Check Python
        try:
            result = subprocess.run([sys.executable, '--version'], capture_output=True, text=True)
            print(f"✓ Python: {result.stdout.strip()}")
        except Exception as e:
            print(f"✗ Python not found: {e}")
            return False
        
        # Check Node.js
        try:
            result = subprocess.run(['node', '--version'], capture_output=True, text=True)
            print(f"✓ Node.js: {result.stdout.strip()}")
        except Exception as e:
            print(f"✗ Node.js not found: {e}")
            return False
        
        # Check if backend directory exists
        if not os.path.exists('backend'):
            print("✗ Backend directory not found")
            return False
        
        # Check if frontend directory exists
        if not os.path.exists('frontend'):
            print("✗ Frontend directory not found")
            return False
        
        print("✓ All dependencies available")
        return True
    
    def install_backend_dependencies(self):
        """Install backend Python dependencies"""
        print("\n📦 Installing backend dependencies...")
        
        # Detect Windows and choose appropriate requirements file
        import platform
        is_windows = platform.system() == "Windows"
        
        requirements_files = []
        if is_windows:
            print("Windows detected, using Windows-compatible requirements...")
            requirements_files = [
                'backend/requirements-stable.txt',
                'backend/requirements-windows.txt',
                'backend/requirements-compatible.txt',
                'backend/requirements.txt'
            ]
        else:
            requirements_files = [
                'backend/requirements-stable.txt',
                'backend/requirements.txt',
                'backend/requirements-compatible.txt'
            ]
        
        for req_file in requirements_files:
            try:
                print(f"Trying {req_file}...")
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', '-r', req_file
                ])
                print(f"✓ Backend dependencies installed using {req_file}")
                return True
            except subprocess.CalledProcessError as e:
                print(f"Failed with {req_file}: {e}")
                continue
        
        print("✗ All requirements files failed")
        print("Please try installing manually: pip install -r backend/requirements-windows.txt")
        return False
    
    def install_frontend_dependencies(self):
        """Install frontend Node.js dependencies"""
        print("\n📦 Installing frontend dependencies...")
        try:
            subprocess.check_call(['npm', 'install'], cwd='frontend')
            print("✓ Frontend dependencies installed")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install frontend dependencies: {e}")
            return False
    
    def start_backend(self):
        """Start the backend server"""
        print("\n🚀 Starting backend server...")
        try:
            # Add current directory to Python path for module imports
            current_dir = os.getcwd()
            sys.path.insert(0, current_dir)
            
            # Use module approach with proper environment
            env = os.environ.copy()
            env['PYTHONPATH'] = current_dir
            
            self.backend_process = subprocess.Popen([
                sys.executable, '-m', 'backend.app'
            ], env=env)
            
            # Wait a moment for server to start
            time.sleep(3)
            
            if self.backend_process.poll() is None:
                print("✓ Backend server started on http://localhost:8000")
                return True
            else:
                print("✗ Backend server failed to start")
                return False
        except Exception as e:
            print(f"✗ Error starting backend: {e}")
            return False
    
    def start_frontend(self):
        """Start the frontend development server"""
        print("\n🚀 Starting frontend server...")
        try:
            # Use shell=True on Windows to properly handle npm commands
            if platform.system() == "Windows":
                self.frontend_process = subprocess.Popen([
                    'npm', 'start'
                ], cwd='frontend', shell=True)
            else:
                self.frontend_process = subprocess.Popen([
                    'npm', 'start'
                ], cwd='frontend')
            
            # Wait a moment for server to start
            time.sleep(5)
            
            if self.frontend_process.poll() is None:
                print("✓ Frontend server started on http://localhost:3000")
                return True
            else:
                print("✗ Frontend server failed to start")
                return False
        except Exception as e:
            print(f"✗ Error starting frontend: {e}")
            return False
    
    def monitor_processes(self):
        """Monitor running processes"""
        while self.running:
            if self.backend_process and self.backend_process.poll() is not None:
                print("\n❌ Backend process stopped unexpectedly")
                self.running = False
                break
            
            if self.frontend_process and self.frontend_process.poll() is not None:
                print("\n❌ Frontend process stopped unexpectedly")
                self.running = False
                break
            
            time.sleep(1)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n\n🛑 Shutting down KisanAI...")
        self.running = False
        
        if self.backend_process:
            self.backend_process.terminate()
            print("✓ Backend stopped")
        
        if self.frontend_process:
            self.frontend_process.terminate()
            print("✓ Frontend stopped")
        
        sys.exit(0)
    
    def run(self):
        """Main run method"""
        print("🌾 KisanAI - Agricultural AI Assistant")
        print("=" * 50)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Activate virtual environment
        self.activate_virtual_environment()
        
        # Skip dependency checks and installation
        print("⚠️  Skipping dependency checks and installation...")
        print("Make sure all dependencies are already installed.")
        
        # Start servers
        if not self.start_backend():
            print("\n❌ Failed to start backend server.")
            sys.exit(1)
        
        if not self.start_frontend():
            print("\n❌ Failed to start frontend server.")
            self.stop_backend()
            sys.exit(1)
        
        # Set running flag
        self.running = True
        
        print("\n🎉 KisanAI is now running!")
        print("=" * 50)
        print("📱 Frontend: http://localhost:3000")
        print("🔧 Backend API: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop the application")
        print("=" * 50)
        
        # Start monitoring
        try:
            self.monitor_processes()
        except KeyboardInterrupt:
            self.signal_handler(signal.SIGINT, None)
    
    def stop_backend(self):
        """Stop backend process"""
        if self.backend_process:
            self.backend_process.terminate()
            self.backend_process.wait()

def main():
    """Main entry point"""
    launcher = KisanAILauncher()
    launcher.run()

if __name__ == "__main__":
    main()
