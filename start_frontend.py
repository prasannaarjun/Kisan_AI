#!/usr/bin/env python3
"""
Startup script for KisanAI frontend
"""
import os
import sys
import subprocess

def check_node():
    """Check if Node.js is installed"""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Node.js version: {result.stdout.strip()}")
            return True
        else:
            print("Node.js not found")
            return False
    except FileNotFoundError:
        print("Node.js not found")
        return False

def install_dependencies():
    """Install npm dependencies"""
    print("Installing frontend dependencies...")
    try:
        subprocess.check_call(['npm', 'install'], cwd='frontend')
        print("Frontend dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def start_frontend():
    """Start the React development server"""
    print("Starting KisanAI frontend...")
    os.chdir('frontend')
    try:
        subprocess.run(['npm', 'start'])
    except KeyboardInterrupt:
        print("\nFrontend stopped by user")
    except Exception as e:
        print(f"Error starting frontend: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🌾 KisanAI Frontend Startup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists('frontend'):
        print("Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Skip Node.js check and dependency installation
    print("⚠️  Skipping Node.js check and dependency installation...")
    print("Make sure Node.js and all dependencies are already installed.")
    
    start_frontend()
