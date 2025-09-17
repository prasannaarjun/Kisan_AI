#!/usr/bin/env python3
"""
Startup script for KisanAI backend
"""
import os
import sys
import subprocess
import logging

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
        print("Dependencies installed successfully!")
        return
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        print("Please try installing manually:")
        print("pip install -r requirements.txt")
        sys.exit(1)

def start_server():
    """Start the FastAPI server"""
    print("Starting KisanAI backend server...")
    os.chdir('backend')
    try:
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🌾 KisanAI Backend Startup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists('backend'):
        print("Error: Please run this script from the project root directory")
        sys.exit(1)
    
    setup_environment()
    # Skip dependency installation
    print("⚠️  Skipping dependency installation...")
    print("Make sure all dependencies are already installed.")
    start_server()
