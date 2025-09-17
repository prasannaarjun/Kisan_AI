# 🔧 KisanAI Troubleshooting Guide

## Common Installation Issues

### 1. Python Version Compatibility

**Issue**: `ERROR: Could not find a version that satisfies the requirement...`

**Solution**: 
- Ensure you're using Python 3.8-3.11
- Check your Python version: `python --version`
- If using Python 3.12+, try the compatible requirements:
  ```bash
  pip install -r backend/requirements-compatible.txt
  ```

### 2. PyTorch Installation Issues

**Issue**: PyTorch installation fails or CUDA issues

**Solutions**:
```bash
# For CPU-only installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2.1. Windows Torchaudio Compatibility Issues

**Issue**: `OSError: [WinError 127] The specified procedure could not be found` with torchaudio

**Quick Fix**:
```bash
python fix_windows_audio.py
```

**Manual Fix**:
```bash
# Uninstall problematic torchaudio
pip uninstall torchaudio -y

# Install compatible versions
pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Or install without torchaudio (uses scipy fallback)
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
```

### 2.2. Transformers Compatibility Issues

**Issue**: `AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'`

**Quick Fix**:
```bash
python fix_compatibility.py
```

**Manual Fix**:
```bash
# Uninstall problematic packages
pip uninstall torch transformers accelerate -y

# Install stable versions
pip install torch==2.0.1 transformers==4.30.2 accelerate==0.20.3

# Or use stable requirements
pip install -r backend/requirements-stable.txt
```

### 2.3. HuggingFace Hub Compatibility Issues

**Issue**: `ImportError: cannot import name 'cached_download' from 'huggingface_hub'`

**Quick Fix**:
```bash
python fix_huggingface.py
```

**Manual Fix**:
```bash
# Uninstall problematic packages
pip uninstall sentence-transformers huggingface_hub -y

# Install compatible versions
pip install huggingface_hub==0.16.4 sentence-transformers==2.2.2
```

### 2.4. Windows Signal Compatibility Issues

**Issue**: `AttributeError: module 'signal' has no attribute 'SIGALRM'`

**Quick Fix**:
```bash
python fix_windows_signal.py
```

**Manual Fix**:
```bash
# Install Windows-compatible versions
pip install transformers==4.30.2 --force-reinstall
pip install openai-whisper

# Or use the stable requirements
pip install -r backend/requirements-stable.txt
```

### 3. NumPy/SciPy Version Conflicts

**Issue**: `ERROR: Ignored the following versions that require a different python version`

**Solution**:
```bash
# Upgrade pip first
pip install --upgrade pip

# Install compatible versions
pip install "numpy>=1.21.0,<1.25.0" "scipy>=1.9.0,<1.12.0"
```

### 4. FAISS Installation Issues

**Issue**: FAISS installation fails

**Solutions**:
```bash
# Try CPU version
pip install faiss-cpu

# Or try conda
conda install -c conda-forge faiss-cpu
```

### 5. Transformers Model Download Issues

**Issue**: Models fail to download or load

**Solutions**:
```bash
# Set environment variables for offline mode
export HF_HOME=./models
export TRANSFORMERS_CACHE=./models

# Or use a different model
export ASR_MODEL=microsoft/speecht5_asr
export LLM_MODEL=microsoft/DialoGPT-medium
```

### 6. WebSocket Connection Issues

**Issue**: Frontend can't connect to backend

**Solutions**:
1. Check if backend is running on port 8000
2. Verify CORS settings in `backend/app.py`
3. Check firewall settings
4. Try accessing `http://localhost:8000/docs` in browser

### 7. Microphone Permission Issues

**Issue**: Microphone not working in browser

**Solutions**:
1. Ensure HTTPS or localhost (browsers require secure context)
2. Check browser microphone permissions
3. Try different browsers (Chrome, Firefox, Edge)
4. Check system microphone settings

### 8. Memory Issues

**Issue**: Out of memory errors

**Solutions**:
```bash
# Use smaller models
export ASR_MODEL=openai/whisper-tiny
export LLM_MODEL=microsoft/DialoGPT-small

# Or reduce batch sizes in the code
```

### 9. Database Issues

**Issue**: SQLite database errors

**Solutions**:
```bash
# Delete existing database and restart
rm backend/chat_sessions.db
python start_backend.py
```

### 10. Node.js Issues

**Issue**: Frontend won't start

**Solutions**:
```bash
# Check Node.js version (should be 16+)
node --version

# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf frontend/node_modules
cd frontend
npm install
```

## Manual Installation Steps

If automated installation fails, try manual installation:

### Backend
```bash
cd backend

# Install core dependencies
pip install fastapi uvicorn pydantic websockets

# Install ML dependencies
pip install torch transformers langchain langgraph

# Install audio processing
pip install librosa soundfile

# Install vector search
pip install faiss-cpu sentence-transformers

# Install other utilities
pip install python-dotenv langid numpy scipy

# Run the application
python app.py
```

### Frontend
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

## Environment Variables

Create a `.env` file in the backend directory:

```env
# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
LOG_LEVEL=INFO

# Model Configuration (use smaller models if having issues)
ASR_MODEL=openai/whisper-tiny
LLM_MODEL=microsoft/DialoGPT-medium
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# RAG Configuration
RAG_TOP_K=3
RAG_INDEX_PATH=./rag_index
```

## System Requirements

### Minimum Requirements
- Python 3.8+
- Node.js 16+
- 4GB RAM
- 2GB free disk space

### Recommended Requirements
- Python 3.9-3.11
- Node.js 18+
- 8GB RAM
- 5GB free disk space
- NVIDIA GPU (optional, for faster inference)

## Getting Help

1. Check the logs in the terminal for specific error messages
2. Try the compatible requirements file: `backend/requirements-compatible.txt`
3. Use smaller models if you have limited resources
4. Check the GitHub issues for similar problems
5. Contact the development team

## Quick Fixes

### Reset Everything
```bash
# Stop all processes
pkill -f "python.*app.py"
pkill -f "npm.*start"

# Clean up
rm -rf backend/__pycache__
rm -rf backend/models
rm -f backend/chat_sessions.db
rm -rf frontend/node_modules

# Reinstall
python run_kisan_ai.py
```

### Test Individual Components
```bash
# Test backend only
python start_backend.py

# Test frontend only
python start_frontend.py

# Test setup
python test_setup.py
```
