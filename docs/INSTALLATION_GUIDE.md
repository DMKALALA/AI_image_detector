# 💻 Installation Guide - macOS & Windows

Complete step-by-step guide for setting up the AI Image Detector on macOS or Windows for development.

---

## 📋 Table of Contents

1. [macOS Installation](#-macos-installation)
2. [Windows Installation](#-windows-installation)
3. [Verification Steps](#-verification-steps)
4. [Common Issues](#-common-issues)

---

## 🍎 macOS Installation

### Prerequisites

- **macOS**: 10.15 (Catalina) or later
- **Xcode Command Line Tools**: Required for compiling packages
- **Homebrew**: Package manager for macOS

### Step 1: Install Xcode Command Line Tools

```bash
xcode-select --install
```

Click "Install" in the popup window and wait for completion.

### Step 2: Install Homebrew (if not installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 3: Install Python 3.9

```bash
# Install Python 3.9
brew install python@3.9

# Verify installation
python3.9 --version
# Should show: Python 3.9.x
```

### Step 4: Install Git

```bash
# Install Git
brew install git

# Verify installation
git --version
```

### Step 5: Clone the Repository

```bash
# Navigate to your projects directory
cd ~/Documents

# Clone the repository
git clone https://github.com/DMKALALA/AI_image_detector.git
cd AI_image_detector

# Checkout main branch
git checkout main
```

### Step 6: Create Virtual Environment

```bash
# Create virtual environment
python3.9 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Your prompt should now show (.venv)
```

### Step 7: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# This will take 5-10 minutes
# It installs PyTorch, Transformers, and all other dependencies
```

**Note for M1/M2 Macs:**
If you encounter issues with PyTorch:

```bash
# Uninstall existing PyTorch
pip uninstall torch torchvision

# Install ARM64-optimized version
pip install torch==2.2.0 torchvision==0.17.0
```

### Step 8: Set Up Environment Variables

```bash
# Create .env file
cat > .env << 'EOF'
SECRET_KEY=$(python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())")
API_KEY=dev-api-key-$(date +%s)
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
ENABLE_MODEL_IMPORTS=1
EOF

# Source the file
export $(cat .env | xargs)
```

### Step 9: Initialize Database

```bash
# Run migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser
# Enter username, email, and password when prompted
```

### Step 10: Run the Development Server

```bash
# Start server
python manage.py runserver

# Server will start at: http://127.0.0.1:8000
```

### Step 11: Verify Installation

Open your browser and visit:
- **Home**: http://127.0.0.1:8000
- **Admin**: http://127.0.0.1:8000/admin

---

## 🪟 Windows Installation

### Prerequisites

- **Windows**: 10 or 11 (64-bit)
- **Admin Access**: Required for some installations

### Step 1: Install Python 3.9

1. Download Python 3.9 from: https://www.python.org/downloads/release/python-390/
   - Click "Windows installer (64-bit)"
2. Run the installer
3. **IMPORTANT**: Check "Add Python 3.9 to PATH"
4. Click "Install Now"
5. Wait for installation to complete

**Verify installation:**
```powershell
python --version
# Should show: Python 3.9.x
```

### Step 2: Install Git for Windows

1. Download Git from: https://git-scm.com/download/win
2. Run the installer
3. Use default settings (click "Next" through all screens)
4. Complete installation

**Verify installation:**
```powershell
git --version
```

### Step 3: Install Visual C++ Build Tools

Required for compiling some Python packages.

1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run installer
3. Select "Desktop development with C++"
4. Click "Install"
5. Wait for installation (may take 15-30 minutes)

### Step 4: Clone the Repository

```powershell
# Open PowerShell or Command Prompt

# Navigate to your projects directory
cd C:\Users\YourUsername\Documents

# Clone the repository
git clone https://github.com/DMKALALA/AI_image_detector.git
cd AI_image_detector

# Checkout main branch
git checkout main
```

### Step 5: Create Virtual Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Your prompt should now show (.venv)
```

### Step 6: Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# This will take 5-10 minutes
```

**If you encounter SSL errors:**
```powershell
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### Step 7: Set Up Environment Variables

```powershell
# Create .env file
$SecretKey = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 50 | % {[char]$_})
$ApiKey = "dev-api-key-$(Get-Date -Format 'yyyyMMddHHmmss')"

@"
SECRET_KEY=$SecretKey
API_KEY=$ApiKey
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
ENABLE_MODEL_IMPORTS=1
"@ | Out-File -FilePath .env -Encoding utf8

# Load environment variables for current session
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
}
```

### Step 8: Initialize Database

```powershell
# Run migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser
# Enter username, email, and password when prompted
```

### Step 9: Run the Development Server

```powershell
# Start server
python manage.py runserver

# Server will start at: http://127.0.0.1:8000
```

### Step 10: Verify Installation

Open your browser and visit:
- **Home**: http://127.0.0.1:8000
- **Admin**: http://127.0.0.1:8000/admin

---

## ✅ Verification Steps

After installation on either macOS or Windows, test these features:

### 1. Upload Test Image

1. Go to http://127.0.0.1:8000
2. Click "Choose File" and select an image
3. Click "Analyze Image"
4. Wait for detection (may take 30-60 seconds on first run)
5. Verify you see results from all 5 methods

### 2. Check Method Comparison

On the results page, verify you see:
- Method 1: Deep Learning
- Method 2: Statistical Patterns
- Method 3: Advanced Spectral Analysis
- Method 4: HuggingFace Ensemble (3 models)
- Method 5: Enterprise Models

### 3. Test Feedback System

1. On results page, scroll to "Was this detection accurate?"
2. Click "Correct" or "Incorrect"
3. Verify you see "Thank you for your feedback!"

### 4. Test Re-upload Learning

1. Upload the same image again
2. Check for "🔄 Re-uploaded" indicator
3. Verify confidence is adjusted

### 5. Check Admin Panel

1. Go to http://127.0.0.1:8000/admin
2. Login with superuser credentials
3. Verify you can see:
   - Image uploads
   - User feedback
   - System logs

### 6. Test API Endpoint

**macOS/Linux:**
```bash
curl -X POST http://127.0.0.1:8000/api/detect/ \
  -H "X-API-Key: $API_KEY" \
  -F "image=@test_image.jpg"
```

**Windows (PowerShell):**
```powershell
$ApiKey = (Get-Content .env | Select-String "API_KEY").ToString().Split("=")[1]
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/detect/" `
  -Method POST `
  -Headers @{"X-API-Key"=$ApiKey} `
  -Form @{image=Get-Item "test_image.jpg"}
```

---

## 🐛 Common Issues

### Issue: "Python not found" (Windows)

**Solution:**
1. Reinstall Python and check "Add Python to PATH"
2. Or add manually:
   - Search "Environment Variables" in Windows
   - Edit PATH
   - Add: `C:\Users\YourUsername\AppData\Local\Programs\Python\Python39`

### Issue: "pip: command not found" (macOS)

**Solution:**
```bash
python3.9 -m pip install --upgrade pip
```

### Issue: Virtual environment not activating

**macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```powershell
.venv\Scripts\activate

# If blocked by execution policy:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: PyTorch installation fails

**Solution - Try CPU-only version:**
```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
```

### Issue: "ModuleNotFoundError" after installation

**Solution:**
```bash
# Ensure virtual environment is activated
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### Issue: Port 8000 already in use

**macOS/Linux:**
```bash
# Find and kill process using port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
python manage.py runserver 8001
```

**Windows:**
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process (replace PID with actual process ID)
taskkill /PID <PID> /F

# Or use different port
python manage.py runserver 8001
```

### Issue: Slow first detection

**This is normal!** First detection loads all models into memory:
- Takes 30-60 seconds on first run
- Subsequent detections are much faster (2-5 seconds)
- Use GPU if available for 5-10x speedup

### Issue: "Service Unavailable" error

**Solution:**
```bash
# Ensure models are enabled
export ENABLE_MODEL_IMPORTS=1  # macOS/Linux
$env:ENABLE_MODEL_IMPORTS="1"  # Windows

# Restart server
python manage.py runserver
```

### Issue: Memory errors during model loading

**Solution:**
```bash
# Reduce memory usage by disabling some methods
# Edit detector/three_method_detection_service.py
# Comment out methods you don't need

# Or use swap space (Linux/macOS)
# Or close other applications
```

---

## 🎯 Next Steps

After successful installation:

1. **Read Project Summary**: See `docs/PROJECT_SUMMARY.md` for all features
2. **Fine-Tune Models**: Follow `docs/FINE_TUNING_GUIDE.md`
3. **Deploy to Server**: Follow `docs/SERVER_DEPLOYMENT_GUIDE.md`
4. **Configure Security**: Read `docs/SECURITY.md`
5. **Set Up Team**: Follow `docs/TEAM_SETUP_GUIDE.md`

---

## 🆘 Getting Help

If you encounter issues not covered here:

1. Check server logs: Look in terminal where `runserver` is running
2. Check documentation: Review all docs in `docs/` folder
3. Check GitHub issues: https://github.com/DMKALALA/AI_image_detector/issues
4. Enable debug mode: Set `DEBUG=True` in `.env`

---

## 🎉 Success!

If you can:
- ✅ Access http://127.0.0.1:8000
- ✅ Upload and detect images
- ✅ See all 5 method results
- ✅ Access admin panel

**Your installation is complete!** 🚀

---

## 📚 Related Documentation

- [Server Deployment Guide](SERVER_DEPLOYMENT_GUIDE.md) - For team hosting
- [Project Summary](PROJECT_SUMMARY.md) - All features explained
- [Environment Variables](ENVIRONMENT_VARIABLES.md) - Configuration options
- [Running Guide](RUNNING.md) - Different operation modes

