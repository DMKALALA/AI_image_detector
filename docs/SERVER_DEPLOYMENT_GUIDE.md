# 🚀 Server Deployment Guide

Complete guide for hosting the AI Image Detector on a dedicated server so your entire team can access it.

---

## 📋 Table of Contents

1. [Server Requirements](#server-requirements)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Installation Steps](#installation-steps)
4. [Network Configuration](#network-configuration)
5. [Production Configuration](#production-configuration)
6. [Running the Server](#running-the-server)
7. [Team Access Setup](#team-access-setup)
8. [Maintenance & Monitoring](#maintenance--monitoring)
9. [Troubleshooting](#troubleshooting)

---

## 🖥️ Server Requirements

### Minimum Hardware Specs:
- **CPU**: 4+ cores (8+ recommended for ML models)
- **RAM**: 16GB minimum (32GB+ recommended)
- **Storage**: 100GB+ free space (for models, datasets, uploads)
- **GPU**: Optional but highly recommended (NVIDIA CUDA-compatible)
- **Network**: Gigabit Ethernet recommended

### Software Requirements:
- **OS**: Ubuntu 20.04/22.04 LTS (recommended) or Windows Server 2019+
- **Python**: 3.9 (exact version for compatibility)
- **Network**: Static IP or DHCP reservation
- **Firewall**: Port 8000 (or custom) open for team access

---

## ✅ Pre-Deployment Checklist

Before starting, ensure you have:

- [ ] Server with static IP address
- [ ] Admin/sudo access to the server
- [ ] Network access configured (see your network diagram)
- [ ] Firewall rules allowing inbound connections
- [ ] Git installed on the server
- [ ] Python 3.9 installed
- [ ] (Optional) CUDA drivers if using GPU

---

## 📦 Installation Steps

### Step 1: Connect to Your Server

**If using SSH (Linux/Mac host):**
```bash
ssh your-username@192.168.20.10  # Use your server's IP
```

**If using Remote Desktop (Windows Server):**
- Use Windows Remote Desktop Connection
- Enter server IP and credentials

### Step 2: Install System Dependencies

**Ubuntu/Debian:**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.9 and dependencies
sudo apt install -y python3.9 python3.9-venv python3.9-dev \
    git build-essential libssl-dev libffi-dev \
    libjpeg-dev zlib1g-dev

# Install PostgreSQL (recommended for production)
sudo apt install -y postgresql postgresql-contrib

# (Optional) Install NVIDIA drivers for GPU
# Check: https://developer.nvidia.com/cuda-downloads
```

**Windows Server:**
```powershell
# Install Python 3.9 from python.org
# Download: https://www.python.org/downloads/release/python-390/

# Install Git for Windows
# Download: https://git-scm.com/download/win

# Install Visual C++ Build Tools (for some Python packages)
# Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### Step 3: Clone the Repository

```bash
# Navigate to web server directory
cd /var/www/  # Linux
# or
cd C:\inetpub\wwwroot\  # Windows

# Clone the repository
git clone https://github.com/DMKALALA/AI_image_detector.git
cd AI_image_detector

# Checkout main branch (with all latest features)
git checkout main
```

### Step 4: Set Up Python Virtual Environment

```bash
# Create virtual environment
python3.9 -m venv .venv

# Activate it
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

### Step 5: Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# This will install:
# - Django 4.2.7
# - PyTorch 2.2.0
# - Transformers 4.36.2
# - And all other dependencies (see requirements.txt)
```

### Step 6: Configure Database

**Option A: SQLite (Development/Small Teams)**
```bash
# Already configured in settings.py - no action needed
# Database file will be created automatically
```

**Option B: PostgreSQL (Recommended for Production)**
```bash
# Create database and user
sudo -u postgres psql
postgres=# CREATE DATABASE ai_detector_db;
postgres=# CREATE USER ai_detector_user WITH PASSWORD 'your-secure-password';
postgres=# GRANT ALL PRIVILEGES ON DATABASE ai_detector_db TO ai_detector_user;
postgres=# \q

# Update settings.py database configuration (see Production Configuration section)
```

### Step 7: Run Django Migrations

```bash
# Create database tables
python manage.py migrate

# Create superuser for admin access
python manage.py createsuperuser
# Follow prompts to set username, email, password
```

### Step 8: Collect Static Files

```bash
# Gather all static files for production serving
python manage.py collectstatic --noinput
```

---

## 🌐 Network Configuration

### Step 1: Configure Firewall

**Ubuntu (ufw):**
```bash
# Allow SSH (if using)
sudo ufw allow 22/tcp

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow Django dev server (port 8000)
sudo ufw allow 8000/tcp

# Enable firewall
sudo ufw enable
```

**Windows Server:**
```powershell
# Open Windows Firewall with Advanced Security
# Create inbound rule for port 8000
New-NetFirewallRule -DisplayName "Django AI Detector" `
    -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

### Step 2: Configure Router/Switch

Based on your network diagram, you'll need to:

1. **Assign Static IP** to your server (e.g., `192.168.20.10`)
   - Set via your Cisco switch or router DHCP settings
   - Or configure static IP on the server itself

2. **Update DNS/Hosts** (optional, for easier access)
   - Add entry: `ai-detector.local → 192.168.20.10`
   - Team members add to their `/etc/hosts` (Linux/Mac) or `C:\Windows\System32\drivers\etc\hosts` (Windows)

3. **Configure VLANs** (if applicable)
   - Ensure server VLAN can communicate with team member VLANs
   - Your network shows multiple VLANs - check inter-VLAN routing

---

## 🔧 Production Configuration

### Step 1: Create Environment Variables File

Create `.env` file in project root:

```bash
# /var/www/AI_image_detector/.env

# Security
SECRET_KEY=your-very-long-random-secret-key-here-change-this
API_KEY=your-api-key-for-api-endpoints-change-this

# Environment
DEBUG=False
ALLOWED_HOSTS=192.168.20.10,ai-detector.local,localhost

# Model Settings
ENABLE_MODEL_IMPORTS=1

# Database (if using PostgreSQL)
DB_ENGINE=django.db.backends.postgresql
DB_NAME=ai_detector_db
DB_USER=ai_detector_user
DB_PASSWORD=your-secure-password
DB_HOST=localhost
DB_PORT=5432

# Static/Media Files
STATIC_ROOT=/var/www/AI_image_detector/staticfiles
MEDIA_ROOT=/var/www/AI_image_detector/media
```

**Generate secure keys:**
```python
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

### Step 2: Update Django Settings

Edit `image_detector_project/settings.py`:

```python
import os
from pathlib import Path

# Load environment variables
SECRET_KEY = os.environ.get('SECRET_KEY')
DEBUG = os.environ.get('DEBUG', 'False') == 'True'
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', 'localhost').split(',')

# Database (if using PostgreSQL)
if os.environ.get('DB_ENGINE'):
    DATABASES = {
        'default': {
            'ENGINE': os.environ.get('DB_ENGINE'),
            'NAME': os.environ.get('DB_NAME'),
            'USER': os.environ.get('DB_USER'),
            'PASSWORD': os.environ.get('DB_PASSWORD'),
            'HOST': os.environ.get('DB_HOST'),
            'PORT': os.environ.get('DB_PORT'),
        }
    }

# Static files
STATIC_ROOT = os.environ.get('STATIC_ROOT', BASE_DIR / 'staticfiles')
MEDIA_ROOT = os.environ.get('MEDIA_ROOT', BASE_DIR / 'media')
```

### Step 3: Set Permissions (Linux)

```bash
# Create media directory
mkdir -p media uploads

# Set ownership
sudo chown -R www-data:www-data /var/www/AI_image_detector
# or your web server user

# Set permissions
chmod -R 755 /var/www/AI_image_detector
chmod -R 775 media uploads
```

---

## 🏃 Running the Server

### Option 1: Django Development Server (Quick Start)

**Good for**: Testing, small teams (< 5 users)

```bash
# Load environment variables
export $(cat .env | xargs)

# Run server accessible to network
python manage.py runserver 0.0.0.0:8000
```

**Access from team members:**
- URL: `http://192.168.20.10:8000`

### Option 2: Gunicorn (Production - Recommended)

**Good for**: Production, larger teams, better performance

```bash
# Install Gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn image_detector_project.wsgi:application \
    --bind 0.0.0.0:8000 \
    --workers 4 \
    --timeout 300 \
    --access-logfile /var/log/gunicorn-access.log \
    --error-logfile /var/log/gunicorn-error.log \
    --daemon
```

### Option 3: Nginx + Gunicorn (Best for Production)

**Good for**: High traffic, SSL/HTTPS, static file serving

#### Install Nginx:
```bash
sudo apt install nginx
```

#### Create Nginx configuration:
```nginx
# /etc/nginx/sites-available/ai-detector

server {
    listen 80;
    server_name 192.168.20.10 ai-detector.local;

    client_max_body_size 50M;

    location /static/ {
        alias /var/www/AI_image_detector/staticfiles/;
    }

    location /media/ {
        alias /var/www/AI_image_detector/media/;
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

#### Enable and start:
```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/ai-detector /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx

# Start Gunicorn
gunicorn image_detector_project.wsgi:application \
    --bind 127.0.0.1:8000 \
    --workers 4 \
    --daemon
```

### Option 4: Systemd Service (Auto-Start on Boot)

Create `/etc/systemd/system/ai-detector.service`:

```ini
[Unit]
Description=AI Image Detector Django Application
After=network.target

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/var/www/AI_image_detector
Environment="PATH=/var/www/AI_image_detector/.venv/bin"
EnvironmentFile=/var/www/AI_image_detector/.env
ExecStart=/var/www/AI_image_detector/.venv/bin/gunicorn \
    --bind 0.0.0.0:8000 \
    --workers 4 \
    --timeout 300 \
    image_detector_project.wsgi:application

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-detector
sudo systemctl start ai-detector
sudo systemctl status ai-detector
```

---

## 👥 Team Access Setup

### Step 1: Provide Team Access Information

Share with your team:

```
AI Image Detector Server Access
================================

Web Interface: http://192.168.20.10:8000
(or http://ai-detector.local if DNS configured)

API Endpoint: http://192.168.20.10:8000/api/detect/
API Key: [share your API_KEY from .env]

Admin Panel: http://192.168.20.10:8000/admin/
Admin User: [superuser you created]
Admin Pass: [superuser password]

Network: Must be on same network or VPN
```

### Step 2: Create Team Member Accounts (Optional)

```bash
# Create additional users via admin panel
# Visit: http://192.168.20.10:8000/admin/
# Login with superuser credentials
# Navigate to Users → Add User
```

### Step 3: API Access for Developers

Team members can use the API:

```python
import requests

API_URL = "http://192.168.20.10:8000/api/detect/"
API_KEY = "your-api-key"

with open("image.jpg", "rb") as f:
    response = requests.post(
        API_URL,
        headers={"X-API-Key": API_KEY},
        files={"image": f}
    )
    
print(response.json())
```

---

## 🔍 Maintenance & Monitoring

### Daily Checks

```bash
# Check service status
sudo systemctl status ai-detector

# Check logs
tail -f /var/log/gunicorn-error.log
tail -f /var/log/nginx/error.log

# Check disk space
df -h

# Check memory
free -h
```

### Weekly Maintenance

```bash
# Update codebase
cd /var/www/AI_image_detector
git pull origin main

# Update dependencies (if needed)
source .venv/bin/activate
pip install -r requirements.txt --upgrade

# Run migrations (if any)
python manage.py migrate

# Restart service
sudo systemctl restart ai-detector
```

### Backup Strategy

```bash
# Backup database (SQLite)
cp db.sqlite3 backups/db_$(date +%Y%m%d).sqlite3

# Backup database (PostgreSQL)
pg_dump ai_detector_db > backups/db_$(date +%Y%m%d).sql

# Backup uploaded images
tar -czf backups/media_$(date +%Y%m%d).tar.gz media/

# Backup fine-tuned models (if any)
tar -czf backups/models_$(date +%Y%m%d).tar.gz hf_finetuned_models/
```

### Monitoring Tools

Install monitoring:

```bash
# Install htop for system monitoring
sudo apt install htop

# Install netdata for web-based monitoring
bash <(curl -Ss https://my-netdata.io/kickstart.sh)
# Access: http://192.168.20.10:19999
```

---

## 🐛 Troubleshooting

### Issue: Can't Access Server from Team Machines

**Solutions:**
1. Check firewall: `sudo ufw status`
2. Verify server is listening: `netstat -tuln | grep 8000`
3. Test from server itself: `curl http://localhost:8000`
4. Check VLAN routing (see your network diagram)
5. Ping server: `ping 192.168.20.10`

### Issue: Models Taking Too Long to Load

**Solutions:**
1. Use GPU acceleration (install CUDA)
2. Reduce model batch size in code
3. Use model quantization
4. Increase server RAM
5. Load models lazily (only when needed)

### Issue: "Service Unavailable" Error

**Solutions:**
1. Check if models are loaded: `export ENABLE_MODEL_IMPORTS=1`
2. Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`
3. Check logs for specific errors
4. Ensure enough RAM/GPU memory

### Issue: Permission Denied Errors

**Solutions:**
```bash
# Fix ownership
sudo chown -R www-data:www-data /var/www/AI_image_detector

# Fix permissions
chmod -R 755 /var/www/AI_image_detector
chmod -R 775 media uploads
```

### Issue: Slow Performance

**Solutions:**
1. Increase Gunicorn workers: `--workers 8`
2. Enable GPU: Check CUDA installation
3. Add more RAM to server
4. Use Nginx for static file serving
5. Enable database query optimization

---

## 📞 Support

For issues specific to this deployment:

1. **Check Documentation**: Review all docs in `docs/` folder
2. **Check Logs**: `tail -f /var/log/gunicorn-error.log`
3. **Test Locally**: Run on your dev machine first
4. **Network Issues**: Check your Cisco switch/router config

---

## 🎉 Success Checklist

Once deployed, verify:

- [ ] Server accessible at `http://192.168.20.10:8000`
- [ ] All 5 detection methods working
- [ ] Team members can upload images
- [ ] API endpoints working with API key
- [ ] Admin panel accessible
- [ ] Logs showing no errors
- [ ] Auto-start on reboot configured
- [ ] Backups configured

**Congratulations! Your AI Image Detector is now live for your team!** 🚀

---

## 📚 Related Documentation

- [Installation Guide for macOS/Windows](INSTALLATION_GUIDE.md)
- [Project Features Summary](PROJECT_SUMMARY.md)
- [Fine-Tuning Guide](FINE_TUNING_GUIDE.md)
- [Security Documentation](SECURITY.md)
- [API Documentation](../README.md#api-endpoints)

