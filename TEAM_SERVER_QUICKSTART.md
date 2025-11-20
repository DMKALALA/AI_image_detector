# 🚀 Quick Start: Hosting AI Detector on Your Server

**For the team member setting up the shared server**

---

## 📋 What You Need

Looking at your server setup (from the images):
- ✅ Server rack with dedicated machine
- ✅ Cisco network switches (Cat6 cabling)
- ✅ Network diagram showing 192.168.x.x subnet
- ✅ Power management (visible in rack)

Perfect! You have everything hardware-wise. Now let's get the software running.

---

## ⚡ Express Setup (Ubuntu Server)

### 1. Connect to Your Server
```bash
ssh your-username@192.168.20.10  # Your server's IP
```

### 2. Quick Install Script
```bash
# Install everything needed
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.9 python3.9-venv git nginx postgresql

# Clone project
cd /var/www
sudo git clone https://github.com/DMKALALA/AI_image_detector.git
cd AI_image_detector

# Setup Python environment
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure Django
export SECRET_KEY=$(python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())")
export API_KEY="team-api-key-$(date +%s)"
export DEBUG=False
export ALLOWED_HOSTS="192.168.20.10,server.local"

# Setup database
python manage.py migrate
python manage.py createsuperuser  # Create admin account

# Start server (quick test)
python manage.py runserver 0.0.0.0:8000
```

**Test it**: Open `http://192.168.20.10:8000` from another machine on your network.

---

## 🔧 Production Setup (Recommended)

### 1. Install Gunicorn
```bash
pip install gunicorn
```

### 2. Create Environment File
```bash
cat > /var/www/AI_image_detector/.env << 'EOF'
SECRET_KEY=your-secret-key-change-this
API_KEY=your-team-api-key
DEBUG=False
ALLOWED_HOSTS=192.168.20.10,ai-detector.local
ENABLE_MODEL_IMPORTS=1
EOF
```

### 3. Create Systemd Service
```bash
sudo nano /etc/systemd/system/ai-detector.service
```

Paste this:
```ini
[Unit]
Description=AI Image Detector
After=network.target

[Service]
User=www-data
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

Save and enable:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-detector
sudo systemctl start ai-detector
sudo systemctl status ai-detector  # Should show "active (running)"
```

### 4. Configure Firewall
```bash
sudo ufw allow 22    # SSH
sudo ufw allow 8000  # Django
sudo ufw enable
```

---

## 🌐 Network Configuration

### On Your Cisco Switch/Router:

1. **Assign Static IP** to server: `192.168.20.10`
   - Or configure DHCP reservation

2. **Check VLAN routing** if needed
   - Your diagram shows multiple VLANs
   - Ensure server VLAN can route to workstation VLANs

3. **Optional: Add DNS entry**
   - `ai-detector.local → 192.168.20.10`

### On Team Member Machines:

Add to hosts file (if not using DNS):

**macOS/Linux**: `/etc/hosts`
```
192.168.20.10 ai-detector.local
```

**Windows**: `C:\Windows\System32\drivers\etc\hosts`
```
192.168.20.10 ai-detector.local
```

---

## 👥 Share with Team

Send this to your team:

```
AI Image Detector is Live! 🎉

Web Interface: http://192.168.20.10:8000
(or http://ai-detector.local)

API Endpoint: http://192.168.20.10:8000/api/detect/
API Key: [paste API_KEY from .env file]

Admin Panel: http://192.168.20.10:8000/admin/
Username: [superuser you created]
Password: [superuser password]

What it does:
✓ Detects AI-generated images with 5 different methods
✓ 85-90% accuracy
✓ Shows detailed analysis & confidence
✓ Learns from your feedback
✓ API for developers

Must be on office network to access.
```

---

## 🔍 Verify Everything Works

### Test Checklist:
- [ ] Server accessible: `curl http://192.168.20.10:8000`
- [ ] Service running: `sudo systemctl status ai-detector`
- [ ] From another computer: Open `http://192.168.20.10:8000`
- [ ] Upload test image
- [ ] See results from all 5 methods
- [ ] Admin panel works
- [ ] API endpoint works (test with curl)

### Test API:
```bash
curl -X POST http://192.168.20.10:8000/api/detect/ \
  -H "X-API-Key: your-api-key" \
  -F "image=@test_image.jpg"
```

---

## 🐛 Quick Troubleshooting

### Can't access from other machines?
```bash
# Check if server is listening
sudo netstat -tuln | grep 8000

# Check firewall
sudo ufw status

# Try from server itself
curl http://localhost:8000
```

### Service won't start?
```bash
# Check logs
sudo journalctl -u ai-detector -n 50

# Check permissions
sudo chown -R www-data:www-data /var/www/AI_image_detector
```

### "Service Unavailable" error?
```bash
# Ensure models are enabled
echo "ENABLE_MODEL_IMPORTS=1" >> /var/www/AI_image_detector/.env
sudo systemctl restart ai-detector
```

---

## 📊 Monitor Server

### Check Status:
```bash
sudo systemctl status ai-detector  # Service status
htop  # System resources
df -h  # Disk space
```

### View Logs:
```bash
sudo journalctl -u ai-detector -f  # Follow logs
```

### Restart If Needed:
```bash
sudo systemctl restart ai-detector
```

---

## 🔄 Update to Latest Code

When we push updates:
```bash
cd /var/www/AI_image_detector
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
sudo systemctl restart ai-detector
```

---

## 📚 Full Documentation

For detailed guides, see:
- **`docs/SERVER_DEPLOYMENT_GUIDE.md`** - Complete deployment guide
- **`docs/PROJECT_SUMMARY.md`** - Everything we built
- **`docs/SECURITY.md`** - Security features
- **`README.md`** - General documentation

---

## 🎯 Your Server Specs (From Images)

Looking at your rack setup:
- **Server**: Looks like a tower/workstation (visible in images)
- **Networking**: Cisco switches with Cat6 cabling
- **Power**: Managed power in rack
- **Access Point**: Cisco AP visible (for wireless access)

**Recommendation**: 
- Your hardware is more than capable!
- Ensure 16GB+ RAM for best performance
- Consider GPU if server has one (5-10x speed boost)

---

## 🎉 That's It!

You now have a production AI Image Detector running for your whole team!

**Questions?** Check the full guides in the `docs/` folder or ask the team.

---

**Your server (from photos) is ready to be an AI detection powerhouse! 🚀**

