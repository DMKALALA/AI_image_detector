# AI Image Detector — Server Operations Guide

How the app was set up and how to operate it on this server.

---

## Server Overview

| Item | Value |
|------|-------|
| **Server OS** | Ubuntu 24.04.4 LTS |
| **Server IP** | `10.27.10.104` |
| **App URL** | http://10.27.10.104 |
| **Server User** | `aisadmin` |
| **App Directory** | `/var/www/AI_image_detector` |
| **Python Version** | 3.9.25 (virtual environment) |
| **Framework** | Django 4.2 |
| **Database** | PostgreSQL 16 |
| **Web Server** | Nginx (port 80) → Gunicorn (port 8000) |

---

## How It Was Set Up

### Step 1 — Project files placed on the server

The Django project was placed at `/var/www/AI_image_detector/` owned by the `aisadmin` user.

A Python virtual environment was created:

```bash
cd /var/www/AI_image_detector
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Environment configured for production

The `.env` file at `/var/www/AI_image_detector/.env` was set up with:

```
DEBUG=False
ALLOWED_HOSTS=10.27.10.104,localhost,127.0.0.1
SECRET_KEY=<production-secret-key>
FORCE_CPU=true
MEMORY_CONSTRAINED=true
```

- `DEBUG=False` prevents Django from showing error details to users
- `FORCE_CPU=true` because there is no GPU on this VM
- `MEMORY_CONSTRAINED=true` because the VM has 4GB RAM

### Step 3 — Gunicorn installed as the application server

Instead of Django's built-in `runserver` (which is for development only), Gunicorn was installed:

```bash
source .venv/bin/activate
pip install gunicorn
```

Gunicorn runs the Django app as a proper WSGI server with 2 worker processes.

### Step 4 — systemd service created for auto-start

A systemd service was created so the app starts automatically when the VM boots and restarts if it crashes.

**Service file:** `/etc/systemd/system/ai-image-detector.service`

```ini
[Unit]
Description=AI Image Detector Django App
After=network.target

[Service]
User=aisadmin
Group=aisadmin
WorkingDirectory=/var/www/AI_image_detector
EnvironmentFile=/var/www/AI_image_detector/.env
ExecStart=/var/www/AI_image_detector/.venv/bin/gunicorn --bind 127.0.0.1:8000 --workers 2 --timeout 300 image_detector_project.wsgi:application
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enabled with:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ai-image-detector
```

### Step 5 — Nginx installed as the reverse proxy

Nginx sits in front of Gunicorn to handle incoming HTTP traffic on port 80, serve static files directly, and proxy everything else to the Django app.

```bash
sudo apt install -y nginx
```

**Config file:** `/etc/nginx/sites-available/ai-image-detector`

```nginx
server {
    listen 80;
    server_name 10.27.10.104;

    client_max_body_size 20M;

    location /static/ {
        alias /var/www/AI_image_detector/staticfiles/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    location /uploads/ {
        alias /var/www/AI_image_detector/uploads/;
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
    }
}
```

Enabled with:

```bash
sudo ln -s /etc/nginx/sites-available/ai-image-detector /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx
```

### Step 6 — Static files collected

Django's static files were collected into a single directory for Nginx to serve:

```bash
source .venv/bin/activate
python manage.py collectstatic --noinput
```

### Step 7 — File permissions set

```bash
sudo chown -R aisadmin:aisadmin /var/www/AI_image_detector
chmod -R u+rwX /var/www/AI_image_detector
```

---

## How to Access the App

1. Connect to the same network as the server (VPN or `10.27.10.x` subnet)
2. Open a browser and go to **http://10.27.10.104**
3. Use `http://` (not `https://` — no SSL is configured)

If the page doesn't load, check that you can ping `10.27.10.104` from your machine first.

---

## How the Request Flow Works

```
Your Browser
    │
    ▼ (port 80)
  Nginx
    ├── /static/*   → serves files from /var/www/AI_image_detector/staticfiles/
    ├── /uploads/*   → serves files from /var/www/AI_image_detector/uploads/
    └── everything else → proxies to Gunicorn on 127.0.0.1:8000
                              │
                              ▼
                          Django App → PostgreSQL database
```

---

## Three Services to Know

All three auto-start on boot. You should never need to start them manually after a reboot.

### 1. `ai-image-detector` — The Django app (Gunicorn)

```bash
sudo systemctl status ai-image-detector    # Check if running
sudo systemctl restart ai-image-detector   # Restart after code changes
sudo systemctl stop ai-image-detector      # Stop the app
sudo journalctl -u ai-image-detector -f    # View live logs
```

### 2. `nginx` — The web server

```bash
sudo systemctl status nginx
sudo systemctl restart nginx
sudo nginx -t                              # Test config before restarting
```

### 3. `postgresql@16-main` — The database

```bash
sudo systemctl status postgresql@16-main
```

---

## Common Tasks

### Restart the app (after code changes)

```bash
sudo systemctl restart ai-image-detector
```

### Update the code from git

```bash
cd /var/www/AI_image_detector
git pull
source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py collectstatic --noinput
sudo systemctl restart ai-image-detector
```

### View app logs

```bash
# Live log stream
sudo journalctl -u ai-image-detector -f

# Last 50 lines
sudo journalctl -u ai-image-detector -n 50

# Nginx access log
sudo tail -f /var/log/nginx/access.log

# Nginx error log
sudo tail -f /var/log/nginx/error.log
```

### Install a Python package

```bash
cd /var/www/AI_image_detector
source .venv/bin/activate
pip install <package-name>
sudo systemctl restart ai-image-detector
```

### Run a Django management command

```bash
cd /var/www/AI_image_detector
source .venv/bin/activate
python manage.py <command>
```

---

## AI Models on the Server

| Model | Path |
|-------|------|
| EfficientNet B0 (fine-tuned) | `trained_models/efficientnet_b0_finetuned.pth` |
| AI Classifier | `hf_finetuned_models/ai_classifier_finetuned/` |
| AI/Human Detector | `hf_finetuned_models/ai_human_detector_finetuned/` |
| ViT AI Detector | `hf_finetuned_models/vit_ai_detector_finetuned/` |

Models load into memory on the first request after a restart. That first request may take 10–30 seconds.

---

## Troubleshooting

### "Site can't be reached" in browser

1. **Are you on the right network?** You must be on the `10.27.10.x` subnet or VPN.
   ```
   ping 10.27.10.104
   ```
   If ping fails → network issue, not an app issue.

2. **Is the VM running?** The VM must be powered on. Check with your hypervisor (Proxmox, Hyper-V, etc.).

3. **Are services running?** SSH into the server and check:
   ```bash
   sudo systemctl status ai-image-detector
   sudo systemctl status nginx
   ```

4. **Is port 80 listening?**
   ```bash
   ss -tlnp | grep :80
   ```

### Static files / CSS not loading

```bash
cd /var/www/AI_image_detector
source .venv/bin/activate
python manage.py collectstatic --noinput
sudo systemctl restart nginx
```

### Uploads not working / images not displaying

```bash
ls -la /var/www/AI_image_detector/uploads/
sudo chown -R aisadmin:aisadmin /var/www/AI_image_detector/uploads
```

### App crashes or 502 Bad Gateway

Check the app logs:
```bash
sudo journalctl -u ai-image-detector --since "10 minutes ago"
```

Then restart:
```bash
sudo systemctl restart ai-image-detector
```

### After a VM reboot

Everything should come back automatically. Wait 1–2 minutes, then try http://10.27.10.104. If it still doesn't respond, SSH in and check service statuses.

---

## Key Configuration Files

| File | What it does |
|------|-------------|
| `/var/www/AI_image_detector/.env` | App environment variables |
| `/var/www/AI_image_detector/image_detector_project/settings.py` | Django settings |
| `/etc/systemd/system/ai-image-detector.service` | Gunicorn service definition |
| `/etc/nginx/sites-available/ai-image-detector` | Nginx site configuration |

---

## Important Notes

- The **VM must be running** for the app to be accessible
- There is **no HTTPS** — all traffic is unencrypted (`http://` only)
- The app runs on **CPU only** (no GPU) with memory optimizations for 4GB RAM
- Maximum upload size is **20MB** (set in Nginx config)
- The first request after a restart is slow (10–30s) while AI models load into memory
