# Deployment Guide - AI Image Detector

This guide covers multiple deployment options for the AI Image Detector Django application.

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Render Deployment (Recommended)](#render-deployment-recommended)
3. [Railway Deployment](#railway-deployment)
4. [PythonAnywhere Deployment](#pythonanywhere-deployment)
5. [Heroku Deployment](#heroku-deployment)
6. [Environment Variables](#environment-variables)
7. [Post-Deployment Tasks](#post-deployment-tasks)

---

## Pre-Deployment Checklist

### 1. Update Settings for Production

Before deploying, you need to:

- [ ] Set `DEBUG = False` in production
- [ ] Configure `ALLOWED_HOSTS` with your domain
- [ ] Set up a production database (PostgreSQL recommended)
- [ ] Configure static file handling
- [ ] Set up media file storage (AWS S3 or similar)
- [ ] Add environment variables for secrets
- [ ] Install production dependencies

### 2. Files to Create

- `Procfile` (for Heroku/Railway)
- `runtime.txt` (Python version)
- `.env.example` (environment variables template)
- `requirements.txt` (update with production dependencies)

---

## Render Deployment (Recommended)

**Render** offers free tier with PostgreSQL and easy deployment.

### Step 1: Prepare Your Repository

Make sure all changes are committed and pushed to GitHub:
```bash
git add .
git commit -m "Prepare for deployment"
git push origin main
```

### Step 2: Create Render Account

1. Go to [render.com](https://render.com)
2. Sign up with your GitHub account
3. Connect your repository

### Step 3: Create PostgreSQL Database

1. In Render dashboard, click "New +" → "PostgreSQL"
2. Choose free tier
3. Copy the **Internal Database URL** (you'll need this)

### Step 4: Create Web Service

1. Click "New +" → "Web Service"
2. Connect your GitHub repository
3. Configure:
   - **Name**: ai-image-detector (or your choice)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt && python manage.py collectstatic --noinput`
   - **Start Command**: `gunicorn image_detector_project.wsgi:application`

### Step 5: Set Environment Variables

In the Render dashboard, add these environment variables:

```
DATABASE_URL=<your_postgres_internal_url>
SECRET_KEY=<generate_a_new_secret_key>
DEBUG=False
ALLOWED_HOSTS=your-app-name.onrender.com
DJANGO_SETTINGS_MODULE=image_detector_project.settings
```

### Step 6: Update settings.py

You'll need to update your Django settings for Render. See the "Settings Configuration" section below.

### Step 7: Deploy

Click "Create Web Service" and Render will automatically deploy!

---

## Railway Deployment

**Railway** provides easy deployment with automatic SSL.

### Step 1: Install Railway CLI

```bash
npm install -g @railway/cli
# OR
brew install railway
```

### Step 2: Login

```bash
railway login
```

### Step 3: Initialize Project

```bash
railway init
```

### Step 4: Create Database

```bash
railway add postgresql
```

### Step 5: Set Environment Variables

```bash
railway variables set SECRET_KEY=your-secret-key
railway variables set DEBUG=False
railway variables set ALLOWED_HOSTS=*.railway.app
```

### Step 6: Deploy

```bash
railway up
```

Railway will automatically detect your Django app and deploy it!

---

## PythonAnywhere Deployment

**PythonAnywhere** offers free hosting for Python web apps.

### Step 1: Create Account

1. Go to [pythonanywhere.com](https://www.pythonanywhere.com)
2. Sign up for free account

### Step 2: Clone Repository

In PythonAnywhere console:
```bash
cd ~
git clone https://github.com/DMKALALA/AI_image_detector.git
cd AI_image_detector
```

### Step 3: Create Virtual Environment

```bash
mkvirtualenv --python=python3.9 ai_detector
pip install -r requirements.txt
```

### Step 4: Set Up Database

```bash
python manage.py migrate
python manage.py collectstatic --noinput
```

### Step 5: Configure Web App

1. Go to "Web" tab in dashboard
2. Click "Add a new web app"
3. Choose Django, then your project directory
4. Update WSGI file path: `ai_image_detector/image_detector_project/wsgi.py`
5. Add path in WSGI: `/home/yourusername/AI_image_detector`
6. Reload web app

---

## Heroku Deployment

**Heroku** requires paid plan now, but here's how:

### Step 1: Install Heroku CLI

```bash
# macOS
brew tap heroku/brew && brew install heroku

# Or download from heroku.com
```

### Step 2: Login

```bash
heroku login
```

### Step 3: Create App

```bash
heroku create your-app-name
```

### Step 4: Add PostgreSQL

```bash
heroku addons:create heroku-postgresql:hobby-dev
```

### Step 5: Set Config Variables

```bash
heroku config:set DEBUG=False
heroku config:set SECRET_KEY=$(python -c 'from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())')
heroku config:set ALLOWED_HOSTS=your-app-name.herokuapp.com
```

### Step 6: Deploy

```bash
git push heroku main
heroku run python manage.py migrate
heroku run python manage.py collectstatic --noinput
```

---

## Environment Variables

### Required Variables

Create a `.env` file (add to `.gitignore`):

```env
SECRET_KEY=your-super-secret-key-here-generate-a-new-one
DEBUG=False
ALLOWED_HOSTS=your-domain.com,localhost
DATABASE_URL=postgresql://user:password@host:port/dbname
```

### Generate Secret Key

```bash
python -c 'from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())'
```

---

## Settings Configuration

### Update settings.py for Production

You'll need to modify `image_detector_project/settings.py`:

```python
import os
from pathlib import Path

# ... existing imports ...
import dj_database_url  # Add this if using Heroku/Render

# ... existing code ...

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get('SECRET_KEY', 'your-default-secret-key-for-dev-only')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get('DEBUG', 'False') == 'True'

ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

# Database
# For production (Render/Heroku/Railway with PostgreSQL)
if 'DATABASE_URL' in os.environ:
    DATABASES = {
        'default': dj_database_url.parse(os.environ.get('DATABASE_URL'))
    }
else:
    # Development SQLite
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'  # For production
STATICFILES_DIRS = [BASE_DIR / 'static']  # For development

# Media files (uploaded images)
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# For production, use cloud storage (AWS S3 recommended)
# See "Media Storage" section below
```

---

## Media Storage (Images)

For production, uploaded images should be stored in cloud storage, not on the server.

### Option 1: AWS S3 (Recommended)

Install:
```bash
pip install django-storages boto3
```

Add to `settings.py`:
```python
INSTALLED_APPS = [
    # ... existing apps ...
    'storages',
]

# AWS S3 Settings
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_STORAGE_BUCKET_NAME = os.environ.get('AWS_STORAGE_BUCKET_NAME')
AWS_S3_REGION_NAME = 'us-east-1'
AWS_S3_FILE_OVERWRITE = False
AWS_DEFAULT_ACL = None

DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
STATICFILES_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
```

### Option 2: Render Disk (Temporary)

For small apps, Render's disk storage works temporarily:
```python
# Keep current MEDIA_ROOT setting
# But be aware: files are lost on redeploy!
```

---

## Production Requirements

### Update requirements.txt

Add production dependencies:

```txt
# ... existing requirements ...
gunicorn==21.2.0
whitenoise==6.6.0
dj-database-url==2.1.0
psycopg2-binary==2.9.9  # PostgreSQL adapter
python-dotenv==1.0.0  # For .env file support
```

### Create Procfile

Create `Procfile` (no extension) in project root:

```
web: gunicorn image_detector_project.wsgi:application --bind 0.0.0.0:$PORT
```

### Create runtime.txt

Create `runtime.txt` in project root:

```
python-3.9.18
```

---

## Post-Deployment Tasks

### 1. Run Migrations

After deployment, always run:
```bash
python manage.py migrate
```

### 2. Create Superuser

```bash
python manage.py createsuperuser
```

### 3. Collect Static Files

```bash
python manage.py collectstatic --noinput
```

### 4. Test Your Deployment

- Visit your deployed URL
- Upload a test image
- Check that detection works
- Verify static files load correctly

### 5. Monitor Logs

Check deployment logs for errors:
```bash
# Render
# View in dashboard → Logs

# Railway
railway logs

# Heroku
heroku logs --tail
```

---

## Quick Start: Render (Easiest)

### Complete Checklist:

1. ✅ Push code to GitHub
2. ✅ Create Render account and connect GitHub
3. ✅ Create PostgreSQL database on Render
4. ✅ Create Web Service:
   - Build: `pip install -r requirements.txt && python manage.py collectstatic --noinput`
   - Start: `gunicorn image_detector_project.wsgi:application`
5. ✅ Add environment variables (SECRET_KEY, DATABASE_URL, DEBUG, ALLOWED_HOSTS)
6. ✅ Update settings.py (see configuration above)
7. ✅ Add gunicorn to requirements.txt
8. ✅ Create Procfile
9. ✅ Deploy!

---

## Troubleshooting

### Static Files Not Loading

- Run `collectstatic` manually
- Check `STATIC_ROOT` in settings
- Verify static files URL configuration
- For Render: Ensure static files are collected during build

### Database Connection Errors

- Check DATABASE_URL is correct
- Verify PostgreSQL is running
- Check network connectivity in deployment platform

### Media Files Not Saving

- For production: Use cloud storage (S3)
- Check file permissions
- Verify MEDIA_ROOT path

### Model Files Not Loading

- Model files (.pth) should be committed to repo OR
- Upload to cloud storage and load from URL
- Add model files to static files if small enough

---

## Cost Estimates

| Platform | Free Tier | Paid Tier |
|----------|-----------|-----------|
| **Render** | Free (PostgreSQL + Web) | $7/month (starter) |
| **Railway** | $5 credit/month | Pay-as-you-go |
| **PythonAnywhere** | Free (limited) | $5/month (hobbyist) |
| **Heroku** | None | $7/month (Eco) |

---

## Recommended Deployment Stack

**For Production:**

1. **Hosting**: Render or Railway
2. **Database**: PostgreSQL (included with hosting)
3. **Storage**: AWS S3 (for uploaded images)
4. **CDN**: CloudFront (optional, for static files)
5. **Monitoring**: Sentry (error tracking)

**For Quick Testing:**

1. **Hosting**: PythonAnywhere (free tier)
2. **Database**: SQLite (built-in)
3. **Storage**: Local disk (temporary)

---

## Next Steps

1. Choose a deployment platform
2. Update settings.py for production
3. Create necessary files (Procfile, etc.)
4. Set up environment variables
5. Deploy and test!

---

**Need Help?** Check the platform-specific documentation or review the project's `PROJECT_DOCUMENTATION.md` for more details.

