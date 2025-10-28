"""
WSGI config for image_detector_project project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "image_detector_project.settings")

# Note: Migrations run during build process (see render.yaml buildCommand)
# No blocking code here to ensure fast worker startup

application = get_wsgi_application()
