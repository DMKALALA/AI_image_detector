"""
WSGI config for image_detector_project project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "image_detector_project.settings")

# Auto-run migrations on startup (for production deployments without shell access)
# This ensures the database is always up-to-date
try:
    import django
    django.setup()
    from django.core.management import call_command
    from django.db import connection
    
    # Check if we can connect to the database
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
    
    # Run migrations if we have a database connection
    # This will only create tables that don't exist
    call_command('migrate', '--run-syncdb', verbosity=0, interactive=False)
except Exception as e:
    # If migrations fail, log but don't crash the app
    # The app will still start and handle errors gracefully
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Auto-migration failed (this is OK on first run): {e}")

application = get_wsgi_application()
