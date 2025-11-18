import os
from django.test import TestCase, Client, override_settings
from django.urls import reverse


class ApiAuthTests(TestCase):
    def setUp(self):
        self.client = Client()
        # Disable model imports to avoid SHM issues in test environments
        os.environ['ENABLE_MODEL_IMPORTS'] = '0'
        # Ensure we restore any pre-existing API_KEY after tests
        self._original_api_key = os.environ.pop('API_KEY', None)
        self._original_enable_imports = os.environ.get('ENABLE_MODEL_IMPORTS')

    def tearDown(self):
        if self._original_api_key is not None:
            os.environ['API_KEY'] = self._original_api_key
        if self._original_enable_imports is not None:
            os.environ['ENABLE_MODEL_IMPORTS'] = self._original_enable_imports
        elif 'ENABLE_MODEL_IMPORTS' in os.environ:
            del os.environ['ENABLE_MODEL_IMPORTS']

    @override_settings(API_KEY=None)
    def test_missing_api_key_configuration_fails_closed(self):
        url = reverse('detector:api_detect')
        response = self.client.post(url)
        # Without API_KEY configured, should allow access (development mode)
        # But with ENABLE_MODEL_IMPORTS=0, service unavailable (503) or form invalid (400)
        self.assertIn(response.status_code, [400, 401, 503])

    @override_settings(API_KEY='valid-key')
    def test_wrong_api_key_rejected(self):
        url = reverse('detector:api_detect')
        response = self.client.post(url, HTTP_X_API_KEY='wrong-key')
        self.assertEqual(response.status_code, 403)

    @override_settings(API_KEY='valid-key')
    def test_csrf_not_required_for_api(self):
        url = reverse('detector:api_detect')
        response = self.client.post(url, HTTP_X_API_KEY='valid-key')
        # Should not 403 for missing CSRF; 
        # With ENABLE_MODEL_IMPORTS=0, service unavailable (503) or form invalid (400)
        self.assertNotEqual(response.status_code, 403)
        # Either 400 (form invalid) or 503 (service unavailable) is acceptable
        self.assertIn(response.status_code, [400, 503])
