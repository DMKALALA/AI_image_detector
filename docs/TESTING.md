# Testing Guide

## Running Tests in Constrained Environments

The AI Image Detector uses PyTorch which requires shared memory (SHM) access. Some test environments (like sandboxes) don't allow SHM access, causing tests to fail before they even run.

### Solution: Disable Model Imports During Tests

Set the `ENABLE_MODEL_IMPORTS` environment variable to `0` to skip PyTorch imports during tests:

```bash
export ENABLE_MODEL_IMPORTS=0
python manage.py test detector
```

### Running Tests

**Option 1: With Model Imports Disabled (Recommended for CI/Sandboxes)**
```bash
# Disable PyTorch imports to avoid SHM issues
export ENABLE_MODEL_IMPORTS=0
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
python manage.py test detector
```

**Option 2: With Model Imports Enabled (Requires SHM Access)**
```bash
# Enable model imports (default)
export ENABLE_MODEL_IMPORTS=1
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
python manage.py test detector
```

**Option 3: Run Tests Outside Sandbox**
If you're in a restricted environment, run tests directly on your machine:
```bash
python manage.py test detector
```

### Environment Variables

- `ENABLE_MODEL_IMPORTS`: Set to `0` to disable PyTorch imports (default: `1`)
- `OMP_NUM_THREADS`: Set to `1` to reduce OpenMP overhead
- `KMP_DUPLICATE_LIB_OK`: Set to `TRUE` to allow duplicate OpenMP libraries

### What Happens When ENABLE_MODEL_IMPORTS=0?

When `ENABLE_MODEL_IMPORTS=0`:
- PyTorch and torchvision imports are skipped
- `get_detection_service()` returns `None`
- API endpoints return 503 (Service Unavailable) with a clear error message
- Web forms show a user-friendly error message
- Tests can run without requiring shared memory access

### Test Coverage

Current tests cover:
- API authentication (API key validation)
- CSRF protection
- File upload validation
- Error handling

### Adding New Tests

When writing tests that don't require model functionality:
1. Set `ENABLE_MODEL_IMPORTS=0` in your test setup
2. Mock `get_detection_service()` if needed
3. Test the business logic without loading models

Example:
```python
from unittest.mock import patch, MagicMock
from django.test import TestCase

class MyTest(TestCase):
    @patch('detector.views.get_detection_service')
    def test_something(self, mock_get_service):
        mock_service = MagicMock()
        mock_service.detect_ai_image.return_value = {
            'is_ai_generated': True,
            'confidence': 0.95
        }
        mock_get_service.return_value = mock_service
        # Your test code here
```

### Troubleshooting

**Error: "OMP: Error #179: Function Can't open SHM2 failed"**
- Solution: Set `ENABLE_MODEL_IMPORTS=0` before running tests

**Error: "PyTorch is not available"**
- This is expected when `ENABLE_MODEL_IMPORTS=0`
- Tests should handle this gracefully
- Check that your test mocks `get_detection_service()` if needed

**Tests hang or timeout**
- Ensure `OMP_NUM_THREADS=1` is set
- Consider running tests outside the sandbox environment

