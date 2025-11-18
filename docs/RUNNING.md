# Running the Application

## Quick Start

The application can run in two modes depending on your environment and needs.

## Mode 1: Normal Run (Models Enabled)

Use this for normal development and production when you have shared memory (SHM) access and want full AI detection functionality.

```bash
export SECRET_KEY=your-secret-key
export API_KEY=your-api-key
export DEBUG=True  # Set to False for production
export ENABLE_MODEL_IMPORTS=1  # Enable PyTorch models (default)

python manage.py runserver
```

**What this does:**
- Loads all PyTorch models (5 detection methods)
- Full AI detection functionality available
- API endpoints work normally
- Requires shared memory (SHM) access

**Production Example:**
```bash
export SECRET_KEY=your-strong-secret-key
export API_KEY=your-secure-api-key
export DEBUG=False
export ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
export ENABLE_MODEL_IMPORTS=1  # or leave unset (defaults to 1)

python manage.py runserver
```

## Mode 2: Sandbox/SHM-Limited Run (No Models)

Use this when running in constrained environments (sandboxes, CI/CD) that don't allow shared memory access, or when you only want to test the UI/API surface without loading models.

```bash
export ENABLE_MODEL_IMPORTS=0  # Disable PyTorch imports
export SECRET_KEY=your-secret-key
export API_KEY=your-api-key
export DEBUG=True

python manage.py runserver
```

**What this does:**
- Skips all PyTorch imports (no SHM required)
- UI and API endpoints are accessible
- API detection endpoints return 503 (Service Unavailable)
- Web forms show user-friendly error messages
- Perfect for testing UI, forms, authentication, etc.

**Note:** With `ENABLE_MODEL_IMPORTS=0`, detection functionality is disabled. Use this mode only to exercise the UI/API surface without loading models.

## Environment Variables Summary

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SECRET_KEY` | Yes | - | Django secret key |
| `API_KEY` | Yes | - | API authentication key |
| `DEBUG` | No | `False` | Enable debug mode (`True`/`False`) |
| `ENABLE_MODEL_IMPORTS` | No | `1` | Enable PyTorch imports (`0`/`1`) |
| `ALLOWED_HOSTS` | No | `localhost,127.0.0.1` | Comma-separated hostnames |

## Verification

### Check if models are loaded:
```bash
# With ENABLE_MODEL_IMPORTS=1
curl http://localhost:8000/api/status/

# Should return:
# {
#   "status": "operational",
#   "trained_model_available": true/false,
#   "device": "cpu" or "cuda",
#   "timestamp": "..."
# }
```

### Check if models are disabled:
```bash
# With ENABLE_MODEL_IMPORTS=0
curl http://localhost:8000/api/status/

# Should return:
# {
#   "status": "unavailable",
#   "trained_model_available": false,
#   "device": "N/A",
#   "message": "Detection service is not available...",
#   "timestamp": "..."
# }
```

## Troubleshooting

### "OMP: Error #179: Function Can't open SHM2 failed"
- **Solution**: Set `ENABLE_MODEL_IMPORTS=0` to run without models
- **Alternative**: Run outside the sandbox/constrained environment

### "SECRET_KEY environment variable is required"
- **Solution**: Set `export SECRET_KEY=your-secret-key`

### "API key required and not configured on server"
- **Solution**: Set `export API_KEY=your-api-key`

### API endpoints return 503
- **With ENABLE_MODEL_IMPORTS=0**: Expected behavior - models are disabled
- **With ENABLE_MODEL_IMPORTS=1**: Check logs for model loading errors

## Development Workflow

1. **Start with models disabled** (faster startup, no SHM issues):
   ```bash
   export ENABLE_MODEL_IMPORTS=0
   export SECRET_KEY=dev-key
   export API_KEY=dev-key
   export DEBUG=True
   python manage.py runserver
   ```

2. **Switch to models enabled** when you need detection:
   ```bash
   export ENABLE_MODEL_IMPORTS=1
   # Restart server
   ```

3. **For production**, always use models enabled:
   ```bash
   export ENABLE_MODEL_IMPORTS=1  # or leave unset
   export DEBUG=False
   export SECRET_KEY=strong-production-key
   export API_KEY=secure-production-key
   export ALLOWED_HOSTS=yourdomain.com
   ```

