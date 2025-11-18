# Environment Variables Reference

## Required Variables

### `SECRET_KEY`
- **Required**: Yes
- **Description**: Django secret key for cryptographic signing
- **Example**: `export SECRET_KEY=your-strong-secret-key-here`
- **Production**: Must be set to a strong, randomly generated key
- **Development**: Can be any string, but use a strong key for security

## Optional Variables

### `DEBUG`
- **Required**: No (default: `False`)
- **Description**: Enable Django debug mode
- **Values**: `True` or `False`
- **Development**: `export DEBUG=True`
- **Production**: `export DEBUG=False` (required)

### `ALLOWED_HOSTS`
- **Required**: No (default: `localhost,127.0.0.1`)
- **Description**: Comma-separated list of allowed hostnames
- **Example**: `export ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com`
- **Production**: Must include your domain name(s)

### `API_KEY`
- **Required**: No (optional for development)
- **Description**: API key for authenticating API requests
- **Example**: `export API_KEY=your-api-key-here`
- **Production**: Recommended for API security
- **Development**: Not required (allows unauthenticated access with warning)

### `ENABLE_MODEL_IMPORTS`
- **Required**: No (default: `1`)
- **Description**: Enable/disable PyTorch model imports
- **Values**: `0` (disabled) or `1` (enabled)
- **Normal Operation**: Leave unset or set to `1` to enable PyTorch models
- **Testing in Sandboxes**: Set to `0` to skip PyTorch imports (avoids SHM errors)
- **Example**: `export ENABLE_MODEL_IMPORTS=0` (for tests only)

### `FORCE_CPU`
- **Required**: No (default: `true`)
- **Description**: Force CPU mode instead of GPU
- **Values**: `true` or `false`
- **Example**: `export FORCE_CPU=true`

### `MEMORY_CONSTRAINED`
- **Required**: No (default: `false`)
- **Description**: Enable memory-constrained mode
- **Values**: `true` or `false`
- **Example**: `export MEMORY_CONSTRAINED=true`

### `DATABASE_URL`
- **Required**: No (default: SQLite)
- **Description**: PostgreSQL database connection URL
- **Example**: `export DATABASE_URL=postgresql://user:password@localhost/dbname`
- **Production**: Recommended for production deployments

## OpenMP Variables (for PyTorch)

### `OMP_NUM_THREADS`
- **Required**: No
- **Description**: Number of OpenMP threads
- **Recommended**: `export OMP_NUM_THREADS=1` (reduces overhead)
- **Example**: `export OMP_NUM_THREADS=1`

### `KMP_DUPLICATE_LIB_OK`
- **Required**: No
- **Description**: Allow duplicate OpenMP libraries
- **Recommended**: `export KMP_DUPLICATE_LIB_OK=TRUE`
- **Example**: `export KMP_DUPLICATE_LIB_OK=TRUE`

## Example Configurations

### Development
```bash
export SECRET_KEY=dev-secret-key-change-in-production
export DEBUG=True
export ALLOWED_HOSTS=localhost,127.0.0.1
# ENABLE_MODEL_IMPORTS defaults to 1 (enabled)
```

### Production
```bash
export SECRET_KEY=your-strong-randomly-generated-secret-key
export DEBUG=False
export ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
export API_KEY=your-secure-api-key
export DATABASE_URL=postgresql://user:password@host/dbname
# ENABLE_MODEL_IMPORTS defaults to 1 (enabled)
```

### Testing (in Sandbox)
```bash
export SECRET_KEY=test-secret-key
export DEBUG=True
export API_KEY=test-api-key
export ENABLE_MODEL_IMPORTS=0  # Disable PyTorch to avoid SHM errors
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
```

### Testing (Full Functionality)
```bash
export SECRET_KEY=test-secret-key
export DEBUG=True
export API_KEY=test-api-key
export ENABLE_MODEL_IMPORTS=1  # Enable PyTorch (requires SHM access)
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
```

## Using .env File

You can create a `.env` file in the project root (make sure it's in `.gitignore`):

```bash
# .env
SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
API_KEY=your-api-key-here
```

The project uses `python-dotenv` to automatically load `.env` files if available.

