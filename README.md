# Title Generation Model - Tinfoil Deployment

This repository follows the same structure as other model deployments in the organization.

## Structure
- `tinfoil-config.yml` - Tinfoil deployment configuration
- `Dockerfile.gguf` - Container build file
- `scripts/gguf_api_server.py` - FastAPI server implementation
- `.github/workflows/build.yml` - CI/CD workflow

## Model
- **Name**: titan-qwen-0.5b-gguf
- **Source**: theprint/TiTan-Qwen2.5-0.5B-GGUF
- **Format**: GGUF (CPU-only)
- **Resources**: 4 CPUs, 4GB RAM

## Deployment
Deployment is handled automatically via GitHub Actions when tags are pushed.

## API Endpoints
- `/v1/chat/completions` - Generate titles
- `/health` - Health check
- `/v1/models` - List models
- `/metrics` - Metrics
