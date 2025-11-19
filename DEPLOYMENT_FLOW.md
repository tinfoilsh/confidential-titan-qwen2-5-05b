# Complete Deployment Flow: TiTan-Qwen2.5-0.5B-GGUF

## Overview
This document explains the complete flow from code to production deployment for the TiTan-Qwen2.5-0.5B-GGUF title generation model.

## Repository Structure

```
confidential-qwen3-vl-8b-main/
├── Dockerfile                    # Container build instructions
├── tinfoil-config.yml            # Tinfoil deployment configuration
├── config-containers.yml         # Local container launch configuration
├── launch_containers.py          # Local deployment script
├── .github/workflows/build.yml   # CI/CD pipeline
└── scripts/
    └── gguf_api_server.py        # FastAPI application
```

## Flow Stages

### Stage 1: Development & Code
**File**: `scripts/gguf_api_server.py`
- FastAPI server implementation
- Loads GGUF model using llama-cpp-python
- Provides OpenAI-compatible `/v1/chat/completions` endpoint
- Reads environment variables: MODEL_NAME, PORT, CONTEXT_SIZE, CUDA_VISIBLE_DEVICES

### Stage 2: Container Build
**File**: `Dockerfile`
- Base image: `python:3.11-slim`
- Installs system dependencies (git, curl, build-essential, cmake)
- Installs Python packages (llama-cpp-python, fastapi, uvicorn, etc.)
- Copies `scripts/gguf_api_server.py` → `/app/api_server.py`
- Sets CMD to run the API server
- **Result**: Container image with the application

### Stage 3: CI/CD Pipeline
**File**: `.github/workflows/build.yml`
- **Trigger**: Push of tags matching `v*` (e.g., `v1.0.0`)
- **Steps**:
  1. Checkout code
  2. Set up Docker Buildx
  3. Login to GitHub Container Registry (ghcr.io)
  4. Build Docker image from `Dockerfile`
  5. Push to `ghcr.io/tinfoilsh/titan-qwen-0.5b-gguf:latest`
  6. Also tag with version (e.g., `ghcr.io/tinfoilsh/titan-qwen-0.5b-gguf:v1.0.0`)
  7. Run Tinfoil build action for deployment

### Stage 4: Configuration
**File**: `tinfoil-config.yml`
- Defines model metadata (name, repo)
- Container configuration:
  - Image: `ghcr.io/tinfoilsh/titan-qwen-0.5b-gguf:latest`
  - Environment variables via args (MODEL_NAME, PORT, CONTEXT_SIZE, CUDA_VISIBLE_DEVICES)
- Resource requirements: 4 CPUs, 4GB RAM, no GPUs
- Shim configuration: port 443 → 8000, paths for API endpoints

**File**: `config-containers.yml`
- Local deployment configuration
- Same structure but used by `launch_containers.py` for local testing
- Contains multiple container definitions (llama3, mistral, gpt-oss, titan-qwen)

### Stage 5: Deployment

#### Option A: Tinfoil Deployment (Production)
- Tinfoil reads `tinfoil-config.yml`
- Pulls image from `ghcr.io/tinfoilsh/titan-qwen-0.5b-gguf:latest`
- Creates container with specified resources
- Sets environment variables
- Routes traffic via shim

#### Option B: Local Deployment (Testing)
**File**: `launch_containers.py`
- Reads `tinfoil-config.yml` (or `config-containers.yml`)
- Constructs Docker command:
  ```bash
  docker run -d --name titan-qwen-0.5b-gguf \
    --network host \
    -v /mnt/ramdisk:/tinfoil \
    -e MODEL_NAME=theprint/TiTan-Qwen2.5-0.5B-GGUF \
    -e PORT=8000 \
    -e CONTEXT_SIZE=8192 \
    -e CUDA_VISIBLE_DEVICES= \
    ghcr.io/tinfoilsh/titan-qwen-0.5b-gguf:latest
  ```
- Executes command to start container

### Stage 6: Runtime
**Container starts**:
1. Runs `/app/api_server.py` (from Dockerfile CMD)
2. `gguf_api_server.py` reads environment variables
3. Downloads model from Hugging Face (if not cached)
4. Loads GGUF model using llama-cpp-python
5. Starts FastAPI server on port 8000
6. Serves endpoints:
   - `/v1/chat/completions` - Title generation
   - `/health` - Health check
   - `/v1/models` - Model info
   - `/metrics` - Metrics

## Complete Flow Diagram

```
Developer
    │
    ├─ Writes/Updates: scripts/gguf_api_server.py
    │
    ├─ Commits & Pushes to Git
    │
    ├─ Tags version: git tag v1.0.0 && git push --tags
    │
    └─ GitHub Actions Triggered (.github/workflows/build.yml)
            │
            ├─ Builds Docker Image (Dockerfile)
            │   └─ Copies gguf_api_server.py → /app/api_server.py
            │
            ├─ Pushes to Registry
            │   └─ ghcr.io/tinfoilsh/titan-qwen-0.5b-gguf:latest
            │
            └─ Tinfoil Deployment (reads tinfoil-config.yml)
                    │
                    ├─ Pulls Image from Registry
                    ├─ Creates Container with Resources
                    ├─ Sets Environment Variables
                    └─ Routes Traffic (443 → 8000)
                            │
                            └─ Container Running
                                    │
                                    └─ API Server Responding
                                            │
                                            └─ /v1/chat/completions
```

## Key Files & Their Roles

| File | Purpose | Used By |
|------|---------|---------|
| `Dockerfile` | Container build instructions | Docker, GitHub Actions |
| `scripts/gguf_api_server.py` | Application code | Packaged into container |
| `.github/workflows/build.yml` | CI/CD automation | GitHub Actions |
| `tinfoil-config.yml` | Production deployment config | Tinfoil, launch_containers.py |
| `config-containers.yml` | Local deployment config | launch_containers.py |
| `launch_containers.py` | Local deployment script | Developers for testing |

## Environment Variables Flow

1. **Defined in**: `tinfoil-config.yml` (args section)
2. **Passed via**: Docker `-e` flags in launch command
3. **Read by**: `gguf_api_server.py` using `os.getenv()`
4. **Used for**:
   - `MODEL_NAME`: Which model to load from Hugging Face
   - `PORT`: Which port to serve on (8000)
   - `CONTEXT_SIZE`: Model context window (8192)
   - `CUDA_VISIBLE_DEVICES`: CPU-only (empty string)

## Deployment Commands

### Production (Tinfoil)
```bash
# 1. Tag and push
git tag v1.0.0
git push --tags

# 2. GitHub Actions automatically:
#    - Builds image
#    - Pushes to ghcr.io
#    - Deploys via Tinfoil
```

### Local Testing
```bash
# 1. Build image locally
docker build -f Dockerfile -t ghcr.io/tinfoilsh/titan-qwen-0.5b-gguf:latest .

# 2. Deploy using launch script
python3 launch_containers.py titan-qwen-0.5b-gguf tinfoil-config.yml

# 3. Test
curl http://localhost:8000/health
```

## Summary

**Build**: Dockerfile → Container Image → Registry  
**Configure**: tinfoil-config.yml → Deployment Spec  
**Deploy**: Tinfoil/launch_containers.py → Running Container  
**Run**: Container → API Server → Model Inference
