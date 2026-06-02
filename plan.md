# Local RTX 4090 Installation Plan

## Goal

Install and configure this OpenSCAD MCP project to run on this machine with local CUDA reconstruction on the NVIDIA RTX 4090.

## Current Environment Findings

- GPU is available: NVIDIA GeForce RTX 4090.
- NVIDIA driver is available: `580.126.20`.
- CUDA compiler is available: `nvcc 12.4`.
- Default Python is `3.11.10`.
- `python3.12` is not installed.
- `uv` is not installed.
- `cmake` is not installed.
- `openscad` is not installed.
- The session is running as `root`; `sudo` is not installed.
- `py312/` does not exist.
- `.env` does not exist.

## Install Steps

1. Install required system packages as root:

   ```bash
   apt-get update
   apt-get install -y git build-essential cmake python3.12 python3.12-venv python3-pip openscad
   ```

2. Install `uv`:

   ```bash
   python3 -m pip install --user uv
   ```

3. Clone CUDA Multi-View Stereo alongside this repo if it is not already present:

   ```bash
   git clone https://github.com/fixstars/cuda-multi-view-stereo.git /workspace/cuda-multi-view-stereo
   ```

4. Configure and build CUDA Multi-View Stereo:

   ```bash
   cmake -S /workspace/cuda-multi-view-stereo -B /workspace/cuda-multi-view-stereo/build
   cmake --build /workspace/cuda-multi-view-stereo/build -j"$(nproc)"
   ```

5. Create the Python 3.12 virtual environment:

   ```bash
   python3.12 -m venv py312
   ```

6. Install Python dependencies:

   ```bash
   uv pip install --python py312/bin/python -r requirements.txt
   ```

7. Create local `.env` configuration:

   ```env
   GEMINI_MODEL=gemini-2.5-flash-image
   CUDA_MVS_PATH=/workspace/cuda-multi-view-stereo
   CUDA_MVS_USE_GPU=True
   REMOTE_CUDA_MVS_ENABLED=False
   ```

8. Verify installed tools:

   ```bash
   python3.12 --version
   py312/bin/python --version
   uv --version
   cmake --version
   openscad --version
   nvidia-smi
   /workspace/cuda-multi-view-stereo/build/app_patch_match_mvs --help
   ```

9. Verify app startup and HTTP API:

   ```bash
   py312/bin/python src/main.py
   curl http://localhost:8000/
   ```

## Notes

- Remote CUDA is disabled because this machine will host both the main app and CUDA reconstruction locally.
- Gemini image generation still requires `GEMINI_API_KEY` in `.env`; this plan does not add secrets.
- The full CUDA MVS build may require additional packages depending on the upstream repository's CMake requirements.
