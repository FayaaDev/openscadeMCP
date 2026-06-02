## OpenSCAD MCP Server

### Setup
- Use Python `3.12` for local work on this repo.
- Preferred isolated runtime: `py312/`.
- Install deps with `uv pip install --python py312/bin/python -r requirements.txt`.
- Gemini image generation defaults to `GEMINI_MODEL=gemini-2.5-flash-image` unless overridden in `.env`.

### Bare Metal Linux RTX 4090
- Use Ubuntu `22.04` or `24.04` on a machine with an NVIDIA RTX `4090`.
- Install system packages:
  - `sudo apt-get update`
  - `sudo apt-get install -y git build-essential cmake python3.12 python3.12-venv python3-pip openscad`
- Install a recent NVIDIA driver and confirm the GPU is visible with `nvidia-smi`.
- Clone and build CUDA Multi-View Stereo alongside this repo:
  - `git clone https://github.com/fixstars/cuda-multi-view-stereo.git`
  - `mkdir -p cuda-multi-view-stereo/build`
  - `cmake -S cuda-multi-view-stereo -B cuda-multi-view-stereo/build`
  - `cmake --build cuda-multi-view-stereo/build -j"$(nproc)"`
- Create the Python runtime for this repo:
  - `python3.12 -m venv py312`
  - `uv pip install --python py312/bin/python -r requirements.txt`
- Configure `.env` for the local image -> CUDA -> OpenSCAD path:
  - `GEMINI_API_KEY=...`
  - `GEMINI_MODEL=gemini-2.5-flash-image`
  - `CUDA_MVS_PATH=/absolute/path/to/cuda-multi-view-stereo`
  - `REMOTE_CUDA_MVS_ENABLED=False`
- Start the app with `py312/bin/python src/main.py`.
- This local path supports:
  - Gemini image generation
  - multi-view approval
  - local CUDA reconstruction on the 4090
  - automatic `.scad` import wrapper generation for the reconstructed model

### Run
- Start the app with `py312/bin/python src/main.py`.
- The app serves FastAPI on `http://localhost:8000`.
- Inspect available tools with `GET /`.
- Call tools with `POST /tool_call` using JSON of the form:
  - `{"tool_name":"generate_multi_view_images","tool_params":{"prompt":"low poly rabbit","num_views":4}}`
- Model previews are available at `/ui/preview/{model_id}`.
- Exported model downloads are available at `/download/{model_id}`.
- Generated images, approved images, previews, and downloaded model assets are served from `/output/...`.

### Current Caveats
- `CUDA_MVS_PATH` defaults to `./cuda-mvs`, which is not installed in this repo.
- Full CUDA reconstruction is not expected to run locally on macOS; prefer remote CUDA for that path.
- Gemini features require an API key in `.env`.
- The legacy Gemini model `gemini-2.0-flash-exp-image-generation` no longer works here; use the configured default or set `GEMINI_MODEL` explicitly.
- Remote CUDA is enabled by default unless disabled in environment config.
- When `REMOTE_CUDA_MVS_SERVER_URL` is configured, the app now prefers that configured server before falling back to LAN discovery.
- `create_3d_model_from_images` only accepts an approved `multi_view_id`; it does not accept a raw local `image_path`.
- Reconstructed models now generate an OpenSCAD `.scad` import wrapper automatically, but the imported mesh is still a mesh, not a recovered parametric primitive.
- The currently working interface is the FastAPI HTTP wrapper around the registered tools, not a separate stdio MCP transport.

### Remote CUDA
- The remote HTTP server lives in `src/remote/cuda_mvs_server.py`.
- For cloud GPU use, set:
  - `REMOTE_CUDA_MVS_ENABLED=True`
  - `REMOTE_CUDA_MVS_USE_LAN_DISCOVERY=False`
  - `REMOTE_CUDA_MVS_SERVER_URL=http://...` or `https://...`
  - `REMOTE_CUDA_MVS_API_KEY=...`
- For local bare-metal GPU use instead, set:
  - `REMOTE_CUDA_MVS_ENABLED=False`
  - `CUDA_MVS_PATH=/absolute/path/to/cuda-multi-view-stereo`
- RunPod setup notes are in `runpod.md`.
- A one-shot bootstrap script for a RunPod GPU pod is in `runpod-bootstrap.sh`.

### Working Rules
- Make the smallest correct change.
- Keep CUDA and optional AI features guarded so the app can boot without every external dependency configured.
- Do not commit secrets or API keys.
