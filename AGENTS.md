## OpenSCAD MCP Server

### Setup
- Use Python `3.12` for local work on this repo.
- Preferred isolated runtime: `py312/`.
- Install deps with `uv pip install --python py312/bin/python -r requirements.txt`.
- Gemini image generation defaults to `GEMINI_MODEL=gemini-2.5-flash-image` unless overridden in `.env`.

### Bare Metal Linux RTX 4090
- Use Ubuntu `22.04` or `24.04` on a machine with an NVIDIA RTX `4090`.
- Install system packages. If running as `root`, omit `sudo`:
  - `sudo apt-get update`
  - `sudo apt-get install -y git build-essential cmake python3.12 python3.12-venv python3-pip openscad`
- Install a recent NVIDIA driver and confirm the GPU is visible with `nvidia-smi`.
- Confirm CUDA Toolkit is installed and `nvcc --version` works. The verified local setup used CUDA `12.4` with NVIDIA driver `580.126.20`.
- CUDA Multi-View Stereo requires OpenCV built with CUDA support. Ubuntu's packaged OpenCV links but fails at runtime with `(-216:No CUDA support)`.
- Build OpenCV with CUDA and `opencv_contrib` alongside this repo:
  - `sudo apt-get install -y build-essential cmake git pkg-config ccache ninja-build libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev gfortran openexr libopenexr-dev libatlas-base-dev libtbb-dev libeigen3-dev libdc1394-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libglew-dev libvtk9-dev python3-dev python3-numpy`
  - `git clone --branch 4.10.0 --depth 1 https://github.com/opencv/opencv.git /workspace/opencv`
  - `git clone --branch 4.10.0 --depth 1 https://github.com/opencv/opencv_contrib.git /workspace/opencv_contrib`
  - `cmake -S /workspace/opencv -B /workspace/opencv/build-cuda -G Ninja -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/workspace/opencv-cuda-install -D OPENCV_EXTRA_MODULES_PATH=/workspace/opencv_contrib/modules -D WITH_CUDA=ON -D CUDA_ARCH_BIN=8.9 -D CUDA_ARCH_PTX= -D WITH_CUBLAS=ON -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON -D WITH_TBB=ON -D WITH_VTK=ON -D WITH_OPENGL=ON -D BUILD_LIST=core,imgproc,imgcodecs,highgui,viz,cudev -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=OFF -D BUILD_JAVA=OFF -D OPENCV_GENERATE_PKGCONFIG=ON`
  - `cmake --build /workspace/opencv/build-cuda --target install -j24`
  - `/workspace/opencv-cuda-install/bin/opencv_version --verbose`
- Clone CUDA Multi-View Stereo alongside this repo and build it against the CUDA-enabled OpenCV install:
  - `git clone https://github.com/fixstars/cuda-multi-view-stereo.git /workspace/cuda-multi-view-stereo`
  - `cmake -S /workspace/cuda-multi-view-stereo -B /workspace/cuda-multi-view-stereo/build-opencv-cuda -D OpenCV_DIR=/workspace/opencv-cuda-install/lib/cmake/opencv4 -D CMAKE_PREFIX_PATH=/workspace/opencv-cuda-install -D CMAKE_BUILD_RPATH=/workspace/opencv-cuda-install/lib`
  - `cmake --build /workspace/cuda-multi-view-stereo/build-opencv-cuda -j"$(nproc)"`
  - `ldd /workspace/cuda-multi-view-stereo/build-opencv-cuda/samples/app_patch_match_mvs` should show OpenCV libraries from `/workspace/opencv-cuda-install/lib`.
- Create the Python runtime for this repo:
  - `python3.12 -m venv py312`
  - `uv pip install --python py312/bin/python -r requirements.txt`
- Configure `.env` for the local image -> CUDA -> OpenSCAD path:
  - `GEMINI_API_KEY=...`
  - `GEMINI_MODEL=gemini-2.5-flash-image`
  - `CUDA_MVS_PATH=/workspace/cuda-multi-view-stereo`
  - `CUDA_MVS_USE_GPU=True`
  - `REMOTE_CUDA_MVS_ENABLED=False`
- Start the app with `py312/bin/python src/main.py`.
- This local path supports:
  - Gemini image generation
  - multi-view approval
  - local CUDA reconstruction on the 4090
  - automatic `.scad` import wrapper generation for the reconstructed model
  - generated `.scad` wrappers under `scad/`, with reconstructed meshes under `output/models/`

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
- For local Linux RTX reconstruction, CUDA MVS should be built in `/workspace/cuda-multi-view-stereo/build-opencv-cuda`; the app wrapper prefers that executable before falling back to `build/`.
- If CUDA MVS fails with `OpenCV ... No CUDA support`, it was linked against Ubuntu's non-CUDA OpenCV. Rebuild OpenCV with CUDA and rebuild CUDA MVS using `OpenCV_DIR=/workspace/opencv-cuda-install/lib/cmake/opencv4`.
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
