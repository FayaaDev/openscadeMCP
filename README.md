# OpenSCAD MCP Server

FastAPI-based MCP wrapper for two supported workflows:

1. Parametric text-to-OpenSCAD primitives
2. Gemini multi-view image generation -> CUDA MVS reconstruction -> OpenSCAD import wrapper

## Supported Scope

- Parametric primitive generation and export
- Gemini image generation
- Multi-view approval workflow before reconstruction
- Local or remote CUDA Multi-View Stereo reconstruction
- Automatic `.scad` wrapper generation for reconstructed models
- RunPod-hosted remote CUDA server support

## Removed Scope

- Venice image generation
- SAM segmentation
- Printer discovery and printing
- Legacy threestudio/image-to-model experiments

## Repo Layout

```text
src/
  main.py                    FastAPI app and MCP tool surface
  config.py                  Environment-driven configuration
  ai/gemini_api.py           Gemini image generation
  models/cuda_mvs.py         Local CUDA MVS wrapper
  models/code_generator.py   Parametric OpenSCAD primitives
  nlp/parameter_extractor.py Primitive parameter extraction
  openscad_wrapper/wrapper.py
  remote/                    Remote CUDA client/server code
  utils/cad_exporter.py      Primitive export helpers
  workflow/image_approval.py Approval handling
```

## Setup

Use Python `3.12`.

```bash
uv pip install --python py312/bin/python -r requirements.txt
```

Create `.env` with at least:

```env
GEMINI_API_KEY=your-key
GEMINI_MODEL=gemini-2.5-flash-image

REMOTE_CUDA_MVS_ENABLED=True
REMOTE_CUDA_MVS_USE_LAN_DISCOVERY=False
REMOTE_CUDA_MVS_SERVER_URL=http://YOUR_RUNPOD_HOST:8765
REMOTE_CUDA_MVS_API_KEY=YOUR_SHARED_SECRET
REMOTE_CUDA_MVS_WAIT_FOR_COMPLETION=True
REMOTE_CUDA_MVS_DEFAULT_FORMAT=obj
```

## Run

```bash
py312/bin/python src/main.py
```

The server starts on `http://localhost:8000`.

Inspect available tools:

```bash
curl http://localhost:8000/
```

Call tools with:

```bash
curl -X POST http://localhost:8000/tool_call \
  -H "Content-Type: application/json" \
  -d '{"tool_name":"generate_multi_view_images","tool_params":{"prompt":"low poly rabbit","num_views":4}}'
```

## Main Tools

- `create_3d_model`
- `modify_3d_model`
- `export_model`
- `generate_image_gemini`
- `generate_multi_view_images`
- `approve_image`
- `reject_image`
- `create_3d_model_from_images`
- `create_3d_model_from_text`
- `discover_remote_cuda_mvs_servers`
- `get_remote_job_status`
- `download_remote_model_result`
- `cancel_remote_job`

## Reconstruction Flow

1. Generate multi-view images with Gemini.
2. Approve at least the minimum number of views.
3. Start CUDA reconstruction locally or on the configured remote server.
4. Download the resulting mesh.
5. Automatically generate a `.scad` file that imports the reconstructed model.

Generated files are served from `/output/...`.

## RunPod

Use:

- `runpod.md` for manual setup
- `runpod-bootstrap.sh` for one-shot bootstrap

The remote CUDA HTTP service lives in `src/remote/cuda_mvs_server.py`.
