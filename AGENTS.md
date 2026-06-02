## OpenSCAD MCP Server

### Setup
- Use Python `3.12` for local work on this repo.
- Preferred isolated runtime: `py312/`.
- Install deps with `uv pip install --python py312/bin/python -r requirements.txt`.

### Run
- Start the app with `py312/bin/python src/main.py`.
- The app serves FastAPI on `http://localhost:8000`.
- The current tool interface is `POST /tool_call`.

### Current Caveats
- `src/main.py` has startup issues: missing imports, missing `src.ai.sam_segmentation`, and config mismatches.
- `CUDA_MVS_PATH` defaults to `./cuda-mvs`, which is not installed in this repo.
- Full CUDA reconstruction is not expected to run locally on macOS; prefer remote CUDA for that path.

### Working Rules
- Make the smallest correct change.
- Keep CUDA and optional AI features guarded so the app can boot without every external dependency configured.
- Do not commit secrets or API keys.
