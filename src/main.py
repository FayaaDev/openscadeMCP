import os
import logging
import sys
import uuid
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from mcp.server.mcpserver.server import MCPServer

if __package__ in {None, ""}:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from src.config import *

# Import components
from src.nlp.parameter_extractor import ParameterExtractor
from src.models.code_generator import CodeGenerator
from src.openscad_wrapper.wrapper import OpenSCADWrapper
from src.utils.cad_exporter import CADExporter
from src.ai.gemini_api import GeminiImageGenerator
from src.models.cuda_mvs import CUDAMultiViewStereo
from src.remote.connection_manager import CUDAMVSConnectionManager
from src.workflow.image_approval import ImageApprovalTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="OpenSCAD MCP Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
parameter_extractor = ParameterExtractor()
code_generator = CodeGenerator("scad", "output")
openscad_wrapper = OpenSCADWrapper("scad", "output")
cad_exporter = CADExporter()


def initialize_local_cuda_mvs() -> Optional[CUDAMultiViewStereo]:
    if not os.path.exists(CUDA_MVS_PATH):
        logger.warning("Local CUDA MVS not available at %s; local reconstruction disabled", CUDA_MVS_PATH)
        return None

    try:
        return CUDAMultiViewStereo(CUDA_MVS_PATH, MODELS_DIR)
    except Exception as exc:
        logger.warning("Failed to initialize local CUDA MVS: %s", exc)
        return None


def initialize_remote_cuda_mvs() -> Optional[CUDAMVSConnectionManager]:
    if not REMOTE_CUDA_MVS["ENABLED"]:
        return None

    try:
        manager = CUDAMVSConnectionManager(
            api_key=REMOTE_CUDA_MVS["API_KEY"] or None,
            discovery_port=REMOTE_CUDA_MVS["DISCOVERY_PORT"],
            connection_timeout=REMOTE_CUDA_MVS["CONNECTION_TIMEOUT"],
            health_check_interval=REMOTE_CUDA_MVS["HEALTH_CHECK_INTERVAL"],
            auto_discover=REMOTE_CUDA_MVS["USE_LAN_DISCOVERY"],
        )

        if REMOTE_CUDA_MVS["SERVER_URL"]:
            manager.add_server(
                {
                    "server_id": "configured-server",
                    "name": "Configured CUDA MVS Server",
                    "url": REMOTE_CUDA_MVS["SERVER_URL"],
                    "status": "available",
                }
            )

        return manager
    except Exception as exc:
        logger.warning("Failed to initialize remote CUDA MVS: %s", exc)
        return None

# Initialize AI components
gemini_generator = GeminiImageGenerator(GEMINI_API_KEY, IMAGES_DIR)
cuda_mvs = initialize_local_cuda_mvs()
image_approval = ImageApprovalTool(APPROVED_IMAGES_DIR)

# Initialize remote processing components if enabled
remote_connection_manager = initialize_remote_cuda_mvs()

# Store models in memory
models = {}
approved_images = {}
remote_jobs = {}

# Create MCP server
mcp_server = MCPServer(name="openscad-mcp-server")
tool_registry: Dict[str, Any] = {}


def register_tool(fn):
    mcp_server.tool()(fn)
    tool_registry[fn.__name__] = fn
    return fn


def require_local_cuda_mvs() -> CUDAMultiViewStereo:
    if not cuda_mvs:
        raise ValueError("Local CUDA MVS is not available on this machine")
    return cuda_mvs


def require_remote_cuda_mvs() -> CUDAMVSConnectionManager:
    if not REMOTE_CUDA_MVS["ENABLED"]:
        raise ValueError("Remote CUDA MVS processing is not enabled")
    if not remote_connection_manager:
        raise ValueError("Remote CUDA MVS connection manager is not initialized")
    return remote_connection_manager


def discover_remote_servers() -> List[Dict[str, Any]]:
    manager = require_remote_cuda_mvs()

    # Prefer already configured servers (for example a cloud-hosted endpoint)
    # before falling back to LAN discovery.
    known_servers = manager.get_servers()
    if known_servers:
        manager.check_all_servers()
        return manager.get_servers()

    return manager.discover_servers()


def get_job_status(job_id: str) -> Dict[str, Any]:
    manager = require_remote_cuda_mvs()
    if job_id not in remote_jobs:
        raise ValueError(f"Job with ID {job_id} not found")

    job_info = remote_jobs[job_id]
    status = manager.get_job_status(job_id, job_info["server_id"])
    if status.get("status") == "success" and "job_info" in status:
        job_state = status["job_info"]
        job_info["status"] = job_state.get("status", job_info.get("status", "unknown"))
        job_info["progress"] = job_state.get("progress", 0)
        job_info["message"] = job_state.get("message", "")
    return job_info


def download_remote_model(job_id: str) -> Dict[str, Any]:
    manager = require_remote_cuda_mvs()
    if job_id not in remote_jobs:
        raise ValueError(f"Job with ID {job_id} not found")

    job_info = remote_jobs[job_id]
    result = manager.download_model(job_id, job_info["server_id"], REMOTE_CUDA_MVS["DEFAULT_OUTPUT_FORMAT"])
    if result.get("status") != "success":
        raise ValueError(result.get("message", "Failed to download remote model"))

    return {
        "model_path": result.get("local_path"),
        "point_cloud_path": None,
        "format": result.get("format"),
    }


def cancel_job(job_id: str) -> Dict[str, Any]:
    manager = require_remote_cuda_mvs()
    if job_id not in remote_jobs:
        raise ValueError(f"Job with ID {job_id} not found")

    job_info = remote_jobs[job_id]
    result = manager.cancel_job(job_id, job_info["server_id"])
    return {"cancelled": result.get("status") == "success", "message": result.get("message", "")}


def build_local_model_from_images(image_paths: List[str], output_name: str) -> Dict[str, Optional[str]]:
    local_cuda_mvs = require_local_cuda_mvs()
    result = local_cuda_mvs.generate_model_from_images(image_paths=image_paths, output_name=output_name)
    point_cloud_path = result.get("point_cloud_file")
    model_path = point_cloud_path

    if point_cloud_path and REMOTE_CUDA_MVS["DEFAULT_OUTPUT_FORMAT"] == "obj":
        model_path = local_cuda_mvs.convert_ply_to_obj(point_cloud_path)

    return {
        "model_path": model_path,
        "point_cloud_path": point_cloud_path,
    }


def build_output_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None

    relative_path = os.path.relpath(path, OUTPUT_DIR)
    return f"/output/{relative_path.replace(os.sep, '/')}"


def create_import_scad(model_path: str, model_id: str) -> Dict[str, Any]:
    scad_code = f"""// Generated OpenSCAD import for model {model_id}
scale_factor = 1.0;
position_x = 0;
position_y = 0;
position_z = 0;
rotation_x = 0;
rotation_y = 0;
rotation_z = 0;

translate([position_x, position_y, position_z])
rotate([rotation_x, rotation_y, rotation_z])
scale(scale_factor)
import(\"{model_path}\");
"""

    scad_file = openscad_wrapper.generate_scad(scad_code, model_id)
    previews = openscad_wrapper.generate_multi_angle_previews(scad_file)
    return {"scad_file": scad_file, "previews": previews}

# Mount output files used by generated images and previews.
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

# Create Jinja2 templates
os.makedirs("templates", exist_ok=True)
templates = Jinja2Templates(directory="templates")

# Create model preview template
with open("templates/preview.html", "w") as f:
    f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenSCAD Model Preview</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
        }
        .preview-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 20px;
        }
        .preview-image {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            background-color: white;
        }
        .preview-image img {
            max-width: 100%;
            height: auto;
        }
        .preview-image h3 {
            margin-top: 10px;
            margin-bottom: 5px;
            color: #555;
        }
        .parameters {
            margin-top: 20px;
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #eee;
        }
        .parameters h2 {
            margin-top: 0;
            color: #333;
        }
        .parameters table {
            width: 100%;
            border-collapse: collapse;
        }
        .parameters table th, .parameters table td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .parameters table th {
            background-color: #f2f2f2;
        }
        .actions {
            margin-top: 20px;
            display: flex;
            gap: 10px;
        }
        .actions button {
            padding: 10px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .actions button:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>OpenSCAD Model Preview: {{ model_id }}</h1>
        
        <div class="parameters">
            <h2>Parameters</h2>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
                {% for key, value in parameters.items() %}
                <tr>
                    <td>{{ key }}</td>
                    <td>{{ value }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        
        <div class="preview-container">
            {% for view, image_path in previews.items() %}
            <div class="preview-image">
                <h3>{{ view|title }} View</h3>
                <img src="{{ image_path }}" alt="{{ view }} view">
            </div>
            {% endfor %}
        </div>
        
        <div class="actions">
            <button onclick="window.location.href='/download/{{ model_id }}'">Download Model</button>
        </div>
    </div>
</body>
</html>
    """)

# Define MCP tools
@register_tool
def create_3d_model(description: str) -> Dict[str, Any]:
    """
    Create a 3D model from a natural language description.
    
    Args:
        description: Natural language description of the 3D model
        
    Returns:
        Dictionary with model information
    """
    # Extract parameters from description
    model_type, parameters = parameter_extractor.extract_parameters(description)
    
    # Generate a unique model ID
    model_id = str(uuid.uuid4())
    
    # Generate OpenSCAD code
    scad_code = code_generator.generate_code(model_type, parameters)
    
    # Save the SCAD file
    scad_file = openscad_wrapper.generate_scad(scad_code, model_id)
    
    # Generate preview images
    previews = openscad_wrapper.generate_multi_angle_previews(scad_file, parameters)
    
    # Export to parametric format (CSG by default)
    success, model_file, error = cad_exporter.export_model(
        scad_file, 
        "csg",
        parameters,
        metadata={
            "description": description,
            "model_type": model_type,
        }
    )
    
    # Store model information
    models[model_id] = {
        "id": model_id,
        "type": model_type,
        "parameters": parameters,
        "description": description,
        "scad_file": scad_file,
        "model_file": model_file if success else None,
        "previews": previews,
        "format": "csg"
    }
    
    # Create response
    response = {
        "model_id": model_id,
        "model_type": model_type,
        "parameters": parameters,
        "preview_url": f"/ui/preview/{model_id}",
        "supported_formats": cad_exporter.get_supported_formats()
    }
    
    return response

@register_tool
def modify_3d_model(model_id: str, modifications: str) -> Dict[str, Any]:
    """
    Modify an existing 3D model.
    
    Args:
        model_id: ID of the model to modify
        modifications: Natural language description of the modifications
        
    Returns:
        Dictionary with updated model information
    """
    # Check if model exists
    if model_id not in models:
        raise ValueError(f"Model with ID {model_id} not found")
    
    # Get existing model information
    model_info = models[model_id]
    
    # Extract parameters from modifications
    _, new_parameters = parameter_extractor.extract_parameters(
        modifications, 
        model_type=model_info["type"],
        existing_parameters=model_info["parameters"]
    )
    
    # Generate OpenSCAD code with updated parameters
    scad_code = code_generator.generate_code(model_info["type"], new_parameters)
    
    # Save the SCAD file
    scad_file = openscad_wrapper.generate_scad(scad_code, model_id)
    
    # Generate preview images
    previews = openscad_wrapper.generate_multi_angle_previews(scad_file, new_parameters)
    
    # Export to parametric format (same as original)
    success, model_file, error = cad_exporter.export_model(
        scad_file, 
        model_info["format"],
        new_parameters,
        metadata={
            "description": model_info["description"] + " | " + modifications,
            "model_type": model_info["type"],
        }
    )
    
    # Update model information
    models[model_id] = {
        "id": model_id,
        "type": model_info["type"],
        "parameters": new_parameters,
        "description": model_info["description"] + " | " + modifications,
        "scad_file": scad_file,
        "model_file": model_file if success else None,
        "previews": previews,
        "format": model_info["format"]
    }
    
    # Create response
    response = {
        "model_id": model_id,
        "model_type": model_info["type"],
        "parameters": new_parameters,
        "preview_url": f"/ui/preview/{model_id}",
        "supported_formats": cad_exporter.get_supported_formats()
    }
    
    return response

@register_tool
def export_model(model_id: str, format: str = "csg") -> Dict[str, Any]:
    """
    Export a 3D model to a specific format.
    
    Args:
        model_id: ID of the model to export
        format: Format to export to (csg, stl, obj, etc.)
        
    Returns:
        Dictionary with export information
    """
    # Check if model exists
    if model_id not in models:
        raise ValueError(f"Model with ID {model_id} not found")
    
    # Get model information
    model_info = models[model_id]
    
    # Check if format is supported
    supported_formats = cad_exporter.get_supported_formats()
    if format not in supported_formats:
        raise ValueError(f"Format {format} not supported. Supported formats: {', '.join(supported_formats)}")
    
    # Export model
    success, model_file, error = cad_exporter.export_model(
        model_info["scad_file"],
        format,
        model_info["parameters"],
        metadata={
            "description": model_info["description"],
            "model_type": model_info["type"],
        }
    )
    
    if not success:
        raise ValueError(f"Failed to export model: {error}")
    
    # Update model information
    models[model_id]["model_file"] = model_file
    models[model_id]["format"] = format
    
    # Create response
    response = {
        "model_id": model_id,
        "format": format,
        "model_file": model_file,
        "download_url": f"/download/{model_id}"
    }
    
    return response

# Add Google Gemini image generation tool
@register_tool
def generate_image_gemini(prompt: str, model: str = GEMINI_MODEL) -> Dict[str, Any]:
    """
    Generate an image using Google Gemini's image generation models.
    
    Args:
        prompt: Text description for image generation
        model: Model to use (default: gemini-2.0-flash-exp-image-generation)
        
    Returns:
        Dictionary with image information
    """
    # Generate a unique image ID
    image_id = str(uuid.uuid4())
    
    # Generate image
    result = gemini_generator.generate_image(prompt, model)
    
    # Create response
    response = {
        "image_id": image_id,
        "prompt": prompt,
        "model": model,
        "image_path": result.get("local_path"),
        "image_url": build_output_url(result.get("local_path")),
    }
    
    return response


# Add multi-view image generation tool
@register_tool
def generate_multi_view_images(prompt: str, num_views: int = 4) -> Dict[str, Any]:
    """
    Generate multiple views of the same 3D object using Google Gemini.
    
    Args:
        prompt: Text description of the 3D object
        num_views: Number of views to generate (default: 4)
        
    Returns:
        Dictionary with multi-view image information
    """
    # Validate number of views
    if num_views < MULTI_VIEW_PIPELINE["MIN_NUM_VIEWS"]:
        raise ValueError(f"Number of views must be at least {MULTI_VIEW_PIPELINE['MIN_NUM_VIEWS']}")
    
    if num_views > MULTI_VIEW_PIPELINE["MAX_NUM_VIEWS"]:
        raise ValueError(f"Number of views cannot exceed {MULTI_VIEW_PIPELINE['MAX_NUM_VIEWS']}")
    
    # Generate a unique multi-view ID
    multi_view_id = str(uuid.uuid4())
    
    # Generate multi-view images
    output_dir = os.path.join(MULTI_VIEW_DIR, multi_view_id)
    results = gemini_generator.generate_multiple_views(prompt, num_views, output_dir=output_dir)
    
    # Create response
    response = {
        "multi_view_id": multi_view_id,
        "prompt": prompt,
        "num_views": num_views,
        "views": [
            {
                "view_id": result.get("view_id", f"view_{i+1}"),
                "view_index": result.get("view_index", i+1),
                "view_direction": result.get("view_direction", ""),
                "image_path": result.get("local_path"),
                "image_url": build_output_url(result.get("local_path")),
            }
            for i, result in enumerate(results)
        ],
        "approval_required": IMAGE_APPROVAL["ENABLED"] and not IMAGE_APPROVAL["AUTO_APPROVE"]
    }
    
    # Store multi-view information for approval
    if IMAGE_APPROVAL["ENABLED"]:
        approved_images[multi_view_id] = {
            "multi_view_id": multi_view_id,
            "prompt": prompt,
            "num_views": num_views,
            "views": response["views"],
            "approved_views": [] if not IMAGE_APPROVAL["AUTO_APPROVE"] else [view["view_id"] for view in response["views"]],
            "rejected_views": [],
            "approval_complete": IMAGE_APPROVAL["AUTO_APPROVE"]
        }
    
    return response


# Add image approval tool
@register_tool
def approve_image(multi_view_id: str, view_id: str) -> Dict[str, Any]:
    """
    Approve an image for 3D model generation.
    
    Args:
        multi_view_id: ID of the multi-view set
        view_id: ID of the view to approve
        
    Returns:
        Dictionary with approval information
    """
    # Check if multi-view ID exists
    if multi_view_id not in approved_images:
        raise ValueError(f"Multi-view set with ID {multi_view_id} not found")
    
    # Get multi-view information
    multi_view_info = approved_images[multi_view_id]
    
    # Check if view ID exists
    view_exists = False
    for view in multi_view_info["views"]:
        if view["view_id"] == view_id:
            view_exists = True
            break
    
    if not view_exists:
        raise ValueError(f"View with ID {view_id} not found in multi-view set {multi_view_id}")
    
    # Check if view is already approved
    if view_id in multi_view_info["approved_views"]:
        return {
            "multi_view_id": multi_view_id,
            "view_id": view_id,
            "status": "already_approved",
            "approved_views": multi_view_info["approved_views"],
            "rejected_views": multi_view_info["rejected_views"],
            "approval_complete": multi_view_info["approval_complete"]
        }
    
    # Remove from rejected views if present
    if view_id in multi_view_info["rejected_views"]:
        multi_view_info["rejected_views"].remove(view_id)
    
    # Add to approved views
    multi_view_info["approved_views"].append(view_id)
    
    # Check if approval is complete
    if len(multi_view_info["approved_views"]) >= IMAGE_APPROVAL["MIN_APPROVED_IMAGES"]:
        multi_view_info["approval_complete"] = True
    
    # Create response
    response = {
        "multi_view_id": multi_view_id,
        "view_id": view_id,
        "status": "approved",
        "approved_views": multi_view_info["approved_views"],
        "rejected_views": multi_view_info["rejected_views"],
        "approval_complete": multi_view_info["approval_complete"]
    }
    
    return response


# Add image rejection tool
@register_tool
def reject_image(multi_view_id: str, view_id: str) -> Dict[str, Any]:
    """
    Reject an image for 3D model generation.
    
    Args:
        multi_view_id: ID of the multi-view set
        view_id: ID of the view to reject
        
    Returns:
        Dictionary with rejection information
    """
    # Check if multi-view ID exists
    if multi_view_id not in approved_images:
        raise ValueError(f"Multi-view set with ID {multi_view_id} not found")
    
    # Get multi-view information
    multi_view_info = approved_images[multi_view_id]
    
    # Check if view ID exists
    view_exists = False
    for view in multi_view_info["views"]:
        if view["view_id"] == view_id:
            view_exists = True
            break
    
    if not view_exists:
        raise ValueError(f"View with ID {view_id} not found in multi-view set {multi_view_id}")
    
    # Check if view is already rejected
    if view_id in multi_view_info["rejected_views"]:
        return {
            "multi_view_id": multi_view_id,
            "view_id": view_id,
            "status": "already_rejected",
            "approved_views": multi_view_info["approved_views"],
            "rejected_views": multi_view_info["rejected_views"],
            "approval_complete": multi_view_info["approval_complete"]
        }
    
    # Remove from approved views if present
    if view_id in multi_view_info["approved_views"]:
        multi_view_info["approved_views"].remove(view_id)
    
    # Add to rejected views
    multi_view_info["rejected_views"].append(view_id)
    
    # Check if approval is complete
    if len(multi_view_info["approved_views"]) >= IMAGE_APPROVAL["MIN_APPROVED_IMAGES"]:
        multi_view_info["approval_complete"] = True
    else:
        multi_view_info["approval_complete"] = False
    
    # Create response
    response = {
        "multi_view_id": multi_view_id,
        "view_id": view_id,
        "status": "rejected",
        "approved_views": multi_view_info["approved_views"],
        "rejected_views": multi_view_info["rejected_views"],
        "approval_complete": multi_view_info["approval_complete"]
    }
    
    return response


# Add 3D model generation from approved images tool
@register_tool
def create_3d_model_from_images(multi_view_id: str, output_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a 3D model from approved multi-view images.
    
    Args:
        multi_view_id: ID of the multi-view set
        output_name: Optional name for the output model
        
    Returns:
        Dictionary with model information
    """
    # Check if multi-view ID exists
    if multi_view_id not in approved_images:
        raise ValueError(f"Multi-view set with ID {multi_view_id} not found")
    
    # Get multi-view information
    multi_view_info = approved_images[multi_view_id]
    
    # Check if approval is complete
    if not multi_view_info["approval_complete"]:
        raise ValueError(f"Approval for multi-view set {multi_view_id} is not complete")
    
    # Check if there are enough approved images
    if len(multi_view_info["approved_views"]) < IMAGE_APPROVAL["MIN_APPROVED_IMAGES"]:
        raise ValueError(f"Not enough approved images. Need at least {IMAGE_APPROVAL['MIN_APPROVED_IMAGES']}, but only have {len(multi_view_info['approved_views'])}")
    
    # Get approved image paths
    approved_image_paths = []
    for view in multi_view_info["views"]:
        if view["view_id"] in multi_view_info["approved_views"]:
            approved_image_paths.append(view["image_path"])
    
    # Generate a unique model ID
    model_id = str(uuid.uuid4())
    
    # Set output name if not provided
    if not output_name:
        output_name = f"model_{model_id[:8]}"
    
    # Create 3D model
    if REMOTE_CUDA_MVS["ENABLED"] and remote_connection_manager:
        servers = discover_remote_servers()
        if not servers:
            raise ValueError("No remote CUDA MVS servers found")

        server_id = servers[0].get("server_id") or servers[0].get("id")
        manager = require_remote_cuda_mvs()
        remote_result = manager.generate_model_from_images(
            image_paths=approved_image_paths,
            output_format=REMOTE_CUDA_MVS["DEFAULT_OUTPUT_FORMAT"],
            wait_for_completion=REMOTE_CUDA_MVS["WAIT_FOR_COMPLETION"],
            poll_interval=REMOTE_CUDA_MVS["POLL_INTERVAL"],
            server_id=server_id,
        )

        if remote_result.get("status") == "success" and remote_result.get("local_path"):
            openscad_result = create_import_scad(remote_result["local_path"], model_id)
            models[model_id] = {
                "id": model_id,
                "type": "cuda_mvs_remote",
                "parameters": {
                    "multi_view_id": multi_view_id,
                    "prompt": multi_view_info["prompt"],
                    "num_views": len(approved_image_paths),
                    "quality": REMOTE_CUDA_MVS["DEFAULT_RECONSTRUCTION_QUALITY"],
                    "output_format": REMOTE_CUDA_MVS["DEFAULT_OUTPUT_FORMAT"],
                },
                "description": f"3D model generated from {len(approved_image_paths)} views of '{multi_view_info['prompt']}'",
                "scad_file": openscad_result["scad_file"],
                "model_file": remote_result.get("local_path"),
                "point_cloud_file": None,
                "previews": openscad_result["previews"],
                "format": remote_result.get("format", REMOTE_CUDA_MVS["DEFAULT_OUTPUT_FORMAT"]),
                "remote_job_id": remote_result.get("job_id"),
            }

            response = {
                "model_id": model_id,
                "multi_view_id": multi_view_id,
                "status": "completed",
                "model_path": remote_result.get("local_path"),
                "scad_file": openscad_result["scad_file"],
                "point_cloud_path": None,
                "format": remote_result.get("format", REMOTE_CUDA_MVS["DEFAULT_OUTPUT_FORMAT"]),
                "preview_url": f"/ui/preview/{model_id}",
            }
        else:
            job_id = remote_result.get("job_id")
            if not job_id:
                raise ValueError(remote_result.get("message", "Failed to process images remotely"))

            remote_jobs[job_id] = {
                "model_id": model_id,
                "multi_view_id": multi_view_id,
                "server_id": remote_result.get("server_id", server_id),
                "job_id": job_id,
                "status": "processing",
                "message": remote_result.get("message", ""),
            }

            response = {
                "model_id": model_id,
                "multi_view_id": multi_view_id,
                "status": "processing",
                "job_id": job_id,
                "server_id": remote_result.get("server_id", server_id),
            }
    else:
        result = build_local_model_from_images(approved_image_paths, output_name)
        openscad_result = create_import_scad(result["model_path"], model_id)
        
        # Store model information
        models[model_id] = {
            "id": model_id,
            "type": "cuda_mvs_local",
            "parameters": {
                "multi_view_id": multi_view_id,
                "prompt": multi_view_info["prompt"],
                "num_views": len(approved_image_paths),
                "quality": REMOTE_CUDA_MVS["DEFAULT_RECONSTRUCTION_QUALITY"],
                "output_format": REMOTE_CUDA_MVS["DEFAULT_OUTPUT_FORMAT"]
            },
            "description": f"3D model generated from {len(approved_image_paths)} views of '{multi_view_info['prompt']}'",
            "scad_file": openscad_result["scad_file"],
            "model_file": result.get("model_path"),
            "point_cloud_file": result.get("point_cloud_path"),
            "previews": openscad_result["previews"],
            "format": REMOTE_CUDA_MVS["DEFAULT_OUTPUT_FORMAT"]
        }
        
        # Create response
        response = {
            "model_id": model_id,
            "multi_view_id": multi_view_id,
            "status": "completed",
            "model_path": result.get("model_path"),
            "scad_file": openscad_result["scad_file"],
            "point_cloud_path": result.get("point_cloud_path"),
            "format": REMOTE_CUDA_MVS["DEFAULT_OUTPUT_FORMAT"],
            "preview_url": f"/ui/preview/{model_id}",
        }
    
    return response


# Add complete pipeline tool (text to 3D model)
@register_tool
def create_3d_model_from_text(prompt: str, num_views: int = 4, wait_for_completion: bool = True) -> Dict[str, Any]:
    """
    Create a 3D model from a text description using the complete pipeline.
    
    Args:
        prompt: Text description of the 3D object
        num_views: Number of views to generate (default: 4)
        wait_for_completion: Whether to wait for remote processing to complete
        
    Returns:
        Dictionary with model information
    """
    # Generate multi-view images
    multi_view_result = generate_multi_view_images(prompt, num_views)
    
    multi_view_id = multi_view_result["multi_view_id"]
    
    # Auto-approve all images if enabled
    if IMAGE_APPROVAL["AUTO_APPROVE"]:
        for view in multi_view_result["views"]:
            approve_image(multi_view_id, view["view_id"])
    else:
        # Return multi-view result for manual approval
        return {
            "status": "awaiting_approval",
            "message": "Please approve or reject each image before proceeding",
            "multi_view_id": multi_view_id,
            "views": multi_view_result["views"]
        }
    
    # Create 3D model from approved images
    model_result = create_3d_model_from_images(multi_view_id)
    
    # If remote processing is not waiting for completion, return job information
    if not wait_for_completion and model_result.get("status") == "processing":
        return model_result
    
    # Return model information
    return model_result


# Add remote CUDA MVS server discovery tool
@register_tool
def discover_remote_cuda_mvs_servers() -> Dict[str, Any]:
    """
    Discover remote CUDA MVS servers on the network.
    
    Returns:
        Dictionary with discovered servers
    """
    if not REMOTE_CUDA_MVS["ENABLED"]:
        raise ValueError("Remote CUDA MVS processing is not enabled")
    
    if not remote_connection_manager:
        raise ValueError("Remote CUDA MVS connection manager is not initialized")
    
    servers = discover_remote_servers()
    
    return {
        "servers": servers,
        "count": len(servers)
    }


# Add remote job status tool
@register_tool
def get_remote_job_status(job_id: str) -> Dict[str, Any]:
    """
    Get the status of a remote CUDA MVS processing job.
    
    Args:
        job_id: ID of the job to get status for
        
    Returns:
        Dictionary with job status
    """
    if not REMOTE_CUDA_MVS["ENABLED"]:
        raise ValueError("Remote CUDA MVS processing is not enabled")
    
    if not remote_connection_manager:
        raise ValueError("Remote CUDA MVS connection manager is not initialized")
    
    # Check if job exists
    if job_id not in remote_jobs:
        raise ValueError(f"Job with ID {job_id} not found")
    
    # Get job information
    job_info = remote_jobs[job_id]
    
    # Get status from server
    status = get_job_status(job_id)
    
    if not status:
        raise ValueError(f"Failed to get status for job with ID {job_id}")
    
    # Update job information
    job_info["status"] = status.get("status", job_info["status"])
    job_info["progress"] = status.get("progress", 0)
    job_info["message"] = status.get("message", "")
    
    return job_info


# Add remote model download tool
@register_tool
def download_remote_model_result(job_id: str) -> Dict[str, Any]:
    """
    Download a processed model from a remote CUDA MVS server.
    
    Args:
        job_id: ID of the job to download model for
        
    Returns:
        Dictionary with model information
    """
    if not REMOTE_CUDA_MVS["ENABLED"]:
        raise ValueError("Remote CUDA MVS processing is not enabled")
    
    if not remote_connection_manager:
        raise ValueError("Remote CUDA MVS connection manager is not initialized")
    
    # Check if job exists
    if job_id not in remote_jobs:
        raise ValueError(f"Job with ID {job_id} not found")
    
    # Get job information
    job_info = remote_jobs[job_id]
    
    # Check if job is completed
    if job_info["status"] != "completed":
        raise ValueError(f"Job with ID {job_id} is not completed (status: {job_info['status']})")
    
    # Download model
    result = download_remote_model(job_id)
    
    if not result:
        raise ValueError(f"Failed to download model for job with ID {job_id}")
    
    # Update job information
    job_info["model_path"] = result.get("model_path")
    job_info["point_cloud_path"] = result.get("point_cloud_path")
    job_info["downloaded"] = True
    job_info["status"] = "completed"
    
    # Update model information if available
    if "model_id" in job_info and job_info["model_id"] in models:
        model_id = job_info["model_id"]
        openscad_result = create_import_scad(result["model_path"], model_id)
        models[model_id]["scad_file"] = openscad_result["scad_file"]
        models[model_id]["model_file"] = result.get("model_path")
        models[model_id]["point_cloud_file"] = result.get("point_cloud_path")
        models[model_id]["previews"] = openscad_result["previews"]
    
    return {
        "job_id": job_id,
        "model_path": result.get("model_path"),
        "scad_file": models[job_info["model_id"]].get("scad_file") if job_info.get("model_id") in models else None,
        "point_cloud_path": result.get("point_cloud_path"),
        "format": REMOTE_CUDA_MVS["DEFAULT_OUTPUT_FORMAT"],
        "preview_url": f"/ui/preview/{job_info['model_id']}" if job_info.get("model_id") in models else None,
    }


# Add remote job cancellation tool
@register_tool
def cancel_remote_job(job_id: str) -> Dict[str, Any]:
    """
    Cancel a remote CUDA MVS processing job.
    
    Args:
        job_id: ID of the job to cancel
        
    Returns:
        Dictionary with cancellation result
    """
    if not REMOTE_CUDA_MVS["ENABLED"]:
        raise ValueError("Remote CUDA MVS processing is not enabled")
    
    if not remote_connection_manager:
        raise ValueError("Remote CUDA MVS connection manager is not initialized")
    
    # Check if job exists
    if job_id not in remote_jobs:
        raise ValueError(f"Job with ID {job_id} not found")
    
    # Get job information
    job_info = remote_jobs[job_id]
    
    # Cancel job
    result = cancel_job(job_id)
    
    if not result:
        raise ValueError(f"Failed to cancel job with ID {job_id}")
    
    # Update job information
    if result.get("cancelled", False):
        job_info["status"] = "cancelled"
        job_info["message"] = "Job cancelled by user"
    
    return {
        "job_id": job_id,
        "cancelled": result.get("cancelled", False),
        "status": job_info["status"],
        "message": job_info.get("message", "")
    }


# FastAPI routes
@app.post("/tool_call")
async def handle_tool_call(request: Request) -> JSONResponse:
    """
    Handle a tool call from a client.
    
    Args:
        request: FastAPI request object
        
    Returns:
        JSON response with tool call result
    """
    # Parse request
    data = await request.json()
    
    # Check if tool name is provided
    if "tool_name" not in data:
        raise HTTPException(status_code=400, detail="Tool name is required")
    
    # Check if tool exists
    tool_name = data["tool_name"]
    if tool_name not in tool_registry:
        raise HTTPException(status_code=404, detail=f"Tool {tool_name} not found")
    
    # Get tool parameters
    tool_params = data.get("tool_params", {})
    
    # Call tool
    try:
        result = tool_registry[tool_name](**tool_params)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error calling tool {tool_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ui/preview/{model_id}")
async def preview_model(request: Request, model_id: str) -> Response:
    """
    Render a preview page for a model.
    
    Args:
        request: FastAPI request object
        model_id: ID of the model to preview
        
    Returns:
        HTML response with model preview
    """
    # Check if model exists
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    
    # Get model information
    model_info = models[model_id]
    
    # Render template
    return templates.TemplateResponse(
        "preview.html",
        {
            "request": request,
            "model_id": model_id,
            "parameters": model_info["parameters"],
            "previews": {view: f"/preview/{view}/{model_id}" for view in model_info["previews"]}
        }
    )

@app.get("/preview/{view}/{model_id}")
async def get_preview(view: str, model_id: str) -> FileResponse:
    """
    Get a preview image for a model.
    
    Args:
        view: View to get preview for
        model_id: ID of the model
        
    Returns:
        Image file response
    """
    # Check if model exists
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    
    # Get model information
    model_info = models[model_id]
    
    # Check if preview exists
    if view not in model_info["previews"]:
        raise HTTPException(status_code=404, detail=f"Preview for view {view} not found")
    
    # Return preview image
    return FileResponse(model_info["previews"][view])

@app.get("/download/{model_id}")
async def download_model(model_id: str) -> FileResponse:
    """
    Download a model file.
    
    Args:
        model_id: ID of the model to download
        
    Returns:
        Model file response
    """
    # Check if model exists
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    
    # Get model information
    model_info = models[model_id]
    
    # Check if model file exists
    if not model_info.get("model_file"):
        raise HTTPException(status_code=404, detail=f"Model file for model with ID {model_id} not found")
    
    # Return model file
    return FileResponse(
        model_info["model_file"],
        filename=f"{model_id}.{model_info['format']}"
    )

@app.get("/")
async def root() -> Dict[str, Any]:
    """
    Root endpoint.
    
    Returns:
        Dictionary with server information
    """
    return {
        "name": "OpenSCAD MCP Server",
        "version": "1.0.0",
        "description": "MCP server for OpenSCAD",
        "tools": list(tool_registry.keys())
    }

# Run server
if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
