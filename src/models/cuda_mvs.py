"""
CUDA Multi-View Stereo wrapper for 3D reconstruction from multiple images.
"""

import os
import subprocess
import logging
import json
import math
import shutil
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class CUDAMultiViewStereo:
    """
    Wrapper for CUDA Multi-View Stereo for 3D reconstruction from multiple images.
    """
    
    def __init__(self, cuda_mvs_path: str, output_dir: str = "output/models"):
        """
        Initialize the CUDA MVS wrapper.
        
        Args:
            cuda_mvs_path: Path to CUDA MVS installation
            output_dir: Directory to store output files
        """
        self.cuda_mvs_path = cuda_mvs_path
        self.output_dir = output_dir
        self.patch_match_mvs_executable = self._find_executable("app_patch_match_mvs")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate installation
        self._validate_installation()
    
    def _validate_installation(self) -> None:
        """
        Validate CUDA MVS installation.
        
        Raises:
            FileNotFoundError: If CUDA MVS installation is not found
        """
        if not os.path.exists(self.cuda_mvs_path):
            raise FileNotFoundError(f"CUDA MVS not found at {self.cuda_mvs_path}")
        
        # Check for required executables
        if not os.path.exists(self.patch_match_mvs_executable):
            raise FileNotFoundError(
                f"Required executable app_patch_match_mvs not found under {os.path.join(self.cuda_mvs_path, 'build')}"
            )

    def _find_executable(self, executable_name: str) -> str:
        for build_name in ("build-opencv-cuda", "build"):
            build_dir = os.path.join(self.cuda_mvs_path, build_name)
            for relative_path in (executable_name, os.path.join("samples", executable_name)):
                exec_path = os.path.join(build_dir, relative_path)
                if os.path.exists(exec_path):
                    return exec_path
        return os.path.join(self.cuda_mvs_path, "build", executable_name)
    
    def generate_model_from_images(self, image_paths: List[str], 
                                camera_params: Optional[Dict[str, Any]] = None,
                                output_name: str = "model") -> Dict[str, Any]:
        """
        Generate a 3D model from multiple images using CUDA MVS.
        
        Args:
            image_paths: List of paths to input images
            camera_params: Optional camera parameters
            output_name: Name for the output files
            
        Returns:
            Dictionary containing paths to generated model files
        """
        try:
            # Create a unique directory for this reconstruction
            model_dir = os.path.join(self.output_dir, output_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Create a camera parameters file if provided
            params_file = None
            if camera_params:
                params_file = os.path.join(model_dir, "camera_params.json")
                with open(params_file, 'w') as f:
                    json.dump(camera_params, f, indent=2)
             
            # Generate camera parameters if not provided
            if not params_file:
                params_file = self._generate_camera_params(image_paths, model_dir)

            mvs_input_dir = self._prepare_cuda_mvs_input(image_paths, model_dir)
            mvs_output_dir = os.path.join(model_dir, "cuda_mvs_output")
            os.makedirs(mvs_output_dir, exist_ok=True)
             
            # Generate point cloud
            point_cloud_file = os.path.join(model_dir, f"{output_name}.ply")
             
            # Run CUDA MVS
            cmd = [
                self.patch_match_mvs_executable,
                mvs_input_dir,
                f"--output-directory={mvs_output_dir}",
                "--max-image-size=800"
            ]
            
            logger.info(f"Running CUDA MVS with command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for process to complete
            stdout, stderr = process.communicate()

            with open(os.path.join(model_dir, "cuda_mvs_stdout.log"), "w") as f:
                f.write(stdout)
            with open(os.path.join(model_dir, "cuda_mvs_stderr.log"), "w") as f:
                f.write(stderr)
             
            if process.returncode != 0:
                logger.error(f"Error running CUDA MVS: {stderr}")
                if "No CUDA support" in stderr:
                    raise RuntimeError(
                        "CUDA MVS is linked against an OpenCV build without CUDA support. "
                        "Install or build OpenCV with CUDA/opencv_contrib support, then rebuild CUDA MVS against it."
                    )
                raise RuntimeError(f"CUDA MVS failed with exit code {process.returncode}: {stderr.strip()}")

            dense_point_cloud_file = os.path.join(mvs_output_dir, "point_cloud_dense.ply")
            if os.path.exists(dense_point_cloud_file):
                shutil.copyfile(dense_point_cloud_file, point_cloud_file)
             
            # Check if output file was created
            if not os.path.exists(point_cloud_file):
                raise FileNotFoundError(f"Output point cloud file not found at {point_cloud_file}")
            
            return {
                "model_id": output_name,
                "output_dir": model_dir,
                "point_cloud_file": point_cloud_file,
                "camera_params_file": params_file,
                "input_images": image_paths
            }
        
        except Exception as e:
            logger.error(f"Error generating 3D model with CUDA MVS: {str(e)}")
            raise
    
    def _generate_camera_params(self, image_paths: List[str], model_dir: str) -> str:
        """
        Generate camera parameters from images.
        
        Args:
            image_paths: List of paths to input images
            model_dir: Directory to save parameter file
            
        Returns:
            Path to camera parameters file
        """
        # This is a simplified version for demonstration
        # In a real implementation, this would use SfM or camera estimation
        
        params = []
        for i, img_path in enumerate(image_paths):
            # Extract image dimensions
            from PIL import Image
            with Image.open(img_path) as img:
                width, height = img.size
            
            # Generate simple camera parameters
            # In reality, these would be estimated from the images
            # or provided by the user
            params.append({
                "image_id": i,
                "image_name": os.path.basename(img_path),
                "width": width,
                "height": height,
                "camera": {
                    "model": "PINHOLE",
                    "focal_length": min(width, height),
                    "principal_point": [width / 2, height / 2],
                    "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    "translation": [0, 0, 0]
                }
            })
        
        # Write parameters to file
        params_file = os.path.join(model_dir, "camera_params.json")
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        return params_file

    def _opencv_matrix(self, rows: int, cols: int, dt: str, data: List[float]) -> Dict[str, Any]:
        return {
            "type_id": "opencv-matrix",
            "rows": rows,
            "cols": cols,
            "dt": dt,
            "data": data,
        }

    def _write_opencv_json(self, path: str, data: Dict[str, Any]) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _prepare_cuda_mvs_input(self, image_paths: List[str], model_dir: str) -> str:
        """
        Create the input files expected by the CUDA MVS sample executable.
        """
        from PIL import Image

        input_dir = os.path.join(model_dir, "cuda_mvs_input")
        image_dir = os.path.join(input_dir, "images")
        os.makedirs(image_dir, exist_ok=True)

        copied_images = []
        cameras = []
        rotations = []
        translations = []
        nimages = len(image_paths)

        for i, image_path in enumerate(image_paths):
            image_name = f"view_{i + 1}{Path(image_path).suffix or '.png'}"
            copied_image_path = os.path.join(image_dir, image_name)
            shutil.copyfile(image_path, copied_image_path)
            copied_images.append(copied_image_path)

            with Image.open(image_path) as img:
                width, height = img.size

            focal_length = float(max(width, height))
            cameras.append([focal_length, 0.0, width / 2.0, 0.0, focal_length, height / 2.0, 0.0, 0.0, 1.0])

            angle = (2.0 * math.pi * i) / max(nimages, 1)
            camera_radius = 2.5
            camera_height = 0.25
            camera_position = [camera_radius * math.sin(angle), camera_height, camera_radius * math.cos(angle)]
            rotations.append(self._look_at_rotation(camera_position, [0.0, 0.0, 0.0]))
            translations.append(camera_position)

        self._write_opencv_json(
            os.path.join(input_dir, "input_images.json"),
            {
                "num_images": nimages,
                "input_images": [
                    {
                        "id": i,
                        "filename": copied_images[i],
                        "K": self._opencv_matrix(3, 3, "d", cameras[i]),
                    }
                    for i in range(nimages)
                ],
            },
        )

        self._write_opencv_json(
            os.path.join(input_dir, "global_poses.json"),
            {
                "num_global_poses": nimages,
                "global_poses": [
                    {
                        "id": i,
                        "R": self._opencv_matrix(3, 3, "d", rotations[i]),
                        "t": self._opencv_matrix(1, 3, "d", translations[i]),
                    }
                    for i in range(nimages)
                ],
            },
        )

        self._write_opencv_json(
            os.path.join(input_dir, "view_id_sets.json"),
            {
                "num_reference_images": nimages,
                "view_id_sets": [
                    {
                        "id": i,
                        "view_id_set": self._opencv_matrix(
                            1,
                            nimages,
                            "i",
                            [i] + [j for j in range(nimages) if j != i],
                        ),
                    }
                    for i in range(nimages)
                ],
            },
        )

        self._write_sparse_point_cloud(os.path.join(input_dir, "point_cloud_sparse.ply"))
        return input_dir

    def _look_at_rotation(self, camera_position: List[float], target: List[float]) -> List[float]:
        def normalize(vector: List[float]) -> List[float]:
            length = math.sqrt(sum(value * value for value in vector))
            return [value / length for value in vector]

        def cross(a: List[float], b: List[float]) -> List[float]:
            return [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]

        forward = normalize([target[i] - camera_position[i] for i in range(3)])
        world_up = [0.0, 1.0, 0.0]
        right = normalize(cross(world_up, forward))
        up = cross(forward, right)

        return [
            right[0], up[0], forward[0],
            right[1], up[1], forward[1],
            right[2], up[2], forward[2],
        ]

    def _write_sparse_point_cloud(self, path: str) -> None:
        points = []
        grid_size = 9
        for y_index in range(grid_size):
            y = -0.5 + y_index / (grid_size - 1)
            for x_index in range(grid_size):
                x = -0.5 + x_index / (grid_size - 1)
                z = 0.0
                points.append((x, y, z, 200, 200, 200))

        with open(path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for point in points:
                f.write("{} {} {} {} {} {}\n".format(*point))
    
    def convert_ply_to_obj(self, ply_file: str, output_dir: Optional[str] = None) -> str:
        """
        Convert PLY point cloud to OBJ mesh.
        
        Args:
            ply_file: Path to input PLY file
            output_dir: Directory to save output OBJ file
            
        Returns:
            Path to output OBJ file
        """
        if not output_dir:
            output_dir = os.path.dirname(ply_file)
        
        # Generate output file path
        obj_file = os.path.join(output_dir, f"{Path(ply_file).stem}.obj")
        
        logger.info(f"Converting PLY to OBJ: {ply_file} -> {obj_file}")

        mesh = self._mesh_from_point_cloud(ply_file)

        try:
            import open3d as o3d

            if not o3d.io.write_triangle_mesh(obj_file, mesh, write_vertex_normals=True):
                raise RuntimeError(f"Failed to write OBJ file: {obj_file}")
        except Exception as exc:
            logger.error("Failed to write OBJ mesh: %s", exc)
            raise
        
        return obj_file

    def convert_ply_to_stl(self, ply_file: str, output_dir: Optional[str] = None) -> str:
        """
        Convert PLY point cloud to an STL mesh that OpenSCAD can import.
        """
        if not output_dir:
            output_dir = os.path.dirname(ply_file)

        stl_file = os.path.join(output_dir, f"{Path(ply_file).stem}.stl")

        logger.info(f"Converting PLY to STL: {ply_file} -> {stl_file}")

        mesh = self._mesh_from_point_cloud(ply_file)

        try:
            import open3d as o3d

            if not o3d.io.write_triangle_mesh(stl_file, mesh, write_ascii=False):
                raise RuntimeError(f"Failed to write STL file: {stl_file}")
        except Exception as exc:
            logger.error("Failed to write STL mesh: %s", exc)
            raise

        return stl_file

    def _mesh_from_point_cloud(self, ply_file: str):
        """
        Build a watertight-ish mesh from a sparse reconstructed point cloud.
        """
        try:
            import open3d as o3d

            pcd = o3d.io.read_point_cloud(ply_file)
            if not pcd.has_points():
                raise ValueError(f"PLY point cloud contains no points: {ply_file}")

            points = list(pcd.points)
            if len(points) < 4:
                raise ValueError(f"At least 4 points are required to create a mesh, found {len(points)}")

            if len(points) >= 20:
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=min(20, len(points) - 1), std_ratio=2.5)

            mesh, _ = pcd.compute_convex_hull()
            mesh.compute_vertex_normals()
            if not mesh.has_triangles():
                raise ValueError(f"Could not create mesh triangles from point cloud: {ply_file}")

            return mesh
        except Exception as exc:
            logger.error("Failed to create mesh from PLY point cloud: %s", exc)
            raise
