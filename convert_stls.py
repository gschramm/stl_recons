# %%
import numpy as np
import trimesh

from scipy.ndimage import label
from pathlib import Path


def voxelize_stl(
    stl_path: str, voxel_size: float, fill_interior: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Voxelizes the inner volume of an STL mesh.

    Parameters:
    - stl_path: Path to the STL file.
    - voxel_size: Size of each voxel (float).
    - fill_interior: Whether to fill the inside of the mesh.

    Returns:
    - voxels: 3D numpy array with 1s (inside) and 0s (outside).
    - origin: Origin of the voxel grid (numpy array).
    - spacing: Voxel spacing (same as voxel_size).
    """
    mesh = trimesh.load(stl_path)
    if not mesh.is_watertight:
        raise ValueError("Mesh must be watertight for interior voxelization")

    # Voxelize using trimesh
    v = mesh.voxelized(pitch=voxel_size)

    # Fill interior voxels if requested
    if fill_interior:
        v = v.fill()

    # Extract binary occupancy grid
    voxels = v.matrix.astype(np.uint8)

    return voxels


stl_files = sorted(list(Path("data").glob("*.stl")))
voxel_size = 0.1  # voxel size in mm

for stl_file in stl_files:
    print(stl_file)

    # Load and voxelize the STL file
    image = voxelize_stl(stl_file, voxel_size)

    image, num_lab = label(1 - image)
    print(num_lab)

    import pymirc.viewer as pv

    vi = pv.ThreeAxisViewer(image)

    _ = input("Press Enter to continue...")

    # Save the voxelized data
    output_path = Path("data") / f"{stl_file.stem}_voxelized_vs_{voxel_size}_mm.npz"
    np.savez_compressed(output_path, image=image, voxel_size=voxel_size)
    print(f"Voxelized {stl_file} and saved to {output_path}")
