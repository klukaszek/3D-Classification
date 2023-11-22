from turtle import distance
from open3d import io, utility, geometry
import numpy as np

def convert_toy4k_npz(path):
    """
    Convert a .npz file containing a point cloud and normals to a .ply file
    """

    # Load npz file containing point cloud and normals
    data = np.load(path, allow_pickle=True)
    data = data['dct']
    data = data.item()

    # Convert point cloud and normals to list
    pc = data['pc']
    pc = pc.tolist()
    normal = data['normals']
    normal = normal.tolist()

    data = []

    # Combine point cloud and normals into one list [[x, y, z, nx, ny, nz], ...]
    for i, points in enumerate(pc):
        data.append(points) 

    for i, normals in enumerate(normal):
        data[i].extend(normals)

    # Convert list to numpy array
    data = np.array(data)

    # Create open3d point cloud
    pcd = geometry.PointCloud()

    # Read first 3 columns of numpy array as point cloud
    pcd.points = utility.Vector3dVector(data[:, 0:3])

    # Read last 3 columns of numpy array as normals
    pcd.normals = utility.Vector3dVector(data[:, 3:6])

    # Generate mesh from point cloud using Ball Pivoting Algorithm

    # Estimate radius for rolling ball
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1 * avg_dist

    # Run ball pivoting algorithm
    bpa_mesh = geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, utility.DoubleVector([radius, radius * 2]))

    # Create path to save .obj file [remove .npz and add .stl]
    path = path.replace('.npz', '.stl')

    # Rotate mesh to align

    # Create rotation matrix
    R = bpa_mesh.get_rotation_matrix_from_xyz((-np.pi/2, -np.pi/4, np.pi))

    # Rotate mesh
    bpa_mesh.rotate(R, center=(0, 0, 0))

    # Save mesh as .obj file
    io.write_triangle_mesh(path, bpa_mesh, write_vertex_normals=True)

    return path