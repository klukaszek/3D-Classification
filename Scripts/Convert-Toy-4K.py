import open3d as o3d
import numpy as np

# Example

# Load npz file containing point cloud and normals
data = np.load('/mnt/c/Classes/CIS4780/3D-Classification/Data/toys4k_point_clouds/chicken/chicken_000/pc10K.npz', allow_pickle=True)
data = data['dct']
data = data.item()

# Convert point cloud and normals to list
pc = data['pc']
pc = pc.tolist()
normal = data['normals']
normal = normal.tolist()

data = []

# Combine point cloud and normals into one list [[x, y, z, nx, ny, nz], ...]
for i in enumerate(pc):
    l = []
    l.extend(pc[i])
    l.extend(normal[i])
    data.append(l)

# Convert list to numpy array
data = np.array(data)

# Create open3d point cloud
pcd = o3d.geometry.PointCloud()

# Read first 3 columns of numpy array as point cloud
pcd.points = o3d.utility.Vector3dVector(data[:, 0:3])

# Read last 3 columns of numpy array as normals
pcd.normals = o3d.utility.Vector3dVector(data[:, 3:6])

# Paint point cloud black (Or use color if provided)
pcd.paint_uniform_color([0, 0, 0])

# Visualize point cloud to make sure it looks correct
o3d.visualization.draw_geometries([pcd])

## Save data as xyzn for use
# with open('chicken.xyzn', 'w') as f:
#     for item in data:
#         for vertex in item:
#            f.write("%s, " % vertex)
#         f.write("\n" % item)

## Save point cloud as ply file (Or use xyzn if you want)
# o3d.io.write_point_cloud("chicken.ply", pcd)