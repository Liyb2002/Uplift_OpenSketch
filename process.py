
import json
import numpy as np
import open3d as o3d
import os

# ---- Paths ----
folder = "dataset/vacuum"
stroke_path = os.path.join(folder, "view1_concept.json")
cam_param_path = os.path.join(folder, "Professional1_vacuum_cleaner_view1_camparam.json")
correspond_path = os.path.join(folder, "professional1_vacuum_cleaner_v1_points.json")
obj_path = os.path.join(folder, "vacuum_cleaner.obj")


# ---- Load Stroke JSON (Fixed) ----
with open(stroke_path, 'r') as f:
    sketch_data = json.load(f)  # Make sure this returns a dict, not a string

canvas_width = sketch_data["canvas"]["width"]
canvas_height = sketch_data["canvas"]["height"]

all_strokes_2d = []

for stroke in sketch_data["strokes"]:
    if isinstance(stroke, str):
        stroke = json.loads(stroke)  # If strokes are embedded as strings, parse them

    if stroke.get("is_removed", False):
        continue

    if isinstance(stroke["points"], str):
        stroke["points"] = json.loads(stroke["points"])  # Defensive parsing if needed

    points = [(pt["x"], pt["y"]) for pt in stroke["points"] if isinstance(pt, dict)]
    if points:
        all_strokes_2d.append(np.array(points))




# --- Load camera JSON ---
with open(cam_param_path, 'r') as f:
    cam_data = json.load(f)

# Get the restricted model
cam = cam_data["restricted"]
mv_matrix = np.array(cam["mvMatrix"])  # 4x4 model-view matrix

# --- Build Intrinsic matrix K ---
f = cam["f"]
u = cam["u"]
v = cam["v"]
skew = cam["skew"]  # should be 0
width = cam_data["width"]
height = width  # Assumed square canvas

# Convert normalized principal point (u,v) to pixels
cx = (u + 1) * width / 2
cy = (v + 1) * height / 2

K = np.array([
    [f, skew, cx],
    [0,  f,   cy],
    [0,  0,   1]
])

# --- Extract [R | t] from mvMatrix ---
# mvMatrix is model-view, i.e., transforms world → camera space.
# To get [R|t], we take the upper 3x4 part:
RT = np.array(mv_matrix)[:3, :]  # shape (3,4)

# Final Projection Matrix
P = K @ RT  # shape (3,4)




# --- Load Correspondences ---
with open(correspond_path, 'r') as f:
    corr = json.load(f)

points_2d = np.array(corr["points_2D_sketch"])   # shape (N, 2)
points_3d = np.array(corr["points_3D_object"])   # shape (N, 3)

# --- Optional: normalize 2D points if canvas != 691x691
# You can skip if already in pixel space matching camera param width

# --- Fit plane to 3D points ---
def fit_plane(pts):
    A = np.c_[pts[:, 0], pts[:, 1], np.ones(pts.shape[0])]
    b = -pts[:, 2]
    coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b, d = coef
    c = 1.0
    return np.array([a, b, c, d])

plane_eq = fit_plane(points_3d)  # a, b, c, d in ax + by + cz + d = 0



# ---- Backproject 2D Stroke Points to 3D ----
def backproject_to_plane(xy, P, plane):
    x, y = xy
    pt_2d_h = np.array([x, y, 1.0])

    # Decompose camera projection matrix: P = K [R | t]
    M = P[:, :3]
    c = -np.linalg.inv(M) @ P[:, 3]  # Camera center in world coords
    ray_dir = np.linalg.pinv(M) @ pt_2d_h
    ray_dir /= np.linalg.norm(ray_dir)

    # Ray: X = c + t * ray_dir
    # Plane: ax + by + cz + d = 0
    a, b, c_, d = plane
    num = -(a * c[0] + b * c[1] + c_ * c[2] + d)
    denom = a * ray_dir[0] + b * ray_dir[1] + c_ * ray_dir[2]
    t = num / denom
    point_3d = c + t * ray_dir
    return point_3d

# Reconstruct strokes
all_strokes_3d = []
for stroke in all_strokes_2d:
    stroke_3d = [backproject_to_plane(pt, P, plane_eq) for pt in stroke]
    all_strokes_3d.append(np.array(stroke_3d))



# ---- Visualization ----
# Load the object mesh
mesh = o3d.io.read_triangle_mesh(obj_path)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.7, 0.7, 0.7])

# Convert strokes to 3D line segments
lines = []
points = []
offset = 0
for stroke in all_strokes_3d:
    if len(stroke) < 2:
        continue
    points.extend(stroke)
    lines.extend([[i + offset, i + 1 + offset] for i in range(len(stroke) - 1)])
    offset += len(stroke)

line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines)
)
line_set.paint_uniform_color([1, 0, 0])  # Red strokes

print("points", points)
# Visualize
# o3d.visualization.draw_geometries([mesh, line_set])
o3d.visualization.draw_geometries([line_set])
