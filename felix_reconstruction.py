import os
import json
import matplotlib.pyplot as plt
import random

def vis_feature_lines(feature_lines):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_frame_on(False)
    ax.grid(False)
    ax.set_axis_off()

    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    for stroke in feature_lines:
        geometry = stroke.get("geometry", [])
        alpha = stroke.get("opacity", 0.5)
        linewidth = 0.6

        for j in range(1, len(geometry)):
            start = geometry[j - 1]
            end = geometry[j]

            x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
            y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
            z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])

            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                    color='black', linewidth=linewidth, alpha=alpha)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)

    ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
    ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
    ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

    plt.show()

# --- Main loading and visualizing ---
root = 'dataset/reconstructions'

for folder_name in os.listdir(root):
    subfolder = os.path.join(root, folder_name)
    json_path = os.path.join(subfolder, 'batches_results_bootstrapped.json')

    feature_lines = []
    if os.path.isfile(json_path):
        with open(json_path) as f:
            data = json.load(f)
            for entry in data:
                proxies = entry.get("final_proxies", [])
                final_corrs = entry.get("final_correspondences", [])

                # Collect all stroke IDs that appear in correspondences
                corr_ids = set()
                for c in final_corrs:
                    corr_ids.add(c.get("stroke_id_0"))
                    corr_ids.add(c.get("stroke_id_1"))

                for i, stroke in enumerate(proxies):
                    if len(stroke) >= 2:
                        opacity = random.uniform(0.6, 0.9) if i in corr_ids else random.uniform(0.2, 0.4)
                        feature_lines.append({"geometry": stroke, "opacity": opacity})

    if len(feature_lines) > 200 or len(feature_lines) < 100:
        continue

    print("folder_name", folder_name)
    vis_feature_lines(feature_lines)
