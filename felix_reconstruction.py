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
import os
import json
import random

root = 'dataset/small'

for folder_name in os.listdir(root):
    subfolder = os.path.join(root, folder_name)
    json_path = os.path.join(subfolder, 'batches_results_bootstrapped.json')
    output_path = os.path.join(subfolder, 'perturbed_all_lines.json')

    output_lines = []

    if not os.path.isfile(json_path):
        continue

    with open(json_path) as f:
        data = json.load(f)

        for entry in data:
            # 1. Fixed strokes → feature lines with higher opacity
            for stroke in entry.get("fixed_strokes", []):
                if len(stroke) >= 2:
                    output_lines.append({
                        "type": "feature_line",
                        "feature_id": 0,
                        "geometry": stroke,
                        "opacity": random.uniform(0.2, 0.4)
                    })

            # 2. Final proxies → construction lines with lower opacity
            for stroke in entry.get("final_proxies", []):
                if len(stroke) >= 2:
                    output_lines.append({
                        "type": "construction_line",
                        "feature_id": 0,
                        "geometry": stroke,
                        "opacity": random.uniform(0.05, 0.1)
                    })

    # vis_feature_lines(output_lines)
    if output_lines:
        with open(output_path, 'w') as out_f:
            json.dump(output_lines, out_f, indent=2)
        print(f"Saved {len(output_lines)} lines with opacity to {output_path}")
