import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

from utils.random_utils import set_random_seed
from visualization.common_custom_viz import get_batch, init
import datetime

import math

# ------------------------------------------------------------------------------------- #

SEED = 33
W_GUIDANCE = 2.0
W_FIRST_FRAME_GUIDANCE = 2.0
USE_FIRST_FRAME_COND = 1
SAMPLE_ID = "shot_000285"
PROMPT = "The camera steadily booms up from a close shot starting on the left, guiding the viewer's focus. Positioned at middle left on the screen, the frame maintains stability throughout the movement."
OUTPUT_DIR = "./gen_traj"
CUSTOM_FIRST_FRAME = None

# ------------------------------------------------------------------------------------- #

def save_trajectory_to_txt(traj, fovs, filename):
    """
    Save camera trajectory matrices to a text file.
    Each line format: r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz fov
    With tab separation between values.
    """
    with open(filename, "w") as f:
        for i, pose_matrix in enumerate(traj):
            # Extract rotation and translation components
            r00, r01, r02 = pose_matrix[0, 0:3]
            r10, r11, r12 = pose_matrix[1, 0:3]
            r20, r21, r22 = pose_matrix[2, 0:3]
            tx, ty, tz = pose_matrix[0:3, 3]
            fov_value = fovs[i] if i < len(fovs) else 0.0
            
            # Format as flat array with tabs between values
            flat_format = f"{r00:.6f}\t{r01:.6f}\t{r02:.6f}\t{tx:.6f}\t"
            flat_format += f"{r10:.6f}\t{r11:.6f}\t{r12:.6f}\t{ty:.6f}\t"
            flat_format += f"{r20:.6f}\t{r21:.6f}\t{r22:.6f}\t{tz:.6f}\t{fov_value:.6f}\n"
            
            f.write(flat_format)

def create_frustum(pose_matrix, fov, scale=0.5):
    """Create vertices for a camera frustum in 3D space from a pose matrix."""
    # Convert FOV to radians
    fov_rad = math.radians(fov)
    
    # Calculate frustum dimensions
    far = scale
    height = 2 * far * math.tan(fov_rad / 2)
    width = height * (16/9)  # 16:9 aspect ratio
    
    # Define frustum vertices in camera space
    vertices_cam = np.array([
        [0, 0, 0],                    # Camera position
        [width/2, height/2, -far],    # Top right
        [-width/2, height/2, -far],   # Top left
        [-width/2, -height/2, -far],  # Bottom left
        [width/2, -height/2, -far]    # Bottom right
    ])
    
    # Extract rotation matrix and position from pose matrix
    rotation_matrix = pose_matrix[:3, :3]
    position = pose_matrix[:3, 3]
    
    # Transform vertices to world space
    vertices_world = np.dot(vertices_cam, rotation_matrix.T) + position
    
    return vertices_world

def visualize_camera_trajectory(trajectory, output_path, fov=45):
    """Create a 3D plot of camera trajectory with frustums and rainbow colors using Unity's coordinate system."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract camera positions (translation part of the trajectory)
    positions = np.array([mat[:3, 3] for mat in trajectory])
    
    # Create rainbow colormap for the trajectory
    colors = plt.cm.rainbow(np.linspace(0, 1, len(trajectory)))
    
    # Plot camera path with rainbow colors - map to Unity coordinate system (Y-up, left-handed)
    # Original: X,Y,Z -> Unity: X,Z,Y (with Z negated for left-handedness)
    for i in range(len(trajectory)-1):
        ax.plot(positions[i:i+2, 0],                  # X stays as X
                positions[i:i+2, 2],                  # Z becomes Y (up in Unity) 
                -positions[i:i+2, 1],                 # -Y becomes Z (forward in Unity)
                color=colors[i], linewidth=2)
    
    # Mark start and end points in Unity coordinate system
    ax.scatter(positions[0, 0], positions[0, 2], -positions[0, 1], c='g', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 2], -positions[-1, 1], c='r', s=100, label='End')
    
    # Draw camera frustums at intervals
    interval = max(1, len(trajectory) // 10)
    for i in range(0, len(trajectory), interval):
        frustum_vertices = create_frustum(trajectory[i], fov, 0.1)
        
        # Draw frustum lines (from camera center to frustum corners) in Unity coordinate system
        for j in range(1, 5):
            ax.plot([frustum_vertices[0, 0], frustum_vertices[j, 0]],
                    [frustum_vertices[0, 2], frustum_vertices[j, 2]],
                    [-frustum_vertices[0, 1], -frustum_vertices[j, 1]],
                    color=colors[i], alpha=0.6)
        
        # Draw frustum face
        face_indices = [1, 2, 3, 4, 1]  # Close the loop
        ax.plot(frustum_vertices[face_indices, 0],
                frustum_vertices[face_indices, 2],
                -frustum_vertices[face_indices, 1],
                color=colors[i], alpha=0.6)
    
    # Add rainbow colorbar legend
    sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1, aspect=10, location='right')
    cbar.set_label('Trajectory Progress')
    
    # Update axis labels to match Unity coordinate system
    ax.set_xlabel('X')
    ax.set_ylabel('Z (Forward)')
    ax.set_zlabel('Y (Up)') 
    ax.set_title('Camera Trajectory with Frustums (Unity Coordinate System)')
    
    # Make the plot balanced using the remapped coordinates
    unity_positions = np.column_stack([
        positions[:, 0],       # X stays X
        positions[:, 2],       # Z becomes Y (up)
        -positions[:, 1]       # -Y becomes Z (forward)
    ])
    
    max_range = np.array([
        unity_positions[:, 0].max() - unity_positions[:, 0].min(),
        unity_positions[:, 1].max() - unity_positions[:, 1].min(),
        unity_positions[:, 2].max() - unity_positions[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (unity_positions[:, 0].max() + unity_positions[:, 0].min()) / 2
    mid_y = (unity_positions[:, 1].max() + unity_positions[:, 1].min()) / 2
    mid_z = (unity_positions[:, 2].max() + unity_positions[:, 2].min()) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Adjust the legend position to bottom right
    ax.legend(loc='lower right')
    
    # Set a better viewing angle for Y-up coordinate system
    ax.view_init(elev=30, azim=-135)
    
    plt.savefig(output_path, dpi=300)
    plt.close()

# ------------------------------------------------------------------------------------- #


if __name__ == "__main__":
    diffuser, clip_model, dataset, device = init("config_custom_viz")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----------------------------------------------------------------------------- #
    # Arguments
    SEED = int(input(f"Seed (default={SEED}): ") or SEED)
    W_GUIDANCE = float(input(f"Guidance (default={W_GUIDANCE}): ") or W_GUIDANCE)
    USE_FIRST_FRAME_COND = int(input(f"Use First Frame Condition (0 or 1, default=1): ") or USE_FIRST_FRAME_COND)
    
    if USE_FIRST_FRAME_COND:
        W_FIRST_FRAME_GUIDANCE = float(input(f"First Frame Guidance (default={W_FIRST_FRAME_GUIDANCE}): ") or W_FIRST_FRAME_GUIDANCE)
        
        print("Enter the first frame pose as 13 values (rotation matrix + translation + fov), Format: r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz fov")
        frame_values = input("> ")
        try:
            values = [float(x) for x in frame_values.split()]
            if len(values) == 13:
                CUSTOM_FIRST_FRAME = values
        except:
            print("Error parsing values. Using reference first frame.")
    SAMPLE_ID = input(f"Sample ID (default={SAMPLE_ID}): ") or SAMPLE_ID
    PROMPT = input(f"Prompt (default='{PROMPT}'): ") or PROMPT
    # ----------------------------------------------------------------------------- #

    # Create sample-specific output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    sample_dir = os.path.join(OUTPUT_DIR, f"lenscript_seed{SEED}_{timestamp}")
    os.makedirs(sample_dir, exist_ok=True)

    # Set arguments
    set_random_seed(SEED)
    diffuser.gen_seeds = np.array([SEED])
    diffuser.guidance_weight = W_GUIDANCE
    diffuser.use_first_frame_condition = USE_FIRST_FRAME_COND
    diffuser.first_frame_guidance_weight = W_FIRST_FRAME_GUIDANCE

    # Inference
    seq_feat = diffuser.net.model.clip_sequential
    batch = get_batch(PROMPT, SAMPLE_ID, clip_model, dataset, seq_feat, device, custom_first_frame=CUSTOM_FIRST_FRAME)
    with torch.no_grad():
        out = diffuser.predict_step(batch)

    # Process outputs - focusing only on camera trajectory
    padding_mask = out["padding_mask"][0].to(bool).cpu()
    padded_traj = out["gen_samples"][0].cpu()
    
    traj = padded_traj[padding_mask].numpy()  # Convert to numpy for visualization
    fovs = out["gen_fovs"][0].cpu().numpy()
    caption = out["caption_raw"][0]

    print(f"\nProcessing sample ID: lenscript with seed: {SEED}")
    
    # Save camera trajectory
    traj_file = os.path.join(sample_dir, "camera_trajectory.txt")
    save_trajectory_to_txt(traj, fovs, traj_file)
    
    # Save camera metadata
    meta_file = os.path.join(sample_dir, "camera_info.txt")
    with open(meta_file, "w") as f:
        f.write(f"# Camera Information\n")
        f.write(f"Prompt: {PROMPT}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Guidance Weight: {W_GUIDANCE}\n")
        f.write(f"First Frame Condition: {USE_FIRST_FRAME_COND}\n")
        if USE_FIRST_FRAME_COND:
            f.write(f"First Frame Guidance Weight: {W_FIRST_FRAME_GUIDANCE}\n")
            if CUSTOM_FIRST_FRAME is not None:
                f.write(f"Using custom first frame pose\n")
    
    print(f"Camera information saved to {meta_file}")
    
    # Generate visualization
    viz_file = os.path.join(sample_dir, "trajectory_visualization.png")
    visualize_camera_trajectory(traj, viz_file)
    print(f"Visualization saved to {viz_file}")
    print("\nProcessing complete!")