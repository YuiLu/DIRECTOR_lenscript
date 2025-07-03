from itertools import product
from typing import List, Tuple

from evo.core import lie_algebra as lie
import numpy as np
import torch
from scipy.stats import mode
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
import torchmetrics.functional as F
from torchtyping import TensorType

# ------------------------------------------------------------------------------------- #

num_samples, num_cams, num_total_cams, num_classes = None, None, None, None
width, height = None, None

# ------------------------------------------------------------------------------------- #

CAM_INDEX_TO_PATTERN = {
    0: "boom up",
    1: "boom down",
    2: "rotate",
    3: "truck",
    4: "push in",
    5: "pull out",
    6: "zoom in",
    7: "zoom out",
    8: "dolly zoom in",
    9: "dolly zoom out",
    10: "pan",
    11: "tilt",
    12: "static",
}

# ------------------------------------------------------------------------------------- #


def to_euler_angles(
    rotation_mat: TensorType["num_samples", 3, 3]
) -> TensorType["num_samples", 3]:
    rotation_vec = torch.from_numpy(
        np.stack(
            [lie.sst_rotation_from_matrix(r).as_rotvec() for r in rotation_mat.numpy()]
        )
    )
    return rotation_vec


def compute_relative(f_t: TensorType["num_samples", 3]):
    max_value = np.max(np.stack([abs(f_t[:, 0]), abs(f_t[:, 1])]), axis=0)
    xy_f_t = np.divide(
        (abs(f_t[:, 0]) - abs(f_t[:, 1])),
        max_value,
        out=np.zeros_like(max_value),
        where=max_value != 0,
    )
    max_value = np.max(np.stack([abs(f_t[:, 0]), abs(f_t[:, 2])]), axis=0)
    xz_f_t = np.divide(
        abs(f_t[:, 0]) - abs(f_t[:, 2]),
        max_value,
        out=np.zeros_like(max_value),
        where=max_value != 0,
    )
    max_value = np.max(np.stack([abs(f_t[:, 1]), abs(f_t[:, 2])]), axis=0)
    yz_f_t = np.divide(
        abs(f_t[:, 1]) - abs(f_t[:, 2]),
        max_value,
        out=np.zeros_like(max_value),
        where=max_value != 0,
    )
    return xy_f_t, xz_f_t, yz_f_t


def compute_camera_dynamics(w2c_poses: TensorType["num_samples", 4, 4], fovs: TensorType["num_samples"] = None, fps: float = 60.0):
    w2c_poses_inv = torch.from_numpy(
        np.array([lie.se3_inverse(t) for t in w2c_poses.numpy()])
    )
    velocities = w2c_poses_inv[:-1].to(float) @ w2c_poses[1:].to(float)

    # --------------------------------------------------------------------------------- #
    # Translation velocity
    t_velocities = fps * velocities[:, :3, 3]
    t_xy_velocity, t_xz_velocity, t_yz_velocity = compute_relative(t_velocities)
    t_vels = (t_velocities, t_xy_velocity, t_xz_velocity, t_yz_velocity)
    # --------------------------------------------------------------------------------- #
    # # Rotation velocity
    a_velocities = to_euler_angles(velocities[:, :3, :3])
    a_xy_velocity, a_xz_velocity, a_yz_velocity = compute_relative(a_velocities)
    a_vels = (a_velocities, a_xy_velocity, a_xz_velocity, a_yz_velocity)

    # FOV velocity if available
    fov_vels = None
    if fovs is not None:
        fov_velocities = fovs[1:] - fovs[:-1]
        fov_vels = (fov_velocities,)
    
    return velocities, t_vels, a_vels, fov_vels


# ------------------------------------------------------------------------------------- #


def perform_segmentation(
    velocities: TensorType["num_samples-1", 3],
    xy_velocity: TensorType["num_samples-1", 3],
    xz_velocity: TensorType["num_samples-1", 3],
    yz_velocity: TensorType["num_samples-1", 3],
    static_threshold: float,
    diff_threshold: float,
) -> List[int]:
    segments = torch.zeros(velocities.shape[0])
    segment_patterns = [torch.tensor(x) for x in product([0, 1, -1], repeat=3)]
    pattern_to_index = {
        tuple(pattern.numpy()): index for index, pattern in enumerate(segment_patterns)
    }

    for sample_index, sample_velocity in enumerate(velocities):
        sample_pattern = abs(sample_velocity) > static_threshold

        # XY
        if (sample_pattern == torch.tensor([1, 1, 0])).all():
            if xy_velocity[sample_index] > diff_threshold:
                sample_pattern = torch.tensor([1, 0, 0])
            elif xy_velocity[sample_index] < -diff_threshold:
                sample_pattern = torch.tensor([0, 1, 0])

        # XZ
        elif (sample_pattern == torch.tensor([1, 0, 1])).all():
            if xz_velocity[sample_index] > diff_threshold:
                sample_pattern = torch.tensor([1, 0, 0])
            elif xz_velocity[sample_index] < -diff_threshold:
                sample_pattern = torch.tensor([0, 0, 1])

        # YZ
        elif (sample_pattern == torch.tensor([0, 1, 1])).all():
            if yz_velocity[sample_index] > diff_threshold:
                sample_pattern = torch.tensor([0, 1, 0])
            elif yz_velocity[sample_index] < -diff_threshold:
                sample_pattern = torch.tensor([0, 0, 1])

        # XYZ
        elif (sample_pattern == torch.tensor([1, 1, 1])).all():
            if xy_velocity[sample_index] > diff_threshold:
                sample_pattern[1] = 0
            elif xy_velocity[sample_index] < -diff_threshold:
                sample_pattern[0] = 0

            if xz_velocity[sample_index] > diff_threshold:
                sample_pattern[2] = 0
            elif xz_velocity[sample_index] < -diff_threshold:
                sample_pattern[0] = 0

            if yz_velocity[sample_index] > diff_threshold:
                sample_pattern[2] = 0
            elif yz_velocity[sample_index] < -diff_threshold:
                sample_pattern[1] = 0

        sample_pattern = torch.sign(sample_velocity) * sample_pattern
        segments[sample_index] = pattern_to_index[tuple(sample_pattern.numpy())]

    return np.array(segments, dtype=int)


def smooth_segments(arr: List[int], window_size: int) -> List[int]:
    smoothed_arr = arr.copy()

    if len(arr) < window_size:
        return smoothed_arr

    half_window = window_size // 2
    # Handle the first half_window elements
    for i in range(half_window):
        window = arr[: i + half_window + 1]
        most_frequent = mode(window, keepdims=False).mode
        smoothed_arr[i] = most_frequent

    for i in range(half_window, len(arr) - half_window):
        window = arr[i - half_window : i + half_window + 1]
        most_frequent = mode(window, keepdims=False).mode
        smoothed_arr[i] = most_frequent

    # Handle the last half_window elements
    for i in range(len(arr) - half_window, len(arr)):
        window = arr[i - half_window :]
        most_frequent = mode(window, keepdims=False).mode
        smoothed_arr[i] = most_frequent

    return smoothed_arr


def remove_short_chunks(arr: List[int], min_chunk_size: int) -> List[int]:
    def remove_chunk(chunks):
        if len(chunks) == 1:
            return False, chunks

        chunk_lenghts = [(end - start) + 1 for _, start, end in chunks]
        chunk_index = np.argmin(chunk_lenghts)
        chunk_length = chunk_lenghts[chunk_index]
        if chunk_length < min_chunk_size:
            _, start, end = chunks[chunk_index]

            # Check if the chunk is at the beginning
            if chunk_index == 0:
                segment_r, start_r, end_r = chunks[chunk_index + 1]
                chunks[chunk_index + 1] = (segment_r, start_r - chunk_length, end_r)

            elif chunk_index == len(chunks) - 1:
                segment_l, start_l, end_l = chunks[chunk_index - 1]
                chunks[chunk_index - 1] = (segment_l, start_l, end_l + chunk_length)

            else:
                if chunk_length % 2 == 0:
                    half_length_l = chunk_length // 2
                    half_length_r = chunk_length // 2
                else:
                    half_length_l = (chunk_length // 2) + 1
                    half_length_r = chunk_length // 2

                segment_l, start_l, end_l = chunks[chunk_index - 1]
                segment_r, start_r, end_r = chunks[chunk_index + 1]
                chunks[chunk_index - 1] = (segment_l, start_l, end_l + half_length_l)
                chunks[chunk_index + 1] = (segment_r, start_r - half_length_r, end_r)

            chunks.pop(chunk_index)

        return chunk_length < min_chunk_size, chunks

    chunks = find_consecutive_chunks(arr)
    keep_removing, chunks = remove_chunk(chunks)
    while keep_removing:
        keep_removing, chunks = remove_chunk(chunks)

    merged_chunks = []
    for segment, start, end in chunks:
        merged_chunks.extend([segment] * ((end - start) + 1))

    return merged_chunks


# ------------------------------------------------------------------------------------- #


def find_consecutive_chunks(arr: List[int]) -> List[Tuple[int, int, int]]:
    chunks = []
    start_index = 0
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            end_index = i - 1
            if end_index >= start_index:
                chunks.append((arr[start_index], start_index, end_index))
            start_index = i

    # Add the last chunk if the array ends with consecutive similar digits
    if start_index < len(arr):
        chunks.append((arr[start_index], start_index, len(arr) - 1))

    return chunks


# ------------------------------------------------------------------------------------- #


class CaptionMetrics(Metric):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.metric_kwargs = dict(
            task="multiclass",
            num_classes=num_classes,
            average="weighted",
            zero_division=0,
        )

        self.fps = 25.0
        self.cam_static_threshold = 0.02
        self.cam_diff_threshold = 0.4
        self.smoothing_window_size = 56
        self.min_chunk_size = 25

        self.add_state("pred_segments", default=[], dist_reduce_fx="cat")
        self.add_state("target_segments", default=[], dist_reduce_fx="cat")

    def segment_camera_trajectories(
        self,
        w2c_poses: TensorType["num_samples", 4, 4],
        fovs: TensorType["num_samples"] = None,
    ) -> TensorType["num_samples"]:
        device = w2c_poses.device
        
        # Get camera dynamics
        _, t_vels, a_vels, fov_vels = compute_camera_dynamics(
            w2c_poses.cpu(),
            fovs=fovs.cpu() if fovs is not None else None,
            fps=self.fps
        )
        cam_velocities, cam_xy_velocity, cam_xz_velocity, cam_yz_velocity = t_vels
        
        # Get rotation velocities
        rot_velocities, rot_xy_velocity, rot_xz_velocity, rot_yz_velocity = a_vels
        
        # Initialize segments with default value (static)
        cam_segments = np.full(cam_velocities.shape[0], 12)  # 12 is static
        
        # Get look-at directions (Z-axis of camera) from poses
        # Camera Z-axis is the third column of rotation matrix
        look_dirs = w2c_poses[:-1, :3, 2].cpu().numpy()
        
        # Check movement magnitude
        movement_magnitude = torch.norm(cam_velocities, dim=1).numpy()
        rotation_magnitude = torch.norm(rot_velocities, dim=1).numpy()
        
        # FOV changes if available
        fov_changes = None
        if fov_vels is not None:
            fov_changes = fov_vels[0].cpu().numpy()
        
        # Process each frame
        for i in range(len(cam_segments)):
            # Is camera significantly moving?
            is_moving = movement_magnitude[i] > self.cam_static_threshold
            is_rotating = rotation_magnitude[i] > 0.01  # Rotation threshold
            
            # Check FOV changes
            has_zoom_in = False
            has_zoom_out = False
            if fov_changes is not None:
                has_zoom_in = fov_changes[i] < -0.5  # FOV decreasing = zoom in
                has_zoom_out = fov_changes[i] > 0.5   # FOV increasing = zoom out
            
            if not is_moving and not is_rotating:
                # Camera is static
                cam_segments[i] = 12  # static
                
            elif not is_moving and is_rotating:
                # Static position but rotating - pan or tilt
                # Check if rotation is more horizontal or vertical
                if abs(rot_velocities[i, 1]) > abs(rot_velocities[i, 0]):
                    cam_segments[i] = 10  # pan (horizontal rotation)
                else:
                    cam_segments[i] = 11  # tilt (vertical rotation)
                    
            elif is_moving:
                # Camera is moving
                movement = cam_velocities[i].numpy()
                look_dir = look_dirs[i]
                
                # Calculate how aligned movement is with look direction
                # using dot product normalized by magnitudes
                alignment = np.dot(movement, look_dir) / (np.linalg.norm(movement) * np.linalg.norm(look_dir))
                
                # Check vertical movement
                if abs(movement[1]) > max(abs(movement[0]), abs(movement[2])) * 1.5:
                    # Vertical movement dominates
                    if movement[1] > 0:
                        cam_segments[i] = 0  # boom up
                    else:
                        cam_segments[i] = 1  # boom down
                        
                # Check if movement is along look-at direction (push/pull) or perpendicular (truck)
                elif abs(alignment) > 0.7:  # Movement aligned with look direction
                    if alignment > 0:
                        cam_segments[i] = 5  # pull out (moving away from target)
                    else:
                        cam_segments[i] = 4  # push in (moving toward target)
                elif abs(alignment) < 0.3:  # Movement perpendicular to look direction
                    cam_segments[i] = 3  # truck
                else:
                    # If we have significant rotation, classify as rotate
                    if is_rotating and rotation_magnitude[i] > 0.05:
                        cam_segments[i] = 2  # rotate
                    else:
                        # If alignment is between thresholds and no significant rotation,
                        # default to truck as it's more common
                        cam_segments[i] = 3  # truck
            
            # Handle zoom effects - these override previous classifications
            if has_zoom_in:
                if cam_segments[i] == 4:  # If previously classified as push in
                    cam_segments[i] = 8  # Change to dolly zoom in
                else:
                    cam_segments[i] = 6  # Standard zoom in
            elif has_zoom_out:
                if cam_segments[i] == 5:  # If previously classified as pull out
                    cam_segments[i] = 9  # Change to dolly zoom out
                else:
                    cam_segments[i] = 7  # Standard zoom out
        
        # Apply smoothing and remove short segments
        cam_segments = smooth_segments(cam_segments, self.smoothing_window_size)
        cam_segments = remove_short_chunks(cam_segments, self.min_chunk_size)
        
        return torch.tensor(cam_segments, device=device)

    # --------------------------------------------------------------------------------- #

    def update(
        self,
        trajectories: TensorType["num_samples", "num_cams", 4, 4],
        raw_labels: TensorType["num_samples", "num_classes*num_total_cams"],
        mask: TensorType["num_samples", "num_cams"],
        fovs: TensorType["num_samples", "num_cams"] = None,
    ) -> Tuple[float, float, float]:
        """Update the state with extracted features."""
        for sample_index in range(trajectories.shape[0]):
            trajectory = trajectories[sample_index][mask[sample_index].to(bool)]
            labels = raw_labels[sample_index][mask[sample_index].to(bool)][:-1]
            if trajectory.shape[0] < 2:
                continue

            sample_fovs = None
            if fovs is not None:
                sample_fovs = fovs[sample_index][mask[sample_index].to(bool)]

            self.pred_segments.append(self.segment_camera_trajectories(trajectory, sample_fovs))
            self.target_segments.append(labels)

    def compute(self) -> Tuple[float, float, float]:
        """ """
        target_segments = dim_zero_cat(self.target_segments)
        pred_segments = dim_zero_cat(self.pred_segments)

        precision = F.precision(pred_segments, target_segments, **self.metric_kwargs)
        recall = F.recall(pred_segments, target_segments, **self.metric_kwargs)
        fscore = F.f1_score(pred_segments, target_segments, **self.metric_kwargs)

        return precision, recall, fscore
