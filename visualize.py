import glob
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from sam3.visualization_utils import COLORS, load_frame


def _require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found. Install it with: sudo apt install ffmpeg")
    return exe


def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
        coords: List of coordinates
        coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")


def _overlay_masks_on_frame(
    frame: np.ndarray, frame_output: dict, alpha: float = 0.5
) -> np.ndarray:
    """Composite SAM3 segmentation masks onto a BGR frame.

    Expects the raw SAM3 output dict with keys 'out_obj_ids' and 'out_binary_masks'.
    """
    out = frame.copy().astype(np.float32)
    obj_ids = frame_output["out_obj_ids"].tolist()
    binary_masks = frame_output["out_binary_masks"]  # (n, H, W)
    for idx, obj_id in enumerate(obj_ids):
        mask = np.array(binary_masks[idx], dtype=bool)
        if not mask.any():
            continue
        color_rgb = (np.array(COLORS[int(obj_id) % len(COLORS)]) * 255).astype(np.uint8)
        color_bgr = color_rgb[::-1].astype(np.float32)
        out[mask] = out[mask] * (1 - alpha) + color_bgr * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_boxes_on_frame(frame: np.ndarray, frame_output: dict) -> np.ndarray:
    """Draw bounding boxes from raw SAM3 output onto a BGR frame.

    Expects 'out_obj_ids' and 'out_boxes_xywh' (normalized 0-1 coords).
    """
    out = frame.copy()
    h, w = frame.shape[:2]
    obj_ids = frame_output["out_obj_ids"].tolist()
    boxes_xywh = frame_output["out_boxes_xywh"]  # (n, 4), normalized

    for idx, obj_id in enumerate(obj_ids):
        bx, by, bw, bh = boxes_xywh[idx]
        x1, y1 = int(bx * w), int(by * h)
        x2, y2 = int((bx + bw) * w), int((by + bh) * h)

        color_rgb = (np.array(COLORS[int(obj_id) % len(COLORS)]) * 255).astype(np.uint8)
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))

        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, thickness=2)
        cv2.putText(
            out,
            f"id={obj_id}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color_bgr,
            2,
            cv2.LINE_AA,
        )

    return out


def load_video_frames(video_path: str) -> list[str]:
    if isinstance(video_path, str) and video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        video_frames_for_vis = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    else:
        video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
        try:
            # integer sort instead of string sort (so that e.g. "2.jpg" is before "11.jpg")
            video_frames_for_vis.sort(
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
            )
        except ValueError:
            # fallback to lexicographic sort if the format is not "<frame_index>.jpg"
            print(
                f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
                f"falling back to lexicographic sort."
            )
            video_frames_for_vis.sort()
    return video_frames_for_vis


def make_output_video(
    video_input: Union[str, list],
    outputs_per_frame: dict,
    output_path: str = "output.mp4",
    fps: float = 30.0,
    alpha: float = 0.5,
) -> str:
    """Render SAM3 segmentation results onto video frames and write an H.264 MP4.

    Args:
        video_input: Path to an .mp4 file, path to a directory of .jpg frames,
                     or a list of image file paths / numpy arrays.
        outputs_per_frame: Raw SAM3 outputs keyed by frame index, i.e.
                           {frame_idx: {'out_obj_ids': ..., 'out_binary_masks': ...}}.
        output_path: Destination .mp4 file path.
        fps: Frames-per-second. When the input is an .mp4 the source FPS is used instead.
        alpha: Mask overlay opacity (0 = invisible, 1 = opaque).

    Returns:
        Absolute path to the written video file.
    """
    ffmpeg = _require_ffmpeg()

    if isinstance(video_input, list):
        frame_paths_or_arrays = video_input
        source_fps = fps
    else:
        frame_paths_or_arrays = load_video_frames(video_input)
        if isinstance(video_input, str) and video_input.endswith(".mp4"):
            cap = cv2.VideoCapture(video_input)
            source_fps = cap.get(cv2.CAP_PROP_FPS) or fps
            cap.release()
        else:
            source_fps = fps

    if not frame_paths_or_arrays:
        raise ValueError(f"No frames found for input: {video_input}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    first_frame = load_frame(frame_paths_or_arrays[0])
    if first_frame.ndim == 2:
        first_frame = np.stack([first_frame] * 3, axis=-1)
    h, w = first_frame.shape[:2]

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)

    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (w, h))
    if not writer.isOpened():
        os.unlink(tmp_path)
        raise RuntimeError(f"cv2.VideoWriter failed to open: {tmp_path}")

    try:
        for frame_idx, frame_src in enumerate(frame_paths_or_arrays):
            frame_rgb = load_frame(frame_src)
            if frame_rgb.ndim == 2:
                frame_rgb = np.stack([frame_rgb] * 3, axis=-1)

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if frame_idx in outputs_per_frame:
                frame_bgr = _overlay_masks_on_frame(
                    frame_bgr, outputs_per_frame[frame_idx], alpha
                )
            writer.write(frame_bgr)
    finally:
        writer.release()

    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", tmp_path, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", output_path],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg encoding failed: {e.stderr.decode()}") from e
    finally:
        os.unlink(tmp_path)

    return str(Path(output_path).resolve())


def make_output_video_bounding_box(
    video_input: Union[str, list],
    outputs_per_frame: dict,
    output_path: str = "output.mp4",
    fps: float = 30.0,
) -> str:
    """Render SAM3 bounding boxes onto video frames and write an H.264 MP4.

    Args:
        video_input: Path to an .mp4 file, path to a directory of .jpg frames,
                     or a list of image file paths / numpy arrays.
        outputs_per_frame: Raw SAM3 outputs keyed by frame index, i.e.
                           {frame_idx: {'out_obj_ids': ..., 'out_boxes_xywh': ...}}.
        output_path: Destination .mp4 file path.
        fps: Frames-per-second. When the input is an .mp4 the source FPS is used instead.

    Returns:
        Absolute path to the written video file.
    """
    ffmpeg = _require_ffmpeg()

    if isinstance(video_input, list):
        frame_paths_or_arrays = video_input
        source_fps = fps
    else:
        frame_paths_or_arrays = load_video_frames(video_input)
        if isinstance(video_input, str) and video_input.endswith(".mp4"):
            cap = cv2.VideoCapture(video_input)
            source_fps = cap.get(cv2.CAP_PROP_FPS) or fps
            cap.release()
        else:
            source_fps = fps

    if not frame_paths_or_arrays:
        raise ValueError(f"No frames found for input: {video_input}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    first_frame = load_frame(frame_paths_or_arrays[0])
    if first_frame.ndim == 2:
        first_frame = np.stack([first_frame] * 3, axis=-1)
    h, w = first_frame.shape[:2]

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)

    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (w, h))
    if not writer.isOpened():
        os.unlink(tmp_path)
        raise RuntimeError(f"cv2.VideoWriter failed to open: {tmp_path}")

    try:
        for frame_idx, frame_src in enumerate(frame_paths_or_arrays):
            frame_rgb = load_frame(frame_src)
            if frame_rgb.ndim == 2:
                frame_rgb = np.stack([frame_rgb] * 3, axis=-1)

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if frame_idx in outputs_per_frame:
                frame_bgr = _draw_boxes_on_frame(frame_bgr, outputs_per_frame[frame_idx])
            writer.write(frame_bgr)
    finally:
        writer.release()

    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", tmp_path, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", output_path],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg encoding failed: {e.stderr.decode()}") from e
    finally:
        os.unlink(tmp_path)

    return str(Path(output_path).resolve())
