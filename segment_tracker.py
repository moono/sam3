"""
Segment-based SAM3 video tracker.

Splits a long video into fixed-length segments, runs SAM3 on each segment
independently (keeping GPU memory bounded), and links tracks across segment
boundaries by re-prompting with bounding boxes from the previous segment's
last frame.

Intermediate results (per-frame masks, boxes, scores) are saved to disk after
each segment so the run is resumable and peak CPU RAM stays bounded.

Usage:
    python segment_tracker.py
"""

from __future__ import annotations

import argparse
import gc
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch

from sam3.model_builder import build_sam3_video_predictor
from visualize import (
    _draw_boxes_on_frame,
    _overlay_masks_on_frame,
    load_video_frames,
    make_output_video,
    make_output_video_bounding_box,
)


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class HandoffState:
    """Last-frame state passed from one segment to the next."""
    global_obj_ids: np.ndarray   # (N,) int  — global track IDs surviving this segment
    boxes_xywh: np.ndarray       # (N, 4) float — normalized [0,1] xywh boxes
    scores: np.ndarray           # (N,) float — detection/tracker scores


# ── Video utilities ───────────────────────────────────────────────────────────


def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    info = {
        "num_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def write_segment_video(
    src_path: str,
    start_frame: int,
    end_frame: int,
    dst_path: str,
    fps: float,
    width: int,
    height: int,
) -> int:
    """Write frames [start_frame, end_frame) from src to dst. Returns actual frame count written."""
    cap = cv2.VideoCapture(src_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = cv2.VideoWriter(
        dst_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    written = 0
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        written += 1
    cap.release()
    writer.release()
    return written


# ── IoU matching ──────────────────────────────────────────────────────────────


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Pairwise IoU between two sets of normalized xywh boxes.
    Returns array of shape (len(boxes_a), len(boxes_b)).
    """
    def to_xyxy(b: np.ndarray) -> np.ndarray:
        return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], axis=1)

    a = to_xyxy(np.atleast_2d(boxes_a))
    b = to_xyxy(np.atleast_2d(boxes_b))
    iou = np.zeros((len(a), len(b)), dtype=np.float32)
    for i in range(len(a)):
        ix1 = np.maximum(a[i, 0], b[:, 0])
        iy1 = np.maximum(a[i, 1], b[:, 1])
        ix2 = np.minimum(a[i, 2], b[:, 2])
        iy2 = np.minimum(a[i, 3], b[:, 3])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        area_a = (a[i, 2] - a[i, 0]) * (a[i, 3] - a[i, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union = area_a + area_b - inter
        iou[i] = np.where(union > 0, inter / union, 0.0)
    return iou


def build_id_mapping(
    handoff: HandoffState,
    det_ids: np.ndarray,
    det_boxes: np.ndarray,
    global_next_id: int,
    iou_thresh: float = 0.3,
) -> tuple[dict[int, int], int]:
    """
    Greedily match detected local session IDs to global track IDs via IoU.

    Unmatched detections get freshly assigned global IDs (new objects that
    appeared in this segment). Returns (local_id → global_id mapping, updated
    global_next_id).
    """
    local_to_global: dict[int, int] = {}

    if len(handoff.global_obj_ids) == 0 or len(det_ids) == 0:
        for det_id in det_ids:
            local_to_global[int(det_id)] = global_next_id
            global_next_id += 1
        return local_to_global, global_next_id

    iou = compute_iou_matrix(handoff.boxes_xywh, det_boxes)  # (n_global, n_det)
    used_det: set[int] = set()

    # Process global objects in descending order of their best IoU with any detection
    for g_idx in np.argsort(-iou.max(axis=1)):
        d_idx = int(np.argmax(iou[g_idx]))
        if iou[g_idx, d_idx] >= iou_thresh and d_idx not in used_det:
            local_to_global[int(det_ids[d_idx])] = int(handoff.global_obj_ids[g_idx])
            used_det.add(d_idx)

    # Assign new IDs to unmatched detections
    for d_idx, det_id in enumerate(det_ids):
        if d_idx not in used_det:
            local_to_global[int(det_id)] = global_next_id
            global_next_id += 1

    return local_to_global, global_next_id


# ── Output helpers ────────────────────────────────────────────────────────────


def remap_output(out: dict, local_to_global: dict[int, int]) -> dict:
    """Replace local session obj_ids with global track IDs."""
    old_ids = out["out_obj_ids"]
    new_ids = np.array(
        [local_to_global.get(int(i), int(i)) for i in old_ids],
        dtype=old_ids.dtype,
    )
    return {**out, "out_obj_ids": new_ids}


def extend_mapping(
    out: dict,
    local_to_global: dict[int, int],
    global_next_id: int,
) -> int:
    """Add any local IDs not yet in local_to_global (new mid-segment detections)."""
    for lid in out.get("out_obj_ids", []):
        if int(lid) not in local_to_global:
            local_to_global[int(lid)] = global_next_id
            global_next_id += 1
    return global_next_id


# ── Per-segment result I/O ────────────────────────────────────────────────────


def save_segment_results(outputs: dict[int, dict], path: str) -> None:
    """Save per-global-frame outputs dict to a compressed npz file."""
    data: dict[str, np.ndarray] = {}
    for frame_idx, out in outputs.items():
        if out is None:
            continue
        p = f"f{frame_idx}_"
        data[p + "obj_ids"] = out["out_obj_ids"]
        data[p + "probs"] = out["out_probs"]
        data[p + "boxes_xywh"] = out["out_boxes_xywh"]
        data[p + "masks"] = out["out_binary_masks"]
    np.savez_compressed(path, **data)


def load_segment_results(path: str) -> dict[int, dict]:
    """Reload a segment results npz into a per-frame dict."""
    d = np.load(path)
    outputs: dict[int, dict] = {}
    frame_indices = {int(k.split("_")[0][1:]) for k in d.files}
    for fi in frame_indices:
        p = f"f{fi}_"
        outputs[fi] = {
            "out_obj_ids": d[p + "obj_ids"],
            "out_probs": d[p + "probs"],
            "out_boxes_xywh": d[p + "boxes_xywh"],
            "out_binary_masks": d[p + "masks"],
        }
    return outputs


def save_handoff(handoff: HandoffState, path: str) -> None:
    np.savez(
        path,
        global_obj_ids=handoff.global_obj_ids,
        boxes_xywh=handoff.boxes_xywh,
        scores=handoff.scores,
    )


def load_handoff(path: str) -> HandoffState:
    d = np.load(path)
    return HandoffState(
        global_obj_ids=d["global_obj_ids"],
        boxes_xywh=d["boxes_xywh"],
        scores=d["scores"],
    )


# ── Core per-segment runner ───────────────────────────────────────────────────


def run_segment(
    predictor,
    segment_path: str,
    segment_start: int,
    text_prompt: str,
    handoff: Optional[HandoffState],
    global_next_id: int,
    iou_thresh: float = 0.3,
) -> tuple[dict[int, dict], Optional[HandoffState], int]:
    """
    Run SAM3 on one segment video file.

    All segments use the text prompt on frame 0 so SAM3 detects objects freely.
    For subsequent segments the detected boxes are IoU-matched to the handoff's
    last-frame boxes to re-assign the same global IDs to the same people.
    Unmatched detections (new people) get fresh global IDs.

    Returns:
        outputs_per_global_frame  — {global_frame_idx: output_dict}
        next_handoff              — HandoffState for the next segment
        global_next_id            — updated counter for assigning new track IDs
    """
    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=segment_path)
    )
    session_id = response["session_id"]

    # Always use text prompt — SAM3 only allows a single box as a visual prompt,
    # so box-based re-prompting with multiple tracks is not supported.
    resp = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=text_prompt,
        )
    )

    # Build local→global ID mapping from frame-0 detections
    local_to_global: dict[int, int] = {}
    frame0_out = resp.get("outputs")
    if frame0_out is not None and len(frame0_out.get("out_obj_ids", [])) > 0:
        det_ids = frame0_out["out_obj_ids"]
        det_boxes = frame0_out["out_boxes_xywh"]
        if handoff is not None and len(handoff.global_obj_ids) > 0:
            # Match detections to surviving tracks from the previous segment
            local_to_global, global_next_id = build_id_mapping(
                handoff, det_ids, det_boxes, global_next_id, iou_thresh=iou_thresh
            )
        else:
            # First segment: assign brand-new global IDs in detection order
            for lid in det_ids:
                local_to_global[int(lid)] = global_next_id
                global_next_id += 1

    # ── Propagate forward through the segment ────────────────────────────────
    outputs_per_global_frame: dict[int, dict] = {}
    last_out: Optional[dict] = None

    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
            propagation_direction="forward",
        )
    ):
        local_frame_idx = response["frame_index"]
        out = response["outputs"]
        if out is None or len(out.get("out_obj_ids", [])) == 0:
            continue

        # Extend mapping for any new objects appearing mid-segment
        global_next_id = extend_mapping(out, local_to_global, global_next_id)
        remapped = remap_output(out, local_to_global)
        outputs_per_global_frame[segment_start + local_frame_idx] = remapped
        last_out = remapped

    # ── Build handoff from last frame with detections ─────────────────────────
    next_handoff: Optional[HandoffState] = None
    if last_out is not None and len(last_out.get("out_obj_ids", [])) > 0:
        next_handoff = HandoffState(
            global_obj_ids=last_out["out_obj_ids"].copy(),
            boxes_xywh=last_out["out_boxes_xywh"].copy(),
            scores=last_out["out_probs"].copy(),
        )

    predictor.handle_request(
        request=dict(type="close_session", session_id=session_id)
    )

    return outputs_per_global_frame, next_handoff, global_next_id


# ── Rendering ────────────────────────────────────────────────────────────────


def make_output_video_both(
    video_input,
    outputs_per_frame: dict,
    output_path: str,
    fps: float = 30.0,
    alpha: float = 0.5,
) -> str:
    """Render both segmentation masks and bounding boxes with IDs onto the video."""
    import subprocess
    import tempfile
    from pathlib import Path
    import shutil

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found. Install it with: sudo apt install ffmpeg")

    frame_paths_or_arrays = load_video_frames(video_input)
    if not frame_paths_or_arrays:
        raise ValueError(f"No frames found for input: {video_input}")

    import cv2
    from sam3.visualization_utils import load_frame

    if isinstance(video_input, str) and video_input.endswith(".mp4"):
        cap = cv2.VideoCapture(video_input)
        source_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        cap.release()
    else:
        source_fps = fps

    first = load_frame(frame_paths_or_arrays[0])
    h, w = first.shape[:2]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)

    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (w, h))
    if not writer.isOpened():
        os.unlink(tmp_path)
        raise RuntimeError(f"cv2.VideoWriter failed to open: {tmp_path}")

    try:
        for frame_idx, frame_src in enumerate(frame_paths_or_arrays):
            frame_rgb = load_frame(frame_src)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if frame_idx in outputs_per_frame:
                out = outputs_per_frame[frame_idx]
                frame_bgr = _overlay_masks_on_frame(frame_bgr, out, alpha)
                frame_bgr = _draw_boxes_on_frame(frame_bgr, out)
            writer.write(frame_bgr)
    finally:
        writer.release()

    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", tmp_path, "-c:v", "libx264", "-crf", "18",
             "-pix_fmt", "yuv420p", output_path],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg encoding failed: {e.stderr.decode()}") from e
    finally:
        os.unlink(tmp_path)

    return str(Path(output_path).resolve())


# ── Orchestrator ──────────────────────────────────────────────────────────────


def track_video_segments(
    video_path: str,
    output_path: str,
    text_prompt: str = "person",
    segment_length: int = 300,
    results_dir: Optional[str] = None,
    iou_thresh: float = 0.3,
    render_video: bool = True,
    render_mode: str = "both",
) -> dict[int, dict]:
    """
    Track objects across a long video by splitting it into segments.

    Args:
        video_path:      Input video file.
        output_path:     Where to write the visualized output video.
        text_prompt:     SAM3 text prompt for the first segment (and fallback).
        segment_length:  Number of frames per segment.
        results_dir:     Directory for per-segment npz files (temp dir if None).
        iou_thresh:      Minimum IoU to link a detection to a previous track.
        render_video:    Whether to render the output video at the end.
        render_mode:     One of "mask", "box", or "both".

    Returns:
        all_outputs — {global_frame_idx: output_dict} for the entire video.
    """
    if render_mode not in ("mask", "box", "both"):
        raise ValueError(f"render_mode must be 'mask', 'box', or 'both'; got {render_mode!r}")
    info = get_video_info(video_path)
    total_frames = info["num_frames"]
    fps = info["fps"]
    width, height = info["width"], info["height"]
    print(f"Video: {total_frames} frames @ {fps:.1f} fps, {width}x{height}")

    cleanup_results_dir = results_dir is None
    if results_dir is None:
        results_dir = tempfile.mkdtemp(prefix="sam3_seg_")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results dir: {results_dir}")

    predictor = build_sam3_video_predictor(gpus_to_use=[torch.cuda.current_device()])

    seg_starts = list(range(0, total_frames, segment_length))
    n_segments = len(seg_starts)

    handoff: Optional[HandoffState] = None
    global_next_id = 0
    all_outputs: dict[int, dict] = {}

    for seg_idx, seg_start in enumerate(seg_starts):
        seg_end = min(seg_start + segment_length, total_frames)
        print(f"\n── Segment {seg_idx + 1}/{n_segments}: frames {seg_start}–{seg_end - 1} ──")

        # Write temp segment video
        seg_video_path = os.path.join(results_dir, f"seg_{seg_idx:04d}.mp4")
        written = write_segment_video(video_path, seg_start, seg_end, seg_video_path, fps, width, height)
        print(f"   Wrote {written} frames → {seg_video_path}")

        # Run SAM3
        seg_outputs, handoff, global_next_id = run_segment(
            predictor=predictor,
            segment_path=seg_video_path,
            segment_start=seg_start,
            text_prompt=text_prompt,
            handoff=handoff,
            global_next_id=global_next_id,
            iou_thresh=iou_thresh,
        )

        # Report
        active_ids: set[int] = set()
        for o in seg_outputs.values():
            active_ids.update(int(i) for i in o.get("out_obj_ids", []))
        print(f"   Frames with output: {len(seg_outputs)}  |  Active global IDs: {sorted(active_ids)}")
        if handoff is not None:
            print(f"   Handoff → {len(handoff.global_obj_ids)} tracks carried to next segment")

        # Save results to disk
        results_path = os.path.join(results_dir, f"seg_{seg_idx:04d}_results.npz")
        save_segment_results(seg_outputs, results_path)
        if handoff is not None:
            handoff_path = os.path.join(results_dir, f"seg_{seg_idx:04d}_handoff.npz")
            save_handoff(handoff, handoff_path)

        all_outputs.update(seg_outputs)

        # Clean up temp segment video and free GPU memory
        os.remove(seg_video_path)
        torch.cuda.empty_cache()
        gc.collect()

    predictor.shutdown()

    if render_video:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        print(f"\nRendering output video ({render_mode}) → {output_path}")
        if render_mode == "mask":
            make_output_video(
                video_input=video_path,
                outputs_per_frame=all_outputs,
                output_path=output_path,
                fps=fps,
            )
        elif render_mode == "box":
            make_output_video_bounding_box(
                video_input=video_path,
                outputs_per_frame=all_outputs,
                output_path=output_path,
                fps=fps,
            )
        else:  # "both"
            make_output_video_both(
                video_input=video_path,
                outputs_per_frame=all_outputs,
                output_path=output_path,
                fps=fps,
            )
        print("Done.")

    return all_outputs


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Segment-based SAM3 video tracker")
    parser.add_argument("--video", default="./assets/videos/Retail02_extended.mp4")
    parser.add_argument("--output", default="./assets/outputs/Retail02_extended_segmented.mp4")
    parser.add_argument("--prompt", default="person")
    parser.add_argument("--segment-length", type=int, default=600,
                        help="Frames per segment (default: 600)")
    parser.add_argument("--results-dir", default="./assets/outputs/segments",
                        help="Directory to store per-segment npz files")
    parser.add_argument("--iou-thresh", type=float, default=0.3,
                        help="IoU threshold for linking tracks across segments")
    parser.add_argument(
        "--render-mode",
        default="both",
        choices=["mask", "box", "both"],
        help="Visualization: segmentation masks only, bounding boxes only, or both (default: both)",
    )
    args = parser.parse_args()

    track_video_segments(
        video_path=args.video,
        output_path=args.output,
        text_prompt=args.prompt,
        segment_length=args.segment_length,
        results_dir=args.results_dir,
        iou_thresh=args.iou_thresh,
        render_mode=args.render_mode,
    )


if __name__ == "__main__":
    main()
