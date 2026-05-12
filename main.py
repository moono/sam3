import os
import torch

from sam3.model_builder import build_sam3_video_predictor
from visualize import make_output_video

# The key differences vs the streaming version:
# build_sam3_video_predictor() with no extra args — streaming_video_frames, async_loading_frames, trim_past_non_cond_mem_for_eval, offload_output_to_cpu_for_eval, max_cached_frame_outputs are all at their defaults
# start_session with no offload_video_to_cpu / offload_state_to_cpu — all frames and state stay on GPU


def streaming_example():
    gpus_to_use = [torch.cuda.current_device()]
    predictor = build_sam3_video_predictor(
        gpus_to_use=gpus_to_use,
        # ── Frame loading (predictor-level) ──────────────────────────────
        # Decode frames on demand; async pre-loading is incompatible with streaming.
        streaming_video_frames=True,
        async_loading_frames=False,
        # ── Tracker memory (model-level, baked in at build time) ─────────
        # Discard old non-conditioning tracker memories beyond the sliding window.
        trim_past_non_cond_mem_for_eval=True,
        # Move predicted masks off-GPU immediately after each frame.
        offload_output_to_cpu_for_eval=True,
        # ── Output cache (predictor-level) ───────────────────────────────
        # Keep only the 32 most-recent frame outputs in the SAM3-level cache.
        max_cached_frame_outputs=32,
    )

    # input_name = "0001.mp4"
    input_name = "Retail02_extended.mp4"
    # input_name = "bedroom.mp4"
    video_path = os.path.join("./assets/videos", input_name)
    output_path = os.path.join(
        "./assets/outputs", f"{os.path.splitext(input_name)[0]}_output_streaming.mp4"
    )

    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
            # ── Session-level memory flags ────────────────────────────────
            # Keep decoded frames on CPU; one frame is moved to GPU per step.
            offload_video_to_cpu=True,
            # Keep tracker state tensors on CPU between frames.
            offload_state_to_cpu=True,
        )
    )
    session_id = response["session_id"]

    prompt_text_str = "person"
    frame_idx = 0  # add a text prompt on frame 0
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=frame_idx,
            text=prompt_text_str,
        )
    )

    # propagate from frame 0 to the end and collect all outputs
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    _ = predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )

    predictor.shutdown()

    output_file1 = make_output_video(
        video_input=video_path,
        outputs_per_frame=outputs_per_frame,
        output_path=output_path,
        fps=30.0,
    )
    print(f"Saved segmentation video: {output_file1}")
    return


def run_all():
    gpus_to_use = [torch.cuda.current_device()]
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

    # input_name = "0001.mp4"
    # input_name = "Retail02_extended.mp4"
    input_name = "bedroom.mp4"
    video_path = os.path.join("./assets/videos", input_name)
    output_path = os.path.join(
        "./assets/outputs", f"{os.path.splitext(input_name)[0]}_output.mp4"
    )

    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=video_path)
    )
    session_id = response["session_id"]

    prompt_text_str = "person"
    frame_idx = 0  # add a text prompt on frame 0
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=frame_idx,
            text=prompt_text_str,
        )
    )

    # propagate from frame 0 to the end and collect all outputs
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    _ = predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )

    predictor.shutdown()

    output_file1 = make_output_video(
        video_input=video_path,
        outputs_per_frame=outputs_per_frame,
        output_path=output_path,
        fps=30.0,
    )
    print(f"Saved segmentation video: {output_file1}")
    return


def main():
    # run_all()
    streaming_example()
    return


if __name__ == "__main__":
    main()
