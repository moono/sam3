### How track IDs are preserved across segments

**1. Each segment runs a fresh SAM3 session**
SAM3 gets a new session for every segment — it has no memory of previous segments. When you run `add_prompt` with `text="person"` on frame 0 of segment N+1, SAM3 freely re-detects all visible people and assigns them `local session IDs` (0, 1, 2, …) that are completely arbitrary and unrelated to IDs from the previous segment.

**2. The handoff captures last-frame state**
At the very end of each segment, `run_segment` records the final frame's detections into a `HandoffState`:

```python
next_handoff = HandoffState(
    global_obj_ids=last_out["out_obj_ids"].copy(),   # e.g. [3, 7, 12]  ← global IDs
    boxes_xywh=last_out["out_boxes_xywh"].copy(),    # where each person was
    scores=last_out["out_probs"].copy(),
    masks=last_out["out_binary_masks"].copy(),        # what each person's mask looked like
)
```

This is the "memory" passed to the next segment. It says: _"At the end of segment N, person with global ID 3 was at this location with this mask shape, ID 7 was here, etc."_

**3. IoU matching bridges the gap**

At the start of segment N+1, frame 0 has a fresh set of detections with local IDs `[0, 1, 2, ...]`. The key question is: **which local ID corresponds to which global ID?**
`build_id_mapping` answers this by computing a **pairwise IoU matrix** between:

* the handoff's masks (from the last frame of segment N)
* the new detections' masks (from the first frame of segment N+1)

Since the two frames are adjacent in time, the same person will appear in nearly the same position with nearly the same mask shape — so their IoU will be high.

```python
handoff masks (from seg N last frame):
  global_id=3 → mask at position A
  global_id=7 → mask at position B
  global_id=12 → mask at position C

new detections (seg N+1 frame 0):
  local_id=0 → mask at position B   ← matches global_id=7
  local_id=1 → mask at position A   ← matches global_id=3
  local_id=2 → mask at position C   ← matches global_id=12

IoU matrix:          local=0  local=1  local=2
  global_id=3  [      0.05,    0.91,    0.02  ]
  global_id=7  [      0.88,    0.04,    0.03  ]
  global_id=12 [      0.02,    0.03,    0.87  ]
```

Greedy matching (highest IoU first, no double-assignments) then builds:

```python
local_to_global = {0: 7, 1: 3, 2: 12}
```

**4. Remapping is applied to every output frame**
With `local_to_global` established, every frame output in the new segment gets its IDs replaced:

```python
remapped = remap_output(out, local_to_global)
# out["out_obj_ids"] = [0, 1, 2]  →  remapped["out_obj_ids"] = [7, 3, 12]
```

This is what makes the global ID for a given person remain the same throughout the entire video — even though SAM3 internally restarted.

**5. New objects get fresh IDs, disappeared objects are simply dropped**
* If a person appears for the first time in segment N+1 (no IoU match ≥ threshold), they get a new ID from global_next_id counter.
* If a person from segment N is not detected in segment N+1's frame 0 (they left the scene or were occluded), their global ID is simply not carried forward — no placeholder or "lost track" state.

**Why mask IoU over box IoU?**
In crowded scenes, two people's bounding boxes can heavily overlap even when they are distinct. Masks, being the actual pixel-level footprint, are much more discriminative — two overlapping boxes might have low mask IoU, correctly distinguishing the two people.