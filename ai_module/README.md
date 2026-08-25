# AI Module — CMU VLN Challenge 2026

Replaces the stock `dummy_vlm` C++ node with a Python ROS 2 node that explores
an unknown scene, builds a lightweight geometric world model, and answers the
three challenge question types.

## Design

The module runs **entirely offline** — no external API calls, and CPU-only
inference. That is deliberate: the evaluation machine has no discrete GPU, and
across the 75 released training questions the spatial vocabulary is a closed
set of roughly ten predicates (`closest to`, `on`, `near`, `between`, `above`,
`below`, `under`, `farthest from`, `avoid`, plus sequencing words). The
language side is therefore tractable with a parser, and the difficulty sits in
perception and geometry.

```
/terrain_map_ext ──► OccupancyGrid ──► frontier exploration ──► /way_point_with_heading
                                              │
/camera/image  ──┐                            ▼
/registered_scan ├──► Keyframe ──► OWL-ViT ──► back-projection ──► GroundedObject
/state_estimation┘     (worker thread)         (depth-gated)            │
                                                                        ▼
/challenge_question ──► ParsedQuestion ──► spatial predicates ──► answer
                                                   ├──► /numerical_response
                                                   ├──► /selected_object_marker
                                                   └──► /way_point_with_heading
```

**Detection is prompted with only the nouns the current question mentions.**
That is what makes open-vocabulary detection affordable on CPU — the detector
never has to find everything, only the three or four things being asked about.
The panorama is split into four overlapping 640px tiles (each resized to 768²
by the processor, so wider tiles cost nothing extra), and detection runs on a
worker thread so exploration is never blocked.

Each 2D box becomes a 3D object by selecting the scan points whose bearing
falls inside it, then keeping the nearest coherent depth band — without that
gate the box swallows the wall behind the object. Boxes are cut at the 8th/92nd
percentile of the surviving points, since object reference is scored on overlap
with the ground-truth box.

Objects re-seen from several viewpoints are far more likely to be real than
strong one-off detections, so confidence is `score × (1 + 0.45·ln n_obs)` and
the reasoning layer only sees objects clearing a confidence bar.

| File | Role |
|------|------|
| `src/dummy_vlm/dummy_vlm/node.py` | ROS interfaces, time budgeting, state machine |
| `src/dummy_vlm/dummy_vlm/terrain.py` | Occupancy grid, obstacle inflation, BFS planner, frontier selection |
| `src/dummy_vlm/dummy_vlm/objects.py` | Voxel accumulation and connected-component object segmentation |
| `src/dummy_vlm/dummy_vlm/question.py` | Question classification and spatial-relation parsing |
| `src/dummy_vlm/dummy_vlm/camera.py` | Panoramic projection and 2D→3D back-projection |
| `src/dummy_vlm/dummy_vlm/grounding.py` | OWL-ViT detection, keyframes, detection fusion |
| `src/dummy_vlm/dummy_vlm/spatial.py` | Spatial predicates over grounded objects |
| `src/dummy_vlm/dummy_vlm/pc2.py` | Dependency-free `PointCloud2` → numpy |

## Time budgeting

Each question allows 10 minutes for exploration and answering combined, and the
system is relaunched per question so nothing carries over. The run is split by
question type — instruction-following reserves most of the budget for driving,
while numerical and object-reference questions explore longer before
committing:

| Type | Explore fraction |
|------|------------------|
| Numerical | 0.70 |
| Object reference | 0.70 |
| Instruction-following | 0.45 |

An answer is also committed early if the scene looks fully mapped, guarded so
that an empty map (data not yet flowing) is never mistaken for full coverage.

## Dependencies

CPU-only `torch`, `transformers`, and `pillow`, on top of the `numpy` the base
image already carries. The OWL-ViT weights are baked into the image at build
time and `HF_HUB_OFFLINE=1` is set, so nothing is fetched from the network
during evaluation. The image is roughly 1.7GB larger than the organizers' base.

If `torch`/`transformers` fail to import the module logs the failure and
degrades to geometry-only answers rather than crashing.

## Validation

`test/test_reasoning.py` runs the language and spatial layers offline. It parses
all 75 released questions, and where the organizers' ground-truth object lists
are present under `scenes/gt/` it replays the spatial reasoning against perfect
perception — which separates reasoning faults from detection faults, since
anything failing there cannot work once the detector is in the loop.

```bash
python3 ai_module/test/test_reasoning.py
```

```
parsing:              75/75 questions clean
object reference:     30/30 resolved to the right class
instruction steps:    60/60 resolved to the right class
between gateways:     11/11 produced a midpoint
numerical:            12/15 exactly correct
```

## Known limitations

- **Colour qualifiers are parsed but not enforced.** `Phrase.colors` is
  populated, and no predicate consumes it, so "how many *red* pillows are on the
  sofa" counts every pillow on the sofa. This accounts for the remaining
  numerical misses. Colour is deliberately kept out of the detector prompt:
  detections labelled "red pillow" and "pillow" do not merge in `Grounder._fuse`,
  so prompting for both would double-count one physical object.
- **A missing class falls back to the whole object set.** `candidates_for`
  returns everything when no label matches, so a target the detector never sees
  yields an arbitrary object rather than no answer. That is intentional — a
  wrong marker scores zero, the same as no marker, and a published answer keeps
  the run scoreable.

## Running locally

**Set `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` explicitly in every shell.** The
image exports it from `~/.bashrc`, which `bash -lc` does not source; without
it the default FastDDS transport uses shared memory, topic discovery succeeds
across containers but no data is delivered.

```bash
cd docker && docker compose -f compose_gpu.yml up --build -d

docker exec -it iros2026_system bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh

docker exec -it iros2026_ai_module bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch dummy_vlm dummy_vlm.launch
```
