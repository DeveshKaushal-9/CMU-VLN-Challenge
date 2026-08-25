"""Open-vocabulary detection and 2D->3D grounding.

Detection runs on a worker thread so exploration is never blocked. The detector
is prompted with only the nouns the current question mentions, which keeps
CPU inference affordable: the panorama is split into overlapping tiles and each
tile costs well under a second.

If torch/transformers are unavailable the module degrades to a disabled state
and the node falls back to purely geometric answers rather than crashing.
"""

import math
import threading
import queue

import numpy as np

MODEL_ID = 'google/owlv2-base-patch16-ensemble'
# OWLv2 replaces OWL-ViT because patch16 at 960px carries ~6x the spatial
# tokens of patch32 at 768px, which is what small objects need. Measured on a
# real livingroom_3 panorama with the same prompts, OWL-ViT found NONE of the
# three framed photos plainly visible on the wall; OWLv2 found nine, and
# tripled detections under 120px. It costs 19.2s per keyframe against 11.2s.
#
# 960px tiles (three, overlapping 240px) beat 640px tiles on both recall and
# total cost, because OWLv2 pads to a 960 square and a 960-wide tile maps to
# it without upscaling. Its post-processing returns unpadded coordinates, so
# boxes need no correction - verified against known object positions.
TILE_WIDTH = 960
TILE_STRIDE = 720
SCORE_THRESHOLD = 0.14    # below this OWL-ViT mostly reports texture, not objects
MERGE_RADIUS = 0.6        # metres; detections closer than this are one object
MIN_OBSERVATIONS = 2      # a weak box seen once is usually texture...
STRONG_SINGLE = 0.30      # ...but a confident one stands on its own
DEPTH_BACK = 0.35         # depth band kept in front of the nearest surface
DEPTH_FRONT = 0.65        # ...and behind it
# OWLv2 takes ~19s per keyframe, so a deep queue just holds stale frames from
# one spot. Keep it shallow: fewer frames, spread further along the path.
MAX_QUEUE = 2
MIN_CONFIDENCE = 0.22     # confidence bar for an object to be reasoned over
GEOM_BONUS = 1.6          # multiplier for a detection backed by lidar geometry


class GroundedObject:
    """An object located in the map frame by fusing detections across views."""

    __slots__ = ('label', 'center', 'extent', 'score', 'n_obs', 'geom')

    def __init__(self, label, center, extent, score):
        self.label = label
        self.center = tuple(float(v) for v in center)
        self.extent = tuple(float(v) for v in extent)
        self.score = float(score)
        self.n_obs = 1
        self.geom = False      # set when a lidar cluster backs this detection

    @property
    def xy(self):
        return self.center[0], self.center[1]

    @property
    def confidence(self):
        """Detection strength reinforced by how often the object was re-seen.

        A weak box that shows up from several viewpoints is far more likely to
        be real than a strong one-off, which is usually texture on a wall.
        """
        c = self.score * (1.0 + 0.45 * math.log(self.n_obs))
        # A detection standing on a real lidar cluster outranks one floating in
        # space. Scores alone cannot separate them - on livingroom_3 the true
        # stool scored 0.29 and lost to a phantom at 0.30 nearly four metres
        # away - but only one of the two has an object-shaped point cloud
        # underneath it.
        return c * (GEOM_BONUS if self.geom else 1.0)

    def merge(self, other):
        """Fold another observation of the same object into this one."""
        w_old, w_new = self.n_obs, 1
        total = w_old + w_new
        self.center = tuple((self.center[i] * w_old + other.center[i] * w_new) / total
                            for i in range(3))
        self.extent = tuple(max(self.extent[i], other.extent[i]) for i in range(3))
        self.score = max(self.score, other.score)
        self.geom = self.geom or other.geom
        self.n_obs = total

    def __repr__(self):
        return (f'<{self.label} @({self.center[0]:.2f},{self.center[1]:.2f},'
                f'{self.center[2]:.2f}) s={self.score:.2f} n={self.n_obs}>')


def merge_distance(a, b):
    """How far apart two detections of one class can sit and still be one object.

    A fixed radius cannot serve both ends of the vocabulary: a 1.65m cabinet
    viewed from two sides yields centres a metre apart and fragments into
    several objects, while a 0.46m vase would happily swallow its neighbour.
    Scale the allowance to the larger footprint instead.
    """
    span = max(a.extent[0], a.extent[1], b.extent[0], b.extent[1])
    return max(MERGE_RADIUS, 0.6 * span)


def tile_offsets(width, tile=TILE_WIDTH, stride=TILE_STRIDE):
    """Left edges of overlapping tiles covering a panorama that wraps around."""
    return list(range(0, width, stride))


class Grounder:
    """Runs open-vocabulary detection over keyframes on a background thread."""

    def __init__(self, logger=None, model_id=MODEL_ID, threshold=SCORE_THRESHOLD):
        self.log = logger
        self.model_id = model_id
        self.threshold = threshold

        self._model = None
        self._proc = None
        self._torch = None
        self.available = False
        self.ready = False
        self.load_error = None

        self._prompts = []
        self._lock = threading.Lock()
        self._objects = []
        self._queue = queue.Queue(maxsize=MAX_QUEUE)
        self._stop = threading.Event()
        self._frames_done = 0

        threading.Thread(target=self._load, daemon=True).start()
        threading.Thread(target=self._work, daemon=True).start()

    # -- lifecycle ----------------------------------------------------------
    def _info(self, msg):
        if self.log:
            self.log.info(msg)

    def _load(self):
        try:
            import torch
            from transformers import Owlv2Processor, Owlv2ForObjectDetection
            # Leave real headroom for the ROS executor. rclpy.spin is single
            # threaded, so if inference takes every core the image callback
            # stops running and the detector starves its own input: a run with
            # cpu_count-1 threads processed 5 keyframes in 406s while frames
            # aged out to 300s, with the camera publishing normally throughout.
            cpus = __import__('os').cpu_count() or 4
            torch.set_num_threads(max(1, cpus // 2))
            self._torch = torch
            self._proc = Owlv2Processor.from_pretrained(self.model_id)
            self._model = Owlv2ForObjectDetection.from_pretrained(self.model_id).eval()
            self.available = True
            self.ready = True
            self._info(f'detector ready ({self.model_id})')
        except Exception as exc:
            self.load_error = str(exc)
            self.available = False
            self.ready = True    # ready in the sense of "done trying"
            self._info(f'detector unavailable, geometry-only mode: {exc}')

    def stop(self):
        self._stop.set()

    # -- public API ---------------------------------------------------------
    def set_prompts(self, nouns):
        """Restrict detection to the nouns this question actually mentions."""
        seen, clean = set(), []
        for n in nouns:
            n = (n or '').strip().lower()
            if n and n not in seen and n not in ('path', 'it', 'them'):
                seen.add(n)
                clean.append(n)
        with self._lock:
            self._prompts = clean
        self._info(f'detector prompts: {clean}')

    def submit(self, keyframe):
        """Offer a keyframe for detection; dropped if the worker is busy."""
        if not self.available or not self._prompts:
            return False
        try:
            self._queue.put_nowait(keyframe)
            return True
        except queue.Full:
            return False

    def objects(self, min_confidence=MIN_CONFIDENCE):
        """Grounded objects worth reasoning over.

        Falls back to the best few when nothing clears the bar, so a hard scene
        still produces an answer instead of nothing.
        """
        with self._lock:
            everything = list(self._objects)
        keep = [o for o in everything
                if o.confidence >= min_confidence
                and (o.n_obs >= MIN_OBSERVATIONS or o.score >= STRONG_SINGLE)]
        if keep:
            return keep
        everything.sort(key=lambda o: o.confidence, reverse=True)
        return everything[:8]

    @property
    def frames_processed(self):
        return self._frames_done

    # -- worker -------------------------------------------------------------
    def _work(self):
        while not self._stop.is_set():
            try:
                kf = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if not self.available:
                continue
            try:
                self._process(kf)
                self._frames_done += 1
            except Exception as exc:      # a bad frame must not kill the thread
                self._info(f'detection error: {exc}')

    def _process(self, kf):
        from PIL import Image

        with self._lock:
            prompts = list(self._prompts)
        if not prompts or kf.image is None:
            return

        texts = [[f'a photo of a {p}' for p in prompts]]
        h, w = kf.image.shape[:2]
        rgb = kf.image[:, :, ::-1]        # BGR (as published) -> RGB

        found = []
        for x0 in tile_offsets(w):
            tile, x_off = self._crop(rgb, x0, w)
            pil = Image.fromarray(tile)
            with self._torch.no_grad():
                inputs = self._proc(text=texts, images=pil, return_tensors='pt')
                out = self._model(**inputs)
                res = self._proc.post_process_grounded_object_detection(
                    outputs=out, threshold=self.threshold,
                    target_sizes=self._torch.tensor([[tile.shape[0], tile.shape[1]]]))[0]

            for score, label_idx, box in zip(res['scores'], res['labels'], res['boxes']):
                bx0, by0, bx1, by1 = (float(v) for v in box)
                found.append((prompts[int(label_idx)], float(score),
                              (bx0 + x_off) % w, by0, (bx1 + x_off) % w, by1))

        self._fuse(kf, found, w)

    @staticmethod
    def _crop(rgb, x0, w, tile=TILE_WIDTH):
        """Crop a tile, wrapping around the panorama seam."""
        x1 = x0 + tile
        if x1 <= w:
            return np.ascontiguousarray(rgb[:, x0:x1]), x0
        return np.ascontiguousarray(
            np.concatenate([rgb[:, x0:w], rgb[:, :x1 - w]], axis=1)), x0

    def _fuse(self, kf, found, width):
        """Turn 2D boxes into map-frame objects and merge with what we have."""
        new = []
        for label, score, x0, y0, x1, y1 in found:
            if x1 < x0:               # box crossed the seam
                x1 += width
            pts = kf.points_in_box(x0, y0, x1, y1)
            if pts.shape[0] < 8:
                continue
            # A detection box also sees whatever is behind the object. Keep the
            # nearest coherent depth band so the box tracks the object surface
            # rather than the wall behind it.
            origin = np.asarray(kf.position, dtype=np.float32)
            rng = np.linalg.norm(pts - origin, axis=1)
            anchor = float(np.percentile(rng, 20))
            pts = pts[(rng >= anchor - DEPTH_BACK) & (rng <= anchor + DEPTH_FRONT)]
            if pts.shape[0] < 8:
                continue
            med = np.median(pts, axis=0)
            pts = pts[np.linalg.norm(pts - med, axis=1) < 1.5]
            if pts.shape[0] < 8:
                continue
            # Object reference is scored on overlap with the ground-truth box,
            # so trim straggler points rather than taking the raw min/max.
            mins = np.percentile(pts, 8, axis=0)
            maxs = np.percentile(pts, 92, axis=0)
            centre = (mins + maxs) / 2.0
            extent = np.maximum(maxs - mins, 0.12)
            if max(extent[0], extent[1]) > 3.5:
                continue              # spans the room: a bad box
            new.append(GroundedObject(label, centre, extent, score))

        if not new:
            return
        with self._lock:
            for obj in new:
                match = None
                for have in self._objects:
                    if have.label != obj.label:
                        continue
                    d = np.linalg.norm(np.asarray(have.center) - np.asarray(obj.center))
                    if d < merge_distance(have, obj):
                        match = have
                        break
                if match is not None:
                    match.merge(obj)
                else:
                    self._objects.append(obj)
