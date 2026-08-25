"""CMU VLN Challenge 2026 - AI module entry point.

The node runs a single question per system launch. Timing starts at launch,
so the run is split into an exploration budget and an answering budget, sized
by question type: instruction-following needs most of its time for driving,
while numerical and object-reference questions want the map as complete as
possible before committing to an answer.
"""

import math
from collections import deque

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String, Int32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import PointCloud2, Image
from visualization_msgs.msg import Marker

from . import pc2, question as Q, spatial
from .camera import Keyframe
from .grounding import Grounder
from .terrain import OccupancyGrid
from .objects import ObjectMap

TOTAL_BUDGET = 600.0      # seconds per question, set by the challenge rules
SAFETY_MARGIN = 20.0      # stop short of the limit to avoid the time penalty

# Fraction of the budget spent exploring before an answer is committed.
EXPLORE_FRACTION = {
    Q.NUMERICAL: 0.70,
    Q.OBJECT_REFERENCE: 0.70,
    Q.INSTRUCTION: 0.45,
}

WAYPOINT_REACH = 1.0      # metres
# A leg's goal is an object centre, so it often sits inside the furniture the
# object rests on. The base autonomy pushes such a waypoint into traversable
# space and the robot halts at the furniture edge, never inside `reach` of the
# published point. Without a timeout the route stalls there and every later leg
# is lost - which on a 6-point instruction is most of the question.
LEG_TIMEOUT = 60.0        # seconds before a leg is treated as done anyway
MIN_COVERAGE_CELLS = 400  # ~16 m2 mapped before "covered" is believable
# Finishing early only earns tie-break bonus points, while a weak answer loses
# real ones, so an early commit has to clear a high bar: a small room maps in
# well under a minute, long before detection has seen it from enough angles.
MIN_EXPLORE_TIME = 150.0  # never commit an answer sooner than this

# The camera publishes at well under 1Hz, so the newest image can be over a
# second old while odometry runs at 200Hz. Pairing a stale image with a fresh
# heading misregisters the panorama: 1920px covers 360 degrees, so 5.33px per
# degree, and 20 degrees of yaw during that window shifts the image 107px. The
# bearing cone then selects floor or wall instead of the object, which is what
# scattered one real object across dozens of 3D positions.
IMAGE_MAX_AGE = 0.6       # seconds; drop keyframes built on a staler image
KEYFRAME_PERIOD = 3.0     # seconds between detection keyframes
KEYFRAME_DIST = 0.8       # or this much travel, whichever comes first
SCAN_HISTORY = 8          # registered scans kept per keyframe
MIN_KEYFRAMES = 20        # detection passes before an early answer is allowed
AVOID_RADIUS = 1.2        # metres to keep clear of a forbidden gateway
SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)


class VLNNode(Node):

    def __init__(self):
        super().__init__('dummyVLM')

        self.declare_parameter('total_budget', TOTAL_BUDGET)
        self.declare_parameter('waypoint_reach', WAYPOINT_REACH)
        self.budget = float(self.get_parameter('total_budget').value)
        self.reach = float(self.get_parameter('waypoint_reach').value)

        # --- state ---------------------------------------------------------
        self.t0 = self.get_clock().now()
        self.vehicle = (0.0, 0.0)
        self.vehicle_z = 0.0
        self.heading = 0.0
        self.question = None
        self.parsed = None
        self.grid = OccupancyGrid()
        self.objects = ObjectMap()
        self.answered = False
        self.answer_marker = None
        self.route = []           # list of (x, y) still to drive
        self.leg_started = None   # when the current leg was first published
        self.visited = []         # frontier targets already issued
        self.current_wp = None
        self.forbidden = []
        self.grounded = []
        self.image_count = 0
        self.last_coverage = 0
        self.stalled_ticks = 0
        self.sweep_i = -1

        self.latest_image = None
        self.latest_image_pose = None   # pose sampled AT image arrival
        self.latest_image_yaw = 0.0
        self.latest_image_t = -1e9
        self.recent_scans = deque(maxlen=SCAN_HISTORY)
        self.last_keyframe_t = -1e9
        self.last_image_used = -1e9   # stamp of the frame already submitted
        self.last_keyframe_xy = None
        self.keyframes_sent = 0
        self.grounder = Grounder(logger=self.get_logger())

        # --- interfaces ----------------------------------------------------
        self.create_subscription(String, '/challenge_question', self.on_question, 5)
        self.create_subscription(Odometry, '/state_estimation', self.on_odom, 5)
        self.create_subscription(PointCloud2, '/terrain_map_ext', self.on_terrain, SENSOR_QOS)
        self.create_subscription(PointCloud2, '/registered_scan', self.on_scan, SENSOR_QOS)
        self.create_subscription(Image, '/camera/image', self.on_image, SENSOR_QOS)

        self.pub_wp = self.create_publisher(Pose2D, '/way_point_with_heading', 5)
        self.pub_num = self.create_publisher(Int32, '/numerical_response', 5)
        self.pub_marker = self.create_publisher(Marker, '/selected_object_marker', 5)

        self.create_timer(0.2, self.tick)
        self.get_logger().info('AI module up; exploring until a question arrives.')

    # -- clock ---------------------------------------------------------------
    @property
    def elapsed(self):
        return (self.get_clock().now() - self.t0).nanoseconds * 1e-9

    def explore_deadline(self):
        frac = EXPLORE_FRACTION.get(
            self.parsed.type if self.parsed else Q.OBJECT_REFERENCE, 0.6)
        return (self.budget - SAFETY_MARGIN) * frac

    # -- callbacks -----------------------------------------------------------
    def on_question(self, msg):
        if self.question is not None:
            return  # published at 1Hz; only the first matters
        self.question = msg.data
        try:
            self.parsed = Q.ParsedQuestion(msg.data)
        except Exception as exc:  # never let a parse error kill the run
            self.get_logger().error(f'parse failed: {exc}')
            self.parsed = None
        self.get_logger().info(f'Question ({self._qtype()}): {msg.data}')
        self.grounder.set_prompts(self._question_nouns())

    def _question_nouns(self):
        """Every noun the question mentions - the detector's whole vocabulary."""
        nouns = []
        if not self.parsed:
            return nouns

        def add(phrase, relations):
            if phrase is not None and phrase.noun:
                nouns.append(phrase.noun)
                if getattr(phrase, 'detector_label', None):
                    nouns.append(phrase.detector_label)
            for rel in relations:
                for anchor in rel.anchors:
                    if anchor.noun:
                        nouns.append(anchor.noun)
                    if getattr(anchor, 'detector_label', None):
                        nouns.append(anchor.detector_label)

        add(self.parsed.target, self.parsed.relations)
        for step in self.parsed.steps:
            add(step.target, step.relations)
        return nouns

    def on_odom(self, msg):
        p = msg.pose.pose.position
        self.vehicle = (p.x, p.y)
        # The sensor rides about 0.75m up. Back-projection measures elevation
        # from this origin, so dropping it tilts every bearing: at 3m a 0.75m
        # error is 14 degrees, which is ~75px of the 640px image height and
        # lands the box on the floor instead of the object.
        self.vehicle_z = p.z
        q = msg.pose.pose.orientation
        self.heading = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                  1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def on_terrain(self, msg):
        try:
            self.grid.update(pc2.to_xyzi(msg))
        except Exception as exc:
            self.get_logger().warn(f'terrain update failed: {exc!r}',
                                   throttle_duration_sec=10.0)

    def on_scan(self, msg):
        try:
            pts = pc2.to_xyzi(msg)
        except Exception as exc:
            self.get_logger().warn(f'scan decode failed: {exc!r}',
                                   throttle_duration_sec=10.0)
            return
        if pts.shape[0] > 0:
            self.objects.add_scan(pts[::2])  # halve the rate of growth
            self.recent_scans.append(pts[:, :3].astype(np.float32))

    def on_image(self, msg):
        self.image_count += 1
        try:
            buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            img = buf[:msg.height * msg.width * 3].reshape(
                msg.height, msg.width, 3)
            # Keep latest_image in BGR, which Grounder converts to RGB before
            # passing it to PIL. ROS camera topics are usually rgb8, while
            # OpenCV-style relays may be bgr8.
            if msg.encoding.lower() == 'rgb8':
                img = img[:, :, ::-1]
            elif msg.encoding.lower() not in ('bgr8', '8uc3'):
                self.get_logger().warn(f'unhandled image encoding: {msg.encoding}')
            self.latest_image = img
            # Freeze the pose that goes with THIS image, not whatever the
            # robot is doing by the time a keyframe is built from it.
            self.latest_image_pose = (self.vehicle[0], self.vehicle[1],
                                      self.vehicle_z)
            self.latest_image_yaw = self.heading
            self.latest_image_t = self.elapsed
        except Exception as exc:
            self.get_logger().warn(f'image decode failed: {exc}')

    def _qtype(self):
        return self.parsed.type if self.parsed else 'unknown'

    # -- main loop -----------------------------------------------------------
    def tick(self):
        """Timer callback, guarded.

        An unhandled exception here does not just skip a cycle: rclpy lets it
        propagate out of the callback and through spin(), which tears the node
        down. A question with no message published scores zero, so any failure
        falls back to publishing a crude answer rather than dying silently.
        """
        try:
            self._tick()
        except Exception as exc:
            self.get_logger().error(f'tick failed: {exc!r}',
                                    throttle_duration_sec=5.0)
            self.emergency_answer()

    def emergency_answer(self):
        """Publish a valid message of the right type after a failure."""
        if self.answered:
            return
        self.answered = True
        try:
            qtype = self._qtype()
            if qtype == Q.NUMERICAL:
                self.pub_num.publish(Int32(data=2))
            elif qtype == Q.INSTRUCTION:
                self.route = list(self.visited[-3:]) or [self.vehicle]
            else:
                self.answer_marker = self.build_marker(
                    self.vehicle[0], self.vehicle[1], 0.5,
                    (0.5, 0.5, 0.5), 'object')
                self.pub_marker.publish(self.answer_marker)
            self.get_logger().warn(f'published fallback {qtype} answer')
        except Exception as exc:
            self.get_logger().error(f'fallback answer failed: {exc!r}')

    def _tick(self):
        if self.answered and not self.route:
            return

        t = self.elapsed
        self.maybe_capture_keyframe(t)
        if t < self.explore_deadline() and not self.answered:
            self.explore()
            return

        if not self.answered:
            self.commit_answer()

        self.drive_route()
        self.republish_marker()

        if t > self.budget - 2.0:
            # Fires on every tick once past the limit; throttle so the tail of
            # the log stays readable.
            self.get_logger().warn('time budget exhausted',
                                   throttle_duration_sec=10.0)

    def maybe_capture_keyframe(self, t):
        """Hand the detector a view whenever we have moved or waited enough."""
        if self.latest_image is None or not self.recent_scans:
            return
        if self.latest_image_pose is None:
            return
        if self.latest_image_t <= self.last_image_used:
            return          # no new frame since the last submission
        age = t - self.latest_image_t
        if age > IMAGE_MAX_AGE:
            # Nothing to gain from projecting a view whose pose we cannot trust.
            self.get_logger().info(
                f'skipping keyframe: image is {age:.1f}s old',
                throttle_duration_sec=20.0)
            return
        moved = (self.last_keyframe_xy is None or
                 math.hypot(self.vehicle[0] - self.last_keyframe_xy[0],
                            self.vehicle[1] - self.last_keyframe_xy[1]) > KEYFRAME_DIST)
        if not moved and (t - self.last_keyframe_t) < KEYFRAME_PERIOD:
            return

        pts = np.concatenate(list(self.recent_scans), axis=0)
        kf = Keyframe(self.latest_image, self.latest_image_pose,
                      self.latest_image_yaw, pts, t)
        if self.grounder.submit(kf):
            self.last_image_used = self.latest_image_t
            self.keyframes_sent += 1
            self.last_keyframe_t = t
            self.last_keyframe_xy = self.vehicle

    # -- exploration ---------------------------------------------------------
    def explore(self):
        if self.current_wp is not None and not self.reached(self.current_wp):
            self.publish_waypoint(*self.current_wp)
            return

        target = self.grid.next_frontier(self.vehicle, self.visited)
        if target is None:
            cov = self.grid.coverage()
            if cov <= self.last_coverage:
                self.stalled_ticks += 1
            else:
                self.stalled_ticks = 0
            self.last_coverage = cov
            # "No frontier" also describes an empty map, so only treat it as
            # full coverage once real terrain has arrived and settled.
            enough_map = cov >= MIN_COVERAGE_CELLS
            settled = self.stalled_ticks > 50          # ~10s without growth
            # Do not answer before perception has had a look at the scene:
            # the map can be complete while detection is still catching up.
            seen_enough = (not self.grounder.available
                           or self.grounder.frames_processed >= MIN_KEYFRAMES
                           or self.elapsed > self.explore_deadline() * 0.8)
            if enough_map and settled and seen_enough and self.elapsed > MIN_EXPLORE_TIME:
                self.get_logger().info(
                    f'scene appears covered (cov={cov}); answering early')
                self.commit_answer()
            elif not enough_map and self.elapsed > 30.0 and cov == 0:
                self.get_logger().warn(
                    'no terrain data after 30s - is the system running?',
                    throttle_duration_sec=10.0)
            else:
                # Frontier exhausted with budget left. Standing still only
                # re-observes one viewpoint, so circulate through the ones
                # already visited and keep feeding the detector new angles.
                self.resweep()
            return

        self.stalled_ticks = 0
        self.visited.append(target)
        self.current_wp = target
        self.publish_waypoint(*target)
        self.get_logger().info(
            f'frontier -> ({target[0]:.1f}, {target[1]:.1f}) '
            f'cov={self.grid.coverage()} vox={self.objects.n_voxels}')

    def resweep_targets(self, radius=3.0):
        """Viewpoints worth revisiting, nearest the objects being asked about.

        The detector manages roughly twenty keyframes in an exploration phase,
        so spending them evenly around the room wastes most of them on
        furniture nobody asked about. Prefer the viewpoints standing near a
        weakly-seen instance of a noun the question mentions: another look
        there is what turns a one-off box into a confident object with usable
        geometry.
        """
        nouns = {n for n in self._question_nouns()}
        if not nouns or not self.visited:
            return self.visited
        weak = [o for o in self.grounder.objects()
                if any(spatial.label_matches(o.label, n) for n in nouns)]
        if not weak:
            return self.visited
        weak.sort(key=lambda o: o.n_obs)          # least-seen first
        near = []
        for o in weak[:4]:
            for v in self.visited:
                if math.hypot(v[0] - o.center[0], v[1] - o.center[1]) < radius \
                        and v not in near:
                    near.append(v)
        return near or self.visited

    def resweep(self):
        """Revisit known viewpoints so detection keeps gaining observations."""
        if not self.visited:
            return
        if self.current_wp is None or self.reached(self.current_wp):
            pool = self.resweep_targets()
            self.sweep_i = (self.sweep_i + 1) % len(pool)
            self.current_wp = pool[self.sweep_i]
            self.get_logger().info(
                f'map covered; re-sweeping viewpoint {self.sweep_i + 1}'
                f'/{len(pool)} (of {len(self.visited)} visited) for more views',
                throttle_duration_sec=15.0)
        self.publish_waypoint(*self.current_wp)

    def reached(self, wp):
        dx, dy = self.vehicle[0] - wp[0], self.vehicle[1] - wp[1]
        return math.hypot(dx, dy) < self.reach

    # -- answering -----------------------------------------------------------
    def snap_to_geometry(self, grounded, cands, radius=1.0):
        """Take the label from the detector and the shape from the lidar.

        A grounded box is cut from the scan points whose bearing falls inside a
        2D detection, which is a cone: it inherits whatever lies behind the
        object and is far looser than the truth. The voxel clusters in
        ObjectMap are built from the same returns without that bias, so where a
        cluster of plausible size sits under a detection it is the better box -
        and object reference is scored on overlap with the true box.

        Guarded deliberately: a detection sitting over the table rather than
        the vase would otherwise adopt the table's geometry, so a cluster is
        only adopted when it is close AND comparable in size.
        """
        if not cands:
            return grounded
        snapped = 0
        for obj in grounded:
            oc = np.asarray(obj.center)
            ov = max(float(np.prod(obj.extent)), 1e-6)
            best, best_d = None, radius
            for c in cands:
                d = float(np.linalg.norm(np.asarray(c.center) - oc))
                if d >= best_d:
                    continue
                # Loose on size: a detection box is a bearing cone and its
                # volume is unreliable, so demanding a close match rejected
                # almost everything (1 of 33 on one run) and wasted the signal.
                if 0.1 <= c.volume / ov <= 10.0:
                    best, best_d = c, d
            if best is not None:
                obj.center = tuple(float(v) for v in best.center)
                obj.extent = tuple(float(v) for v in best.extent)
                obj.geom = True
                snapped += 1
        if snapped:
            self.get_logger().info(
                f'snapped {snapped}/{len(grounded)} detections onto lidar clusters')
        return grounded

    def commit_answer(self):
        self.answered = True
        cands = self.objects.candidates()
        grounded = self.snap_to_geometry(self.grounder.objects(), cands)
        self.get_logger().info(
            f'answering at t={self.elapsed:.0f}s: {len(grounded)} grounded '
            f'({self.grounder.frames_processed}/{self.keyframes_sent} keyframes), '
            f'{len(cands)} geometric candidates')
        if grounded:
            self.get_logger().info(
                'grounded: ' + ', '.join(
                    f'{o.label}@({o.center[0]:.1f},{o.center[1]:.1f}) '
                    f's={o.score:.2f}x{o.n_obs}' for o in grounded[:12]))

        qtype = self._qtype()
        self.grounded = grounded
        if qtype == Q.NUMERICAL:
            self.answer_numerical(cands)
        elif qtype == Q.INSTRUCTION:
            self.answer_instruction(cands)
        else:
            self.answer_object_reference(cands)

    def answer_numerical(self, cands):
        grounded = self.grounded
        if grounded and self.parsed and self.parsed.target:
            count = spatial.count_matching(
                self.parsed.target, self.parsed.relations, grounded)
        else:
            # Nothing detected, so fall back to the prior. Ground-truth counts
            # across the 15 training scenes are 2,6,6,2,4,3,3,8,2,2,6,2,6,1,3:
            # 2 is the mode (5 of 15), while 1 is correct only once.
            count = 2
        count = max(0, min(int(count), 20))
        self.pub_num.publish(Int32(data=int(count)))
        self.get_logger().info(f'numerical response: {count}')

    def answer_object_reference(self, cands):
        grounded = self.grounded
        target = None
        if grounded and self.parsed and self.parsed.target:
            target, score = spatial.resolve(
                self.parsed.target, self.parsed.relations, grounded)
            if target is not None:
                self.get_logger().info(
                    f'resolved "{self.parsed.target.text}" -> {target} '
                    f'(score {score:.2f})')
        if target is None:
            target = self.pick_candidate(cands)
        if target is None:
            self.get_logger().warn('no candidate found; marking vehicle pose')
            target = None
            cx, cy, cz = self.vehicle[0], self.vehicle[1], 0.5
            extent = (0.5, 0.5, 0.5)
            label = 'unknown'
        else:
            cx, cy, cz = target.center
            extent = target.extent
            label = target.label or (self.parsed.target.noun
                                     if self.parsed and self.parsed.target
                                     else 'object')

        self.answer_marker = self.build_marker(cx, cy, cz, extent, label)
        self.pub_marker.publish(self.answer_marker)
        # The marker centre doubles as the navigation goal for this task.
        self.route = [(cx, cy)]
        self.get_logger().info(
            f'object reference -> {label} at ({cx:.2f}, {cy:.2f}, {cz:.2f})')

    def answer_instruction(self, cands):
        """Sequence one waypoint per ordered step of the command."""
        route = []
        used = []
        self.forbidden = []
        grounded = self.grounded
        steps = self.parsed.steps if self.parsed else []
        for step in steps:
            gate = self.gateway(step, grounded)
            if step.kind == 'avoid':
                # Not a goal: remember it so the route can be pushed away.
                if gate is not None:
                    self.forbidden.append(gate)
                    self.get_logger().info(
                        f'avoid gateway at ({gate[0]:.1f}, {gate[1]:.1f})')
                continue
            if step.kind == 'through' and gate is not None:
                route.append(gate)
                self.get_logger().info(
                    f'step through -> gateway ({gate[0]:.1f}, {gate[1]:.1f})')
                continue
            pick = None
            if grounded:
                pick, _ = spatial.resolve(step.target, step.relations,
                                          grounded, exclude=tuple(used))
                if pick is not None:
                    used.append(pick)
                    self.get_logger().info(
                        f'step {step.kind} "{step.target.text}" -> {pick}')
            if pick is None:
                pick = self.pick_candidate(cands, exclude=route)
            if pick is not None:
                route.append(pick.xy)

        if not route:
            route = [v for v in self.visited[-3:]] or [self.vehicle]

        self.route = route
        self.get_logger().info(
            f'instruction route: {[(round(x,1), round(y,1)) for x, y in route]}')

    def gateway(self, step, grounded):
        """Midpoint of a 'between X and Y' path constraint, in the map frame.

        Path steps name a gap to drive through (or around), not an object, so
        the useful waypoint is the point between the two anchors.
        """
        if not grounded:
            return None
        for rel in step.relations:
            if rel.predicate != 'between' or not rel.anchors:
                continue
            if len(rel.anchors) >= 2:
                a = spatial.resolve_anchor(rel.anchors[0], grounded)
                b = spatial.resolve_anchor(rel.anchors[1], grounded, exclude=(a,))
            else:
                # "between the two columns" names one plural anchor rather than
                # two separate ones: take the two best objects of that class.
                pool = spatial.candidates_for(rel.anchors[0], grounded)
                pool = sorted(pool, key=spatial.confidence, reverse=True)
                if len(pool) < 2:
                    continue
                a, b = pool[0], pool[1]
            if a is None or b is None:
                continue
            return ((a.center[0] + b.center[0]) / 2.0,
                    (a.center[1] + b.center[1]) / 2.0)

        # "take the path near the TV" names one landmark to pass rather than a
        # gap to thread. Without this the step falls through to resolving the
        # word "path", which matches nothing and yields an arbitrary object.
        for rel in step.relations:
            if not rel.anchors:
                continue
            pool = spatial.class_members(rel.anchors[0], grounded)
            if pool:
                return max(pool, key=spatial.confidence).xy
        return None

    def detour(self, goal):
        """Offset a leg that would drive through a forbidden gateway."""
        for fx, fy in getattr(self, 'forbidden', ()):
            ax, ay = self.vehicle
            dx, dy = goal[0] - ax, goal[1] - ay
            seg = math.hypot(dx, dy)
            if seg < 1e-3:
                continue
            t = max(0.0, min(1.0, ((fx - ax) * dx + (fy - ay) * dy) / (seg * seg)))
            px, py = ax + t * dx, ay + t * dy
            if math.hypot(px - fx, py - fy) < AVOID_RADIUS:
                # Step perpendicular to the leg, away from the forbidden point.
                nx, ny = -dy / seg, dx / seg
                side = 1.0 if ((fx - px) * nx + (fy - py) * ny) < 0 else -1.0
                return (px + side * nx * AVOID_RADIUS * 1.5,
                        py + side * ny * AVOID_RADIUS * 1.5)
        return None

    def pick_candidate(self, cands, exclude=()):
        """Choose the most plausible object candidate.

        Without labels this favours graspable-scale clusters near the robot;
        the perception layer replaces this with a grounded match.
        """
        best, best_score = None, -1e18
        for c in cands:
            if any(abs(c.xy[0] - x) < 0.4 and abs(c.xy[1] - y) < 0.4
                   for x, y in exclude):
                continue
            vol = c.volume
            if vol < 0.001 or vol > 4.0:
                continue
            dx, dy = c.xy[0] - self.vehicle[0], c.xy[1] - self.vehicle[1]
            dist = math.hypot(dx, dy)
            score = math.log(c.n_points + 1) - 0.15 * dist
            if score > best_score:
                best, best_score = c, score
        return best

    # -- publishing ----------------------------------------------------------
    def build_marker(self, x, y, z, extent, label):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = str(label)
        m.id = 0
        m.action = Marker.ADD
        m.type = Marker.CUBE
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z)
        m.pose.orientation.w = 1.0
        m.scale.x = float(max(extent[0], 0.1))
        m.scale.y = float(max(extent[1], 0.1))
        m.scale.z = float(max(extent[2], 0.1))
        m.color.a = 0.5
        m.color.r = 0.0
        m.color.g = 0.0
        m.color.b = 1.0
        return m

    def republish_marker(self):
        if self.answer_marker is not None:
            self.answer_marker.header.stamp = self.get_clock().now().to_msg()
            self.pub_marker.publish(self.answer_marker)

    def publish_waypoint(self, x, y, theta=0.0):
        msg = Pose2D()
        msg.x = float(x)
        msg.y = float(y)
        msg.theta = float(theta)
        self.pub_wp.publish(msg)

    def drive_route(self):
        if not self.route:
            return
        goal = self.route[0]
        if self.leg_started is None:
            self.leg_started = self.elapsed

        via = self.detour(goal)
        if via is not None:
            self.publish_waypoint(*via)
            return

        arrived = self.reached(goal)
        stalled = (self.elapsed - self.leg_started) > LEG_TIMEOUT
        if arrived or stalled:
            if stalled and not arrived:
                self.get_logger().warn(
                    f'leg to ({goal[0]:.1f}, {goal[1]:.1f}) timed out after '
                    f'{LEG_TIMEOUT:.0f}s - advancing so later legs still run')
            self.route.pop(0)
            self.leg_started = None
            if self.route:
                self.get_logger().info(f'waypoint reached; {len(self.route)} left')
            else:
                self.get_logger().info('route complete')
            return
        self.publish_waypoint(*goal)


def main(args=None):
    rclpy.init(args=args)
    node = VLNNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # SIGINT/SIGTERM during shutdown is not an error
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
