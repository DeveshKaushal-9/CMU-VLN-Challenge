#!/usr/bin/env python3
"""Offline checks for the language and spatial layers.

Runs the parser over every released question, and - when the organizers'
ground-truth object lists are present under scenes/gt/ - replays the spatial
reasoning against perfect perception. That separates reasoning faults from
detection faults: anything failing here cannot work once OWL-ViT is in the
loop.

    python3 ai_module/test/test_reasoning.py

Exits non-zero if a parse crashes, a question is misclassified, or a resolved
object is of the wrong class.
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'ai_module', 'src', 'dummy_vlm'))

from dummy_vlm import question as Q, spatial  # noqa: E402

QUESTIONS = os.path.join(ROOT, 'questions', 'questions.json')
GT_DIR = os.path.join(ROOT, 'scenes', 'gt')

# Ground-truth counts for the one numerical question in each of the 15 training
# scenes, in questions.json order.
GT_COUNTS = [2, 6, 6, 2, 4, 3, 3, 8, 2, 2, 6, 2, 6, 1, 3]

_GT_LINE = re.compile(
    r'\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+"(.*)"\s*$')


class GtObject:
    """Stands in for a GroundedObject with perfect label and geometry."""

    __slots__ = ('label', 'center', 'extent', 'score', 'n_obs')

    def __init__(self, label, center, extent):
        self.label, self.center, self.extent = label, center, extent
        self.score, self.n_obs = 1.0, 1

    @property
    def xy(self):
        return self.center[0], self.center[1]

    @property
    def confidence(self):
        return 1.0

    def __repr__(self):
        return f'{self.label}@({self.center[0]:.1f},{self.center[1]:.1f})'


def load_gt(scene):
    path = os.path.join(GT_DIR, scene + '.txt')
    if not os.path.exists(path):
        return None
    objs = []
    with open(path) as fh:
        for line in fh:
            m = _GT_LINE.match(line)
            if m:
                g = m.groups()
                objs.append(GtObject(g[8],
                                     (float(g[1]), float(g[2]), float(g[3])),
                                     (float(g[4]), float(g[5]), float(g[6]))))
    return objs


EXPECTED = {'numerical': Q.NUMERICAL,
            'object_reference': Q.OBJECT_REFERENCE,
            'instruction_following': Q.INSTRUCTION}


def main():
    scenes = json.load(open(QUESTIONS))
    failures = []

    # -- parsing, which needs no scene data --------------------------------
    parsed_ok = 0
    for scene in scenes:
        for kind, questions in scene['questions'].items():
            for q in questions:
                try:
                    p = Q.ParsedQuestion(q)
                except Exception as exc:
                    failures.append(f'parse crash [{scene["scene"]}] {q}: {exc}')
                    continue
                if p.type != EXPECTED[kind]:
                    failures.append(
                        f'misclassified [{scene["scene"]}] want {EXPECTED[kind]} '
                        f'got {p.type}: {q}')
                elif p.type == Q.INSTRUCTION and not p.steps:
                    failures.append(f'no steps [{scene["scene"]}]: {q}')
                elif p.type != Q.INSTRUCTION and not (p.target and p.target.noun):
                    failures.append(f'no target [{scene["scene"]}]: {q}')
                else:
                    parsed_ok += 1
    print(f'parsing:              {parsed_ok}/75 questions clean')

    if not os.path.isdir(GT_DIR):
        print(f'\nscenes/gt not present - skipping spatial checks.')
        return 1 if failures else 0

    # -- spatial reasoning against perfect perception ----------------------
    ref_ok = ref_tot = 0
    step_ok = step_tot = 0
    gate_ok = gate_tot = 0
    num_ok = num_tot = 0

    for scene, want_count in zip(scenes, GT_COUNTS):
        objs = load_gt(scene['scene'])
        if not objs:
            continue

        for q in scene['questions']['object_reference']:
            ref_tot += 1
            p = Q.ParsedQuestion(q)
            pick, _ = spatial.resolve(p.target, p.relations, objs)
            if pick is not None and spatial.label_matches(pick.label, p.target.noun):
                ref_ok += 1
            else:
                failures.append(
                    f'object reference [{scene["scene"]}] {q} -> {pick} '
                    f'(want a {p.target.noun!r})')

        for q in scene['questions']['numerical']:
            num_tot += 1
            p = Q.ParsedQuestion(q)
            got = spatial.count_matching(p.target, p.relations, objs)
            num_ok += (got == want_count)

        for q in scene['questions']['instruction_following']:
            p = Q.ParsedQuestion(q)
            used = []
            for step in p.steps:
                if step.kind in ('through', 'avoid'):
                    if not any(r.predicate == 'between' for r in step.relations):
                        continue
                    gate_tot += 1
                    gate_ok += _gateway(step, objs) is not None
                    continue
                step_tot += 1
                pick, _ = spatial.resolve(step.target, step.relations, objs,
                                          exclude=tuple(used))
                if pick is not None and spatial.label_matches(pick.label,
                                                              step.target.noun):
                    step_ok += 1
                    used.append(pick)
                else:
                    failures.append(
                        f'instruction step [{scene["scene"]}] {step.target.text!r} '
                        f'-> {pick} in: {q}')

    print(f'object reference:     {ref_ok}/{ref_tot} resolved to the right class')
    print(f'instruction steps:    {step_ok}/{step_tot} resolved to the right class')
    print(f'between gateways:     {gate_ok}/{gate_tot} produced a midpoint')
    print(f'numerical:            {num_ok}/{num_tot} exactly correct '
          f'(colour-qualified counts are a known gap)')

    if failures:
        print('\nFAILURES')
        for f in failures:
            print('  -', f)
        return 1
    print('\nall structural checks passed')
    return 0


def _gateway(step, objs):
    """Mirror of VLNNode.gateway, so the test covers the same path."""
    for rel in step.relations:
        if rel.predicate != 'between' or not rel.anchors:
            continue
        if len(rel.anchors) >= 2:
            a = spatial.resolve_anchor(rel.anchors[0], objs)
            b = spatial.resolve_anchor(rel.anchors[1], objs, exclude=(a,))
        else:
            pool = sorted(spatial.candidates_for(rel.anchors[0], objs),
                          key=spatial.confidence, reverse=True)
            if len(pool) < 2:
                continue
            a, b = pool[0], pool[1]
        if a is not None and b is not None:
            return ((a.center[0] + b.center[0]) / 2.0,
                    (a.center[1] + b.center[1]) / 2.0)
    for rel in step.relations:
        if not rel.anchors:
            continue
        pool = spatial.class_members(rel.anchors[0], objs)
        if pool:
            return max(pool, key=spatial.confidence).xy
    return None


if __name__ == '__main__':
    sys.exit(main())
