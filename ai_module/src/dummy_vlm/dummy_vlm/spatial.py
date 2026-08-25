"""Spatial predicates over grounded objects.

Each predicate scores every candidate against the relation's anchors; higher is
better. Scores are combined additively across the relations attached to a
phrase, so a target satisfying several constraints outranks one satisfying
only the first.
"""

import numpy as np

NEAR_RADIUS = 2.0       # metres, what "near" means in a room-scale scene
SUPPORT_GAP = 0.45      # max vertical gap for "on"
OVERLAP_EPS = 1e-6


def singular(word):
    """Crude singulariser; the object vocabulary is all common nouns."""
    w = word.lower()
    for suffix, repl in (('ies', 'y'), ('ses', 's'), ('xes', 'x'),
                         ('hes', 'h'), ('s', '')):
        if w.endswith(suffix) and len(w) > len(suffix) + 1:
            return w[:-len(suffix)] + repl
    return w


def _loose_eq(a, b):
    return a == b or a in b or b in a


def label_matches(label, noun):
    """Loose match between a detector label and a parsed head noun."""
    if not label or not noun:
        return False
    a, b = singular(label), singular(noun)
    if _loose_eq(a, b):
        return True
    # Open and closed compounds are the same word: a scene labelled
    # "night stand" answers a question asking for a "nightstand".
    return _loose_eq(a.replace(' ', ''), b.replace(' ', ''))


def phrase_label_matches(label, phrase):
    """Prefer exact compound labels when the parser kept one.

    Anchored on the head noun, so a modifier cannot match on its own: without
    that, "wall lamp" substring-matches an object labelled "wall", "paper cup"
    matches "paper", and the wrong object wins before the head-noun pass runs.
    """
    det = getattr(phrase, 'detector_label', '') or ''
    if not det:
        return False
    a = singular((label or '').lower())
    b = singular(det.lower())
    if not (_loose_eq(a, b) or _loose_eq(a.replace(' ', ''), b.replace(' ', ''))):
        return False
    return label_matches(label, getattr(phrase, 'noun', '') or b)


def centre(obj):
    return np.asarray(obj.center, dtype=np.float64)


def horiz_dist(a, b):
    ca, cb = centre(a), centre(b)
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))


def dist(a, b):
    return float(np.linalg.norm(centre(a) - centre(b)))


def xy_overlap(a, b):
    """Fraction of the smaller footprint that overlaps the other, in XY."""
    ca, cb = centre(a), centre(b)
    ea = np.asarray(a.extent, dtype=np.float64)
    eb = np.asarray(b.extent, dtype=np.float64)
    lo = np.maximum(ca[:2] - ea[:2] / 2, cb[:2] - eb[:2] / 2)
    hi = np.minimum(ca[:2] + ea[:2] / 2, cb[:2] + eb[:2] / 2)
    inter = np.prod(np.maximum(hi - lo, 0.0))
    smaller = min(np.prod(ea[:2]), np.prod(eb[:2]))
    return float(inter / max(smaller, OVERLAP_EPS))


def top_z(obj):
    return centre(obj)[2] + obj.extent[2] / 2.0


def bottom_z(obj):
    return centre(obj)[2] - obj.extent[2] / 2.0


# -- individual predicates --------------------------------------------------

def score_closest(cand, anchors):
    return -min(dist(cand, a) for a in anchors)


def score_farthest(cand, anchors):
    return min(dist(cand, a) for a in anchors)


def score_near(cand, anchors):
    d = min(dist(cand, a) for a in anchors)
    # Flat reward inside the radius, decaying outside, so several genuinely
    # near objects are not split hairs over.
    return -max(0.0, d - NEAR_RADIUS) - 0.1 * d


def score_on(cand, anchors):
    """cand rests on anchor.

    Deliberately does NOT require the candidate to clear the anchor's top face:
    a pillow on a sofa sits *inside* the sofa's bounding box, because that box
    includes the backrest. What actually distinguishes "on" is sitting within
    the anchor's footprint at a higher centre height.
    """
    best = -1e6
    for a in anchors:
        overlap = xy_overlap(cand, a)
        dz = centre(cand)[2] - centre(a)[2]
        hd = horiz_dist(cand, a)
        s = (3.0 * overlap
             + (0.8 if dz > -0.05 else -2.0)
             - 0.4 * abs(dz)
             - 2.0 * max(0.0, hd - NEAR_RADIUS * 0.75))
        best = max(best, s)
    return best


def score_above(cand, anchors):
    best = -1e6
    for a in anchors:
        dz = centre(cand)[2] - centre(a)[2]
        hd = horiz_dist(cand, a)
        s = (1.0 if dz > 0 else -1.0) * min(abs(dz), 2.0) \
            + 0.5 * xy_overlap(cand, a) - 0.2 * hd \
            - 2.0 * max(0.0, hd - NEAR_RADIUS)
        best = max(best, s)
    return best


def score_below(cand, anchors):
    best = -1e6
    for a in anchors:
        dz = centre(a)[2] - centre(cand)[2]
        hd = horiz_dist(cand, a)
        s = (1.0 if dz > 0 else -1.0) * min(abs(dz), 2.0) \
            + 0.5 * xy_overlap(cand, a) - 0.2 * hd \
            - 2.0 * max(0.0, hd - NEAR_RADIUS)
        best = max(best, s)
    return best


def score_between(cand, anchors):
    """cand lies on the segment joining two anchors."""
    if len(anchors) < 2:
        return score_near(cand, anchors)
    p = centre(cand)[:2]
    a = centre(anchors[0])[:2]
    b = centre(anchors[1])[:2]
    ab = b - a
    length = float(np.linalg.norm(ab))
    if length < OVERLAP_EPS:
        return -float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / (length * length), 0.0, 1.0))
    perp = float(np.linalg.norm(p - (a + t * ab)))
    # Penalise being off the segment and being pushed toward either end.
    return -perp - 2.0 * abs(t - 0.5) * 0.5


def score_with(cand, anchors):
    """"table with the figurine on it" - the anchor sits on the candidate."""
    return score_on_inverted(cand, anchors)


def score_on_inverted(cand, anchors):
    """The anchor rests on the candidate - "table with a figurine on it"."""
    best = -1e6
    for a in anchors:
        overlap = xy_overlap(cand, a)
        dz = centre(a)[2] - centre(cand)[2]
        hd = horiz_dist(cand, a)
        s = (3.0 * overlap
             + (0.8 if dz > -0.05 else -2.0)
             - 0.4 * abs(dz)
             - 2.0 * max(0.0, hd - NEAR_RADIUS * 0.75))
        best = max(best, s)
    return best


PREDICATES = {
    'closest to': score_closest,
    'nearest to': score_closest,
    'farthest from': score_farthest,
    'furthest from': score_farthest,
    'near': score_near,
    'next to': score_near,
    'in front of': score_near,
    'behind': score_near,
    'between': score_between,
    'on': score_on,
    'on top of': score_on,
    'in': score_on,
    'above': score_above,
    'below': score_below,
    'under': score_below,
    'with': score_with,
}


SUPERLATIVES = {'closest to', 'nearest to', 'farthest from', 'furthest from'}
ANCHOR_MODIFIERS = SUPERLATIVES | {'near', 'next to', 'on', 'on top of', 'with',
                                   'above', 'below', 'under'}


def attach_anchor_modifiers(relations):
    """Group relations, attaching trailing clauses to the anchor they modify.

    "the monitors on the table closest to the door" parses flat, as though the
    monitors were both on a table and closest to a door. The superlative
    actually selects *which table*. The same pattern appears with non-
    superlative modifiers such as "lamp on the nightstand with a photo".

    Returns a list of (predicate, [(anchor_phrase, [modifier_relations])]).
    """
    grouped = []
    for rel in relations:
        if rel.predicate in ANCHOR_MODIFIERS and grouped:
            pred, anchors = grouped[-1]
            phrase, mods = anchors[-1]
            anchors[-1] = (phrase, mods + [rel])
            grouped[-1] = (pred, anchors)
        else:
            grouped.append((rel.predicate, [(a, []) for a in rel.anchors]))
    return grouped


def ground_anchor(phrase, objects, modifiers=(), exclude=()):
    """Resolve an anchor, optionally disambiguated by trailing modifiers."""
    pool = [o for o in candidates_for(phrase, objects) if o not in exclude]
    if not pool:
        return None
    if not modifiers:
        return max(pool, key=confidence)

    best, best_score = None, -1e18
    for cand in pool:
        total = 0.2 * confidence(cand)
        usable = False
        for mod in modifiers:
            if mod.predicate == 'between':
                # Ordered pair: expanding these would scramble the two ends.
                refs = [ground_anchor(a, objects, exclude=(cand,)) for a in mod.anchors]
                refs = [r for r in refs if r is not None]
            else:
                # A bare modifier anchor means its whole class - "the sofa
                # under the pictures" holds for a sofa under any picture - and
                # every predicate below already maxes over the anchor list.
                refs = [m for a in mod.anchors
                        for m in class_members(a, objects) if m is not cand]
                if not refs:
                    refs = [r for r in (ground_anchor(a, objects, exclude=(cand,))
                                        for a in mod.anchors) if r is not None]
            if not refs:
                continue
            fn = PREDICATES.get(mod.predicate)
            if fn is None:
                continue
            usable = True
            total += fn(cand, refs)
        if usable and total > best_score:
            best, best_score = cand, total
    return best if best is not None else max(pool, key=confidence)


def ground_relations(relations, objects, exclude=()):
    """Ground every relation's anchors, honouring attached modifiers.

    A bare anchor stands for its whole class: "the bowl on the table" means the
    bowl on ANY table, and every predicate already maxes over the anchor list,
    so handing it the class is what makes that reading work. Grounding to one
    arbitrary instance instead scores every candidate against whichever table
    happened to sort first - which picks the bowl nearest that table rather
    than the bowl actually resting on one.

    Two exceptions keep their single instance: a modified anchor ("the sofa
    under the pictures") names one object by construction, and 'between' names
    an ordered pair, whose two ends must not be collapsed into one class list.
    """
    out = []
    for pred, anchor_specs in attach_anchor_modifiers(relations):
        anchors = []
        for ap, mods in anchor_specs:
            if mods or pred == 'between':
                a = ground_anchor(ap, objects, mods, exclude=exclude)
                if a is None:
                    a = ground_anchor(ap, objects, mods)
                if a is not None:
                    anchors.append(a)
            else:
                pool = [o for o in class_members(ap, objects) if o not in exclude]
                anchors.extend(pool or class_members(ap, objects))
        if anchors:
            out.append((pred, anchors))
    return out


# -- membership tests, for counting ----------------------------------------
# Counting asks "does this object satisfy the relation?", which the ranking
# scores above cannot answer. They are comparative, not absolute, and several
# are negative for every candidate - score_near is -max(0, d-R) - 0.1*d, so it
# never reaches zero - which means any fixed threshold over them counts nothing.

def _rests_on(cand, anchor):
    """cand sits within anchor's footprint, no lower than the anchor centre."""
    return (xy_overlap(cand, anchor) > 0.10
            and centre(cand)[2] - centre(anchor)[2] > -0.10)


def satisfies(pred, cand, anchors):
    """Whether cand stands in relation `pred` to any of `anchors`."""
    anchors = [a for a in anchors if a is not cand]
    if not anchors:
        return True
    if pred in ('near', 'next to', 'in front of', 'behind'):
        return min(dist(cand, a) for a in anchors) <= NEAR_RADIUS
    if pred in ('on', 'on top of', 'in'):
        return any(_rests_on(cand, a) for a in anchors)
    if pred == 'with':
        return any(_rests_on(a, cand) for a in anchors)
    if pred == 'above':
        return any(centre(cand)[2] - centre(a)[2] > 0.05
                   and horiz_dist(cand, a) <= NEAR_RADIUS for a in anchors)
    if pred in ('below', 'under'):
        return any(centre(a)[2] - centre(cand)[2] > 0.05
                   and horiz_dist(cand, a) <= NEAR_RADIUS for a in anchors)
    if pred == 'between':
        return score_between(cand, anchors) > -1.0
    # Superlatives select which anchor is meant; they do not filter the pool.
    return True


def class_members(phrase, objects):
    """Every object of a phrase's class, most specific label first.

    A label equal to the whole phrase beats one that merely contains the head
    noun, so "coffee table" grounds to the coffee table rather than whichever
    plain "table" happens to come first. Empty when the class is absent.
    """
    det = singular((getattr(phrase, 'detector_label', '') or '').lower())
    if det:
        literal = [o for o in objects
                   if singular((getattr(o, 'label', '') or '').lower()) == det]
        if literal:
            return literal
    exact = [o for o in objects if phrase_label_matches(getattr(o, 'label', None), phrase)]
    if exact:
        return exact
    return [o for o in objects if label_matches(getattr(o, 'label', None), phrase.noun)]


def candidates_for(phrase, objects):
    """class_members, falling back to everything so a pick is always possible."""
    return class_members(phrase, objects) or list(objects)


def confidence(obj):
    """Prefer the grounder's view-reinforced confidence when it exposes one."""
    return float(getattr(obj, 'confidence', getattr(obj, 'score', 0.0)))


def resolve_anchor(phrase, objects, exclude=()):
    """Best single object for an anchor phrase, ignoring its own sub-relations."""
    pool = [o for o in candidates_for(phrase, objects) if o not in exclude]
    if not pool:
        return None
    return max(pool, key=confidence)


def resolve(phrase, relations, objects, exclude=()):
    """Pick the object best satisfying a phrase and its spatial relations.

    Returns (object, score) or (None, 0.0) when nothing matches the noun.
    """
    pool = [o for o in candidates_for(phrase, objects) if o not in exclude]
    if not pool:
        return None, 0.0
    if not relations:
        best = max(pool, key=confidence)
        return best, confidence(best)

    grounded = ground_relations(relations, objects, exclude=tuple(pool))

    if not grounded:
        best = max(pool, key=confidence)
        return best, confidence(best)

    best, best_total = None, -1e18
    for cand in pool:
        total = 1.2 * confidence(cand)
        for pred, anchors in grounded:
            fn = PREDICATES.get(pred)
            if fn is None:
                continue
            if cand in anchors:
                total -= 5.0      # an object cannot be its own anchor
                continue
            total += fn(cand, anchors)
        if total > best_total:
            best, best_total = cand, total
    return best, best_total


def count_matching(phrase, relations, objects):
    """How many objects plausibly satisfy the phrase - the numerical answer."""
    pool = [o for o in objects
            if label_matches(getattr(o, 'label', None), phrase.noun)]
    if not pool:
        return 2  # nothing detected; 2 is the modal ground-truth count
    if not relations:
        return len(pool)

    grounded = ground_relations(relations, objects)
    if not grounded:
        return len(pool)

    n = sum(1 for cand in pool
            if all(satisfies(pred, cand, anchors)
                   for pred, anchors in grounded))
    # Every candidate failing usually means the anchor was grounded to the
    # wrong instance, in which case the unfiltered pool is the better guess.
    return n or len(pool)
