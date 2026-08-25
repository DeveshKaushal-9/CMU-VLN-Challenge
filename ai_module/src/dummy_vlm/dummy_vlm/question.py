"""Parsing for the three challenge question types.

Across the 75 released training questions the spatial vocabulary is a closed
set of roughly ten predicates, so a rule-based parser covers the grammar
without needing a language model.
"""

import re

NUMERICAL = 'numerical'
OBJECT_REFERENCE = 'object_reference'
INSTRUCTION = 'instruction_following'

# Longest first so "closest to" is matched before "close"/"to".
RELATIONS = [
    'closest to', 'nearest to', 'farthest from', 'furthest from',
    'in front of', 'next to', 'on top of', 'between',
    'above', 'below', 'under', 'behind', 'near', 'with', 'on', 'in',
]

_REL_RE = re.compile(r'\b(' + '|'.join(re.escape(r) for r in RELATIONS) + r')\b')

# Ordered instruction steps.
_STEP_SPLIT = re.compile(
    r',?\s*\b(?:and\s+)?(?:then|first,?|stop at|stopping at|'
    r'stop by|avoiding|avoid|take the path|go to|go near|go through)\b', re.I)

_DETERMINERS = re.compile(r'^\s*(?:the|a|an|two|three|some)\s+', re.I)
# "...and then to the flowers near the jar" leaves the infinitive "to" on the
# chunk. \s+ keeps this off "two", "toaster", and friends.
_LEAD_TO = re.compile(r'^\s*to\s+', re.I)
# "the wall lamp that is between ..." -> "wall lamp between ...";
# "how many sofas are below ..." -> "sofas below ...".
_FILLER = re.compile(
    r'\b(?:that\s+is|that\s+are|which\s+is|which\s+are|'
    r'that|which|are\s+there|are|is)\b', re.I)
_HAS = re.compile(r'\b(?:that\s+has|which\s+has|has|have|having)\b', re.I)
_IT_ON = re.compile(r'\bon\s+it\b', re.I)
_PRONOUNS = {'it', 'them', 'they', 'one', 'itself'}
_TRAILING = re.compile(r'[.?!,]+\s*$')

COLORS = {'black', 'white', 'red', 'blue', 'green', 'yellow', 'orange',
          'purple', 'pink', 'brown', 'grey', 'gray', 'teal', 'beige',
          'silver', 'gold', 'wooden', 'glass', 'metal'}
SIZES = {'big', 'large', 'small', 'tall', 'short', 'long', 'little', 'tiny'}


def clean(text):
    text = _TRAILING.sub('', text.strip())
    text = _LEAD_TO.sub('', text)
    return _DETERMINERS.sub('', text).strip()


class Phrase:
    """A noun phrase plus the adjectives that narrow it."""

    def __init__(self, text):
        self.text = clean(text)
        words = re.findall(r'[a-z]+', self.text.lower())
        self.colors = [w for w in words if w in COLORS]
        self.sizes = [w for w in words if w in SIZES]
        # Head noun: last word before any relation clause.
        head = _REL_RE.split(self.text.lower())[0].strip()
        head_words = re.findall(r'[a-z]+', head)
        self.head = ' '.join(head_words)
        self.noun = _singular(head_words[-1]) if head_words else self.text.lower()
        # OWL-ViT does better with the object phrase than a bare head noun:
        # "trash can", "door frame", and "coffee table" are very different
        # from generic cans, frames, and tables.
        self.detector_label = self._detector_label(head_words)

    def __repr__(self):
        return f'<Phrase {self.text!r} noun={self.noun!r}>'

    def _detector_label(self, head_words):
        keep = [w for w in head_words if w not in COLORS and w not in SIZES]
        return ' '.join(keep[-3:]) if keep else self.noun


class Relation:
    def __init__(self, predicate, anchors):
        self.predicate = predicate
        self.anchors = anchors  # list of Phrase

    def __repr__(self):
        return f'<Rel {self.predicate} {self.anchors}>'


class Step:
    """One ordered leg of an instruction-following command."""

    def __init__(self, kind, target, relations):
        self.kind = kind          # 'goto' | 'stop' | 'through' | 'avoid'
        self.target = target      # Phrase
        self.relations = relations

    def __repr__(self):
        return f'<Step {self.kind} {self.target}>'


class ParsedQuestion:
    def __init__(self, raw):
        self.raw = raw
        self.type = classify(raw)
        self.target = None
        self.relations = []
        self.steps = []

        if self.type == INSTRUCTION:
            self.steps = _parse_steps(raw)
        else:
            body = re.sub(r'^\s*(?:how many|count the number of|find)\b', '',
                          raw, flags=re.I)
            body = re.sub(r'\b(?:are there|are|is)\b\s*$', '', clean(body))
            self.target, self.relations = _parse_target(body)

    def __repr__(self):
        return (f'<Question {self.type} target={self.target} '
                f'rels={self.relations} steps={self.steps}>')


# Instructions are the only type that command movement; object references
# may arrive either as "Find the ..." or as a bare noun phrase.
_MOTION = re.compile(
    r'\b(?:go|goes|going|take the path|takes|head|heads|navigate|'
    r'move|walk|proceed|drive|stop at|stopping at|avoid|avoiding|'
    r'travel|follow the path|first,)\b', re.I)


def classify(text):
    low = text.strip().lower()
    if low.startswith('how many') or 'count the number' in low:
        return NUMERICAL
    if low.startswith('find'):
        return OBJECT_REFERENCE
    if _MOTION.search(low):
        return INSTRUCTION
    return OBJECT_REFERENCE


def _parse_target(body):
    """Split a noun phrase into its head object and its spatial relations."""
    body = _HAS.sub(' with ', body)
    body = _IT_ON.sub(' ', body)
    body = clean(_FILLER.sub(' ', body))
    body = re.sub(r'\s{2,}', ' ', body).strip()
    match = _REL_RE.search(body.lower())
    if not match:
        return Phrase(body), []

    target = Phrase(body[:match.start()])
    relations = []
    rest = body[match.start():]

    while True:
        m = _REL_RE.search(rest.lower())
        if not m:
            break
        pred = m.group(1)
        rest = rest[m.end():]
        nxt = _REL_RE.search(rest.lower())
        # Keep the trailing clause attached to the final anchor.
        chunk = rest if nxt is None else rest[:nxt.start()]

        if pred == 'between':
            parts = re.split(r'\band\b', chunk, maxsplit=1)
            anchors = [Phrase(p) for p in parts if clean(p)]
        else:
            anchors = [Phrase(chunk)] if clean(chunk) else []

        anchors = [a for a in anchors if a.noun not in _PRONOUNS]
        if anchors:
            relations.append(Relation(pred, anchors))
        if nxt is None:
            break
        rest = rest[nxt.start():]
        # Nested relation belongs to the previous anchor, not the target.
        m2 = _REL_RE.search(rest.lower())
        if m2 is None:
            break

    return target, relations


def _singular(word):
    for suffix, repl in (('ies', 'y'), ('sses', 'ss'), ('xes', 'x'),
                         ('hes', 'h'), ('s', '')):
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[:-len(suffix)] + repl
    return word


def _kind_for(marker):
    marker = marker.lower()
    if 'avoid' in marker:
        return 'avoid'
    if 'stop' in marker:
        return 'stop'
    if 'path' in marker or 'through' in marker:
        return 'through'
    return 'goto'


def _parse_steps(raw):
    """Break an instruction into ordered legs with their constraint kinds."""
    text = _TRAILING.sub('', raw.strip())
    # "pass by" is a navigation leg in the released grammar.
    text = re.sub(r'\bpass by\b', 'go near', text, flags=re.I)
    text = re.sub(r'\band\s+finally,?\s+to\b', 'then go to', text, flags=re.I)
    markers = [m.group(0) for m in _STEP_SPLIT.finditer(text)]
    chunks = _STEP_SPLIT.split(text)

    steps = []
    # chunks[0] is whatever preceded the first marker; usually empty.
    tail = chunks[0].strip(' ,')
    if tail and not markers:
        target, rels = _parse_target(tail)
        return [Step('goto', target, rels)]
    if tail:
        # A leading leg no marker introduced, as in "Go between the bench and
        # the bed and stop at ...". Dropping it loses a whole waypoint.
        lead = _lead_step(tail)
        if lead is not None:
            steps.append(lead)

    for marker, chunk in zip(markers, chunks[1:]):
        chunk = chunk.strip(' ,')
        if not chunk:
            continue
        kind = _kind_for(marker)
        extra_goal = None
        if kind == 'through':
            # "take the path between X and Y to Z" contains both a gateway
            # waypoint and a destination. Keep both, in order.
            m = _destination_to(chunk)
            if m is not None:
                extra_goal = m.group(1).strip()
                chunk = chunk[:m.start()].strip(' ,')
        target, rels = _parse_target(chunk)
        if not target.text:
            if kind in ('through', 'avoid') and rels:
                target = Phrase('path')
            else:
                continue
        steps.append(Step(kind, target, rels))
        if extra_goal:
            goal, goal_rels = _parse_target(extra_goal)
            if goal.text:
                steps.append(Step('goto', goal, goal_rels))
    return steps


def _lead_step(chunk):
    """Parse a navigation leg that appears before the first step marker.

    The marker list covers "go to"/"go near"/"go through" but not every verb
    phrase, so a command may open with a leg the splitter never sees.
    """
    m = re.match(r'\s*(?:go|goes|going|head|heads|move|walk|proceed|drive|'
                 r'travel|navigate)\b\s*', chunk, re.I)
    if m is None:
        return None
    rest = chunk[m.end():].strip(' ,')
    if not rest:
        return None
    target, rels = _parse_target(rest)
    if not target.text:
        if not rels:
            return None
        target = Phrase('path')
    # "go between X and Y" names a gap to drive through, not an object to reach.
    kind = 'through' if any(r.predicate == 'between' for r in rels) else 'goto'
    return Step(kind, target, rels)


def _destination_to(chunk):
    """Find the destination "to" in a path clause, not "closest to" etc."""
    for m in re.finditer(r'\bto\s+(?:the\s+)?(.+)$', chunk, flags=re.I):
        before = chunk[:m.start()].rstrip().lower()
        prev = re.findall(r'[a-z]+', before[-20:])
        if prev and prev[-1] in {'closest', 'nearest', 'farthest', 'furthest'}:
            continue
        return m
    return None
