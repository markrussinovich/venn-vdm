"""Recompose the two motion-led review cuts from curated numeric evidence.

    python examples/same_memory_video/render.py --output /tmp/same-memory-motion

No training, network calls, third-party music or source speech playback.
"""
import argparse
import ast
import hashlib
import json
import math
import re
import subprocess
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/media/same-memory"
BG, INK, MUTED = "#0c141b", "#f6f8f9", "#a9b7be"
GREEN, GOLD, BLUE, LINE, RED = "#8ce8ce", "#ffbf70", "#7899ce", "#29434d", "#f58d93"
CAPTION = "#b7c8e3"
FPS = 30
EXAMPLE_STATE = "Please refund this charge"
EXAMPLE_QUESTION = "What does the customer want?"
EXAMPLE_OPTIONS = ("Refund", "Replacement", "Information")


def load_evidence():
    evidence = {beat: json.loads((EVIDENCE / f"{beat}-eval.json").read_text())
                for beat in ("history", "sec", "wine")}
    record = json.loads((EVIDENCE / "head-execution.json").read_text())
    if record["status"] != "executed" or record["source_sha256"] != sha(Path(__file__).with_name("head_demo.py")):
        raise ValueError("Execute the displayed head_demo.py before rendering")
    evidence["head"] = record
    return evidence


@lru_cache(maxsize=1)
def head_code():
    source = Path(__file__).with_name("head_demo.py").read_text()
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "register_head")
    return ast.get_source_segment(source, function.body[0].value).splitlines()


def shot(start, end, kind, card, **data):
    if len(card.split()) > 8:
        raise ValueError(f"Card exceeds eight words: {card}")
    return dict(start=start, end=end, kind=kind, card=card, **data)


def timelines(_e):
    return {
        "landscape": [
            shot(0, 4, "setup", "Both choose from answers supplied with your question"),
            shot(4, 13, "selection", "Jev returns a selected answer and its probabilities", model="Jev"),
            shot(13, 24, "selection", "Venn scores answers using numeric representations of meaning", model="Venn"),
            shot(24, 28, "history", "Now predict outcomes from complaints to this company", mode="question"),
            shot(28, 34, "history", "Use Jev's label to group past outcomes", mode="baseline"),
            shot(34, 42, "history", "Venn retrieves similar complaints with recorded outcomes", mode="retrieve"),
            shot(42, 48, "history", "Evaluate outcome rankings on previously unseen complaints", mode="primary"),
            shot(48, 52, "history", "Now compare only complaints requesting a refund", mode="refund"),
            shot(52, 57, "history", "Imagine removing the complaint archive", mode="wipe"),
            shot(57, 64, "sec", "Now learn industry labels from company filings", mode="learn"),
            shot(64, 69, "sec", "Test both methods on the cookbook's filings", mode="sample"),
            shot(69, 75, "sec", "Then test on additional held-out company filings", mode="confirmation"),
            shot(75, 81, "sec", "Inspect one filing's predictions", mode="example"),
            shot(81, 89, "sec", "Venn learns industry labels from past filings", mode="neighbors"),
            shot(89, 94, "work", "Predict the critic's score from wine tasting notes", mode="question"),
            shot(94, 102, "work", "Jev builds features by answering questions about notes", mode="flow"),
            shot(102, 116, "code", "Venn fits a prediction head on stored embeddings"),
            shot(116, 122, "work", "Evaluate critic scores on previously unseen wine notes", mode="accuracy"),
            shot(122, 127, "work", "Both learn from the same 1,200 training notes", mode="time"),
            shot(127, 134, "price", "Price uses a different, already-stored embedding cache", mode="question"),
            shot(134, 140, "price", "Venn's time excludes the already-completed embedding step", mode="time"),
            shot(140, 143, "price", "Local predictions require no additional API charges", mode="cost"),
            shot(143, 147, "scoreboard", "Additional tests favor Jev on these reading tasks"),
            shot(147, 150, "end", "Build decisions from your own data"),
        ],
        "vertical": [
            shot(0, 4, "setup", "Both choose from answers supplied with your question"),
            shot(4, 13, "selection", "Jev returns a selected answer and its probabilities", model="Jev"),
            shot(13, 25, "selection", "Venn scores answers using numeric representations of meaning", model="Venn"),
            shot(25, 29, "history", "Now predict outcomes from complaints to this company", mode="question"),
            shot(29, 36, "history", "Use Jev's label to group past outcomes", mode="baseline"),
            shot(36, 44, "history", "Venn retrieves similar complaints with recorded outcomes", mode="retrieve"),
            shot(44, 50, "history", "Evaluate outcome rankings on previously unseen complaints", mode="primary"),
            shot(50, 54, "history", "Imagine removing the complaint archive", mode="wipe"),
            shot(54, 57, "history", "Restore those cases to make retrieval useful again", mode="restore"),
            shot(57, 60, "end", "Build decisions from your own data"),
        ],
    }


def ease(t):
    t = min(1., max(0., t))
    return t * t * (3 - 2 * t)


def mix(a, b, t):
    return a + (b - a) * t


def rgb(color):
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


@lru_cache(maxsize=100)
def font(size, bold=False, mono=False):
    names = (["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"] if mono else [
        f"/usr/share/fonts/truetype/lato/Lato-{'Bold' if bold else 'Regular'}.ttf",
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"])
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    raise RuntimeError("Install Lato or DejaVu fonts")


@lru_cache(maxsize=2)
def backdrop(vertical):
    w, h = (720, 1280) if vertical else (1280, 720)
    y, x = np.mgrid[:h, :w]
    light = np.exp(-((x / w - .48) ** 2 + (y / h - .43) ** 2) * 5)
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for i, (low, high) in enumerate(((7, 16), (13, 33), (20, 43))):
        image[:, :, i] = low + (high - low) * light
    return Image.fromarray(image)


class Canvas:
    def __init__(self, vertical, t):
        self.vertical, self.t = vertical, t
        self.image = backdrop(vertical).copy()
        self.w, self.h = self.image.size
        self.d = ImageDraw.Draw(self.image)
        self.texts = []
        self.text_boxes = []
        self.numbers = []
        for i in range(52):
            x = ((i * 197 + t * (9 + i % 5)) % (self.w + 100)) - 50
            y = ((i * 107 - t * (3 + i % 3)) % self.h)
            self.d.ellipse((x, y, x + 2, y + 2), fill=LINE)

    def text(self, x, y, value, size=36, color=INK, bold=False, anchor="mm", mono=False, bounds=None):
        if size < 32:
            raise ValueError(f"No fine print: {value}")
        f = font(size, bold, mono)
        box = self.d.textbbox((x, y), value, font=f, anchor=anchor)
        if box[0] < 28 or box[1] < 20 or box[2] > self.w - 28 or box[3] > self.h - 20:
            raise ValueError(f"Text outside safe area: {value!r} {box}")
        if bounds is not None and (box[0] < bounds[0] or box[1] < bounds[1]
                                   or box[2] > bounds[2] or box[3] > bounds[3]):
            raise ValueError(f"Text outside its container: {value!r} {box} {bounds}")
        self.d.text((x + 1, y + 2), value, font=f, fill="#02070c", anchor=anchor)
        self.d.text((x, y), value, font=f, fill=color, anchor=anchor)
        self.texts.append(value)
        self.text_boxes.append((value, box))
        self.numbers.extend(re.findall(r"(?<![A-Za-z])\d[\d,.]*(?:%)?", value))

    def check_text_overlap(self):
        for i, (name, a) in enumerate(self.text_boxes):
            for other, b in self.text_boxes[i + 1:]:
                if a[0] < b[2] + 3 and a[2] + 3 > b[0] and a[1] < b[3] + 3 and a[3] + 3 > b[1]:
                    raise ValueError(f"Overlapping text: {name!r}, {other!r}")

    def answer_box(self, x, y, value, color, active=False):
        half_width, half_height = 240, 29
        self.d.rounded_rectangle((x - half_width, y - half_height, x + half_width, y + half_height),
                                 radius=12, fill="#1b303a" if active else BG,
                                 outline=color if active else LINE, width=3)
        self.text(x, y, value, 36, color if active else INK, True,
                  bounds=(x - half_width + 20, y - half_height + 8,
                          x + half_width - 20, y + half_height - 8))

    def caption(self, value):
        words, lines, line = value.split(), [], ""
        size = 38 if self.vertical else 40
        top = self.h - (156 if self.vertical else 124)
        for graphic, box in self.text_boxes:
            if box[3] > top - 12:
                raise ValueError(f"Graphic text enters narration area: {graphic!r}")
        for word in words:
            candidate = (line + " " + word).strip()
            if self.d.textlength(candidate, font=font(size)) > self.w - 112:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
        if len(lines) == 2:
            candidates = [(" ".join(words[:i]), " ".join(words[i:])) for i in range(1, len(words))]
            widths = [(pair, [self.d.textlength(part, font=font(size)) for part in pair])
                      for pair in candidates]
            lines = list(min((pair for pair, sizes in widths if max(sizes) <= self.w - 112),
                             key=lambda pair: abs(self.d.textlength(pair[0], font=font(size))
                                                  - self.d.textlength(pair[1], font=font(size)))))
        self.d.rectangle((0, top, self.w, self.h), fill=BG)
        self.d.line((56, top, self.w - 56, top), fill=LINE, width=1)
        boxes = [font(size).getbbox(line, anchor="mm") for line in lines]
        block_top = min(box[1] + i * 44 for i, box in enumerate(boxes))
        block_bottom = max(box[3] + i * 44 for i, box in enumerate(boxes))
        y = (top + 16 + self.h - 24 - block_top - block_bottom) / 2
        for line in lines:
            self.text(self.w / 2, y, line, size, CAPTION,
                      bounds=(56, top + 16, self.w - 56, self.h - 24))
            y += 44

    def ring(self, x, y, radius, color=GREEN, width=2):
        self.d.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=width)

    def dot(self, x, y, radius=5, color=GREEN):
        self.d.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    def trail(self, points, color=GREEN, width=3):
        if len(points) > 1:
            self.d.line(points, fill=color, width=width, joint="curve")

    def document(self, x, y, scale=1, color=BLUE, lean=0, hero=False):
        w, h = 70 * scale, 94 * scale
        slant = lean * h
        points = [(x - w / 2 + slant, y - h / 2), (x + w / 2 + slant, y - h / 2),
                  (x + w / 2 - slant, y + h / 2), (x - w / 2 - slant, y + h / 2)]
        shadow = [(a + 9 * scale, b + 12 * scale) for a, b in points]
        self.d.polygon(shadow, fill="#040b11")
        self.d.polygon(points, fill="#1b303a", outline=color, width=max(1, round(2 * scale)))
        for j in range(3):
            yy = y - h / 5 + j * h / 6
            self.d.line((x - w * .28, yy, x + w * .24, yy), fill=color, width=max(1, round(scale)))
        if hero:
            self.ring(x, y, max(w, h) * .76, GOLD, 3)

    def pulse(self, x, y, t, color=GREEN, size=90):
        for i in range(3):
            p = (t * .55 + i / 3) % 1
            self.ring(x, y, 15 + p * size, color if p < .55 else LINE, 2)


def orbit(c, t, color=BLUE, radius=None):
    radius = radius or min(c.w * .38, 330)
    cx, cy = c.w / 2, c.h * .46
    for i in range(84):
        a = i * 2.39996 + t * .075
        r = radius * math.sqrt((i + 1) / 84)
        depth = .65 + .35 * math.sin(a + t * .14)
        x = cx + math.cos(a) * r
        y = cy + math.sin(a) * r * (.85 if c.vertical else .5)
        c.dot(x, y, 2 + depth * 3, color if i % 6 == 0 else LINE)


def conveyor(c, t, zoom=False):
    for i in range(52):
        z = (i / 9 - t * .6) % 6
        scale = .3 + (1 - z / 6) ** 2 * 1.8
        lane = (i % 7 - 3)
        x = c.w / 2 + lane * 115 * scale
        y = c.h * .45 + (z - 2.6) * (105 if c.vertical else 68) * scale
        if -120 < x < c.w + 120 and -150 < y < c.h + 150:
            c.document(x, y, scale, GREEN if (i + int(t * 3)) % 5 else BLUE, lean=math.sin(t + i) * .04)
    beam = c.w * (.2 + .6 * ((t * .33) % 1))
    c.d.line((beam, c.h * .15, beam, c.h * .77), fill=GREEN, width=3)
    if zoom:
        scale = 2.2 + ease(t / 3) * .5
        c.document(c.w / 2, c.h * .43, scale, GOLD, lean=math.sin(t * .7) * .015, hero=True)
        c.text(c.w / 2, c.h * .43 + 120, "Refund", 58, GOLD, True)


def title(c, t):
    orbit(c, t, GREEN)
    p = ease(t / 2.5)
    for i in range(32):
        a = i * 2.39996 + t * .2
        r = 190 + i * 4
        x = mix(c.w / 2 + math.cos(a) * r, c.w / 2 + (i % 8 - 3.5) * 35, p)
        y = mix(c.h * .43 + math.sin(a) * r * .5, c.h * .4 + (i // 8 - 1.5) * 35, p)
        c.dot(x, y, 6, GREEN)
    c.text(c.w / 2, c.h * .56, "Decisions over", 56 if c.vertical else 76, GREEN, True)
    c.text(c.w / 2, c.h * .56 + 85, "stored embeddings", 56 if c.vertical else 76, GREEN, True)


def choices(c, positions, color, selected=False):
    for (x, y), label in zip(positions, EXAMPLE_OPTIONS):
        active = selected and label == "Refund"
        c.answer_box(x, y, label, color, active)


def setup(c, t):
    if c.vertical:
        c.text(c.w / 2, c.h * .16, "How Venn compares", 50, INK, True)
        c.text(c.w / 2, c.h * .23, "with Jev", 50, INK, True)
        positions = [(c.w / 2, c.h * y) for y in (.52, .60, .68)]
    else:
        c.text(c.w / 2, 108, "How Venn compares with Jev", 64, bold=True)
        positions = [(c.w / 2, y) for y in (330, 420, 510)]
    c.text(c.w / 2, c.h * .4 if c.vertical else 230, EXAMPLE_QUESTION, 38, bold=True)
    choices(c, positions, BLUE)
    for i, (x, y) in enumerate(positions):
        if int(t / .9) % 3 == i:
            c.pulse(x - 285, y, t, BLUE, 18)


def selection(c, t, model, reuse=False):
    color = GOLD if model == "Jev" else GREEN
    question_at, choices_at, reader_at = 1.3, 2.6, 4.2
    embeddings_at, head_at = 5.2, 6.6
    answer_at = 8 if model == "Venn" else 6.2
    c.text(c.w / 2, 76 if c.vertical else 60, f"How {model} chooses", 50, color, True)
    cx = c.w / 2 if c.vertical else 330
    ys = (175, 225, 315, 365, 460) if c.vertical else (134, 178, 250, 294, 358)
    for appear, label_y, value_y, label, value in [
        (0, ys[0], ys[1], "Example input", EXAMPLE_STATE),
        (question_at, ys[2], ys[3], "Question", EXAMPLE_QUESTION),
    ]:
        if t >= appear or reuse:
            focus = not reuse and t < appear + 1.3
            c.text(cx, label_y, label, 32, color if focus else MUTED)
            c.text(cx, value_y, value, 34, INK if focus else MUTED)
            if focus:
                c.trail([(cx - 265, label_y - 10), (cx - 265, value_y + 15)], color, 3)
    positions = ([(cx, y) for y in (535, 600, 665)] if c.vertical else
                 [(cx, y) for y in (413, 479, 545)])
    chosen = t >= answer_at or reuse
    if t >= choices_at or reuse:
        c.text(cx, ys[4], "Possible answers", 32, color if t < reader_at and not reuse else MUTED)
        choices(c, positions, color, chosen)
    y = 820 if c.vertical else 315
    end = (c.w / 2 if c.vertical else 970) if model == "Jev" else (510 if c.vertical else 1060)
    start = end if model == "Jev" else (210 if c.vertical else 760)
    if c.vertical:
        paths = [[(cx, 720), (start, 755), (start, y - 70)]]
    else:
        paths = [[(630, source), (660, source), (start - 90, y), (start - 70, y)]
                 for source in (198, 326, 480)]
    if t >= reader_at or reuse:
        for i, path in enumerate(paths):
            c.trail(path, LINE, 3)
            if not reuse and t < head_at:
                p = (t * .7 + i / 3) % 1
                c.dot(mix(path[0][0], path[-1][0], p), mix(path[0][1], path[-1][1], p), 5, color)
    if model == "Venn" and (t >= reader_at or reuse):
        c.ring(start, y, 65, LINE if reuse else color, 3)
        c.text(start, y, "Reader", 32, MUTED if reuse else color, True,
               bounds=(start - 52, y - 34, start + 52, y + 34))
        if t >= embeddings_at or reuse:
            c.trail([(start + 65, y), (end - 75, y)], LINE, 3)
            for i in range(9):
                p = (t * .35 + i / 9) % 1
                x = mix(start + 80, end - 80, p)
                c.d.line((x, y - 18, x, y + 8 + i % 4 * 4), fill=color, width=2)
            c.text((start + end) / 2, 920 if c.vertical else 415, "Stored embeddings", 32, color, True)
    elif model == "Jev" and t >= reader_at:
        c.text(end, 920 if c.vertical else 415, "Decision model", 32, color)
    if t >= (head_at if model == "Venn" else reader_at) or reuse:
        c.ring(end, y, 75, color, 3)
        c.text(end, y, "Jev" if model == "Jev" else "Head", 48, color, True,
               bounds=(end - 63, y - 42, end + 63, y + 42))
    if chosen:
        arrow_start = (945 if c.vertical else 440) if model == "Jev" else y + 85
        c.trail([(end, arrow_start), (end, 955 if c.vertical else 452)], color, 3)
        c.text(end if not c.vertical else cx, 1000 if c.vertical else 492, "Selected answer", 34, MUTED)
        c.text(end if not c.vertical else cx, 1060 if c.vertical else 548, "Refund", 52, color, True)


def code_scene(c, t, e):
    c.text(c.w / 2, 48, "Executed API replay", 40, GREEN, True)
    lines = head_code()
    active = min(int(t * len(lines) / 10), len(lines) - 1) if t <= 10 else -1
    for i, line in enumerate(lines):
        y = 110 + i * 43
        if i == active:
            c.d.rectangle((44, y - 20, c.w - 44, y + 23), fill="#1b303a")
        c.text(58, y, line, 32, GREEN if "register_family" in line else INK, anchor="lm", mono=True,
               bounds=(58, y - 19, c.w - 58, y + 21))
    if t > 10 and e["head"]["status"] == "executed":
        c.text(c.w / 2, c.h * .76, "Custom head registered", 38, GREEN, True)
    for i in range(24):
        p = (t * .25 + i / 24) % 1
        c.dot(80 + p * (c.w - 160), c.h * .80, 3 + i % 3, GREEN)


def history(c, t, e, mode):
    data = e["history"]
    dots = data["dots"]
    left, right = c.w * .1, c.w * .9
    top, height = c.h * .24, c.h * (.48 if c.vertical else .43)
    if mode in {"primary", "refund"}:
        top, height = (c.h * .34, c.h * .18) if c.vertical else (c.h * .32, c.h * .36)
    center = c.w / 2
    if mode == "question":
        orbit(c, t, BLUE)
        c.text(center, c.h * .16, "Company complaints", 46, GOLD, True)
        c.document(center, c.h * .44, 1.8, GOLD, hero=True)
        c.text(center, c.h * .73, "Will they return the money?", 40, bold=True)
        return
    if mode == "baseline":
        c.text(center, c.h * .14, "Complaint outcome baseline", 42, GOLD, True)
        if c.vertical:
            x1, y1, x2, y2 = center, c.h * .32, center, c.h * .65
        else:
            x1, y1, x2, y2 = c.w * .25, c.h * .44, c.w * .75, c.h * .44
        c.document(x1, y1, 1.2, GOLD)
        c.text(x1, y1 - 100, "Jev label", 36, GOLD, True)
        c.text(x1, y1 + 100, "Refund", 42, GOLD, True)
        c.trail([(x1 + (0 if c.vertical else 90), y1 + (130 if c.vertical else 0)),
                 (x2 - (0 if c.vertical else 150), y2 - (110 if c.vertical else 0))], LINE, 3)
        for i in range(24):
            x = x2 + (i % 6 - 2.5) * 32
            y = y2 + (i // 6 - 1.5) * 28 + math.sin(t + i) * 5
            c.dot(x, y, 5, GREEN if i % 4 == 0 else BLUE)
        c.text(x2, y2 - 100, "Past outcomes", 36, BLUE, True)
        c.text(x2, y2 + 100, "Group's relief rate", 36, BLUE, True)
        return
    if mode == "retrieve":
        orbit(c, t, BLUE)
        cy = c.h * .36
        c.document(center, cy, 1.55, GOLD, hero=True)
        if t <= 5.5:
            c.text(center, cy - 145, "Refund?", 48, GOLD, True)
        for i, row in enumerate(data["hero"]["precedents"]):
            p = ease((t - i * .45) / 1.3)
            if c.vertical:
                tx = center if i == 4 else c.w * (.25 if i % 2 == 0 else .75)
                ty = cy + 190 + (i // 2) * 170
            else:
                tx, ty = 125 + i * (c.w - 250) / 4, cy + 190 + (i % 2) * 30
            x, y = mix(center, tx, p), mix(cy, ty, p)
            color = GREEN if row["outcome"] == "Closed with monetary relief" else BLUE
            c.trail([(center, cy + 60), (x, y)], LINE, 2)
            flight = (t * .6 + i * .17) % 1
            c.dot(mix(x, center, flight), mix(y, cy + 60, flight), 4, color)
            c.document(x, y, .8, color, lean=math.sin(t + i) * .03)
            if p > .9:
                c.text(x, y + 72, "Relief" if color == GREEN else "No relief", 32, color)
        if t > 5.5:
            c.text(center, c.h * .17, "Her outcome: no relief", 38, GOLD, True)
        return
    burst = ease((t - .8) / 4) if mode == "burst" else 1.
    wipe = (1 - ease(t / 2.2) if mode == "restore" else
            ease((t - .8) / 3) if mode == "wipe" else float(mode == "empty"))
    for i, (score, label, hero) in enumerate(dots if mode != "refund" else []):
        scatter = left + score * (right - left)
        bar = left + data["hero"]["M"] * (right - left)
        x = mix(mix(bar, scatter, burst), center, wipe)
        y = top + (i * 47 % 253) / 253 * height
        if mode not in {"empty", "wipe"}:
            x += math.sin(t * .6 + i) * 2
        r = 8 if hero else 3.7
        color = GREEN if label else BLUE
        if mode == "empty":
            color = LINE
        c.dot(x, y, r, GOLD if hero else color)
        if hero:
            c.ring(x, y, 20 + 4 * math.sin(t * 2), GOLD, 3)
    if mode == "burst":
        c.text(center, c.h * .14, "Refund", 48, GOLD, True)
    if mode in {"primary", "refund"}:
        group = "primary" if mode == "primary" else "picture"
        values = data["metrics"][group]
        metric_pair(c, t, [("Jev label average", values["M"], GOLD), ("Venn similar cases", values["V"], GREEN)],
                    lambda v: f"{v:.3f}", y=c.h * .16)
        c.text(center, c.h * .78, "Ranking quality (higher is better)", 32, MUTED)
        if mode == "refund" and not c.vertical:
            c.text(center, c.h * .42, "Refund requests only", 42, GOLD, True)
            for i in range(9):
                x = c.w * (.18 + .64 * ((i / 9 + t * .1) % 1))
                c.document(x, c.h * .56 + math.sin(t + i) * 12, .4, BLUE)
    if mode in {"wipe", "empty"}:
        c.text(center, c.h * .14, "Complaint lookup", 44, GOLD, True)
        edge = left + (right - left) * min(1, t / 3)
        c.d.rectangle((edge, top - 25, edge + 5, top + height + 25), fill=GOLD)
        if wipe == 1:
            c.pulse(center, top + height / 2, t, LINE, 60)
            c.text(center, c.h * .77, "No past outcomes to retrieve", 38, GOLD, True)
    if mode == "restore":
        c.text(center, c.h * .14, "Complaint lookup", 44, GREEN, True)
        c.text(center, c.h * .77, "Complaint archive restored", 38, GREEN, True)


def metric_pair(c, t, values, formatter, y=None):
    if c.vertical:
        positions = [(c.w / 2, c.h * .24), (c.w / 2, c.h * .62)]
    else:
        positions = [(c.w * .25, y or c.h * .44), (c.w * .75, y or c.h * .44)]
    for i, ((label, value, color), (x, yy)) in enumerate(zip(values, positions)):
        p = ease((t - i * .25) / .8)
        bounds = ((56, 20, c.w - 56, c.h - 180) if c.vertical else
                  (48 + i * c.w / 2, 20, (i + 1) * c.w / 2 - 48, c.h - 148))
        value_text = formatter(value)
        size = 84 if c.vertical else 104
        c.text(x, yy - 52, label, 36, color, True, bounds=bounds)
        c.text(x, yy + 22 + (1 - p) * 20, value_text, size, color, True, bounds=bounds)
        width = max(220, c.d.textlength(value_text, font=font(size, True)) + 36)
        c.trail([(x - width / 2, yy + 100), (x + width / 2, yy + 100)], LINE, 3)
        c.dot(x - width / 2 + width * ((t * .3 + i / 2) % 1), yy + 100, 5, color)


def sec(c, t, e, mode):
    data = e["sec"]
    if mode == "learn":
        c.text(c.w / 2, c.h * .15, "Company filings", 46, GREEN, True)
        for i in range(18):
            p = (t * .2 + i / 18) % 1
            x = mix(c.w * .12, c.w * .85, p)
            y = mix(c.h * (.3 + (i % 3) * .14), c.h * .5, ease(p))
            c.document(x, y, .45, GREEN if i % 3 else BLUE)
            if p < .55:
                c.dot(x + 24, y - 25, 8, GOLD if i % 3 else BLUE)
        c.pulse(c.w * .85, c.h * .5, t, GREEN, 55)
        c.text(c.w * .25, c.h * .76, "Embedding + filed industry", 36, GREEN, True)
        c.text(c.w * .79, c.h * .76, "Linear head", 40, GREEN, True)
        return
    if mode in {"sample", "confirmation"}:
        rows = data["test" if mode == "sample" else "confirmation"]
        p = ease(t / (2.5 if mode == "sample" else 3))
        n = max(1, int(len(rows) * p))
        for i in range(60):
            travel = (i / 60 + t * .16) % 1
            index = min(len(rows) - 1, int((i / 60 * n + t * 7) % len(rows)))
            correct = rows[index][0] == rows[index][2]
            spread = (i % 7 - 3) * (27 if c.vertical else 60)
            x = mix(c.w / 2 + spread, c.w * (.76 if correct else .24) + spread * .3, ease(travel))
            y = c.h * (.28 + travel * .48)
            c.document(x, y, .28 + travel * .18, GREEN if correct else BLUE,
                       lean=math.sin(t + i) * .08)
        if mode == "confirmation":
            metric_pair(c, t, [("Jev", sum(r[0] == r[1] for r in rows[:n]) / n, GOLD),
                               ("Venn", sum(r[0] == r[2] for r in rows[:n]) / n, GREEN)],
                        lambda v: f"{v:.0%}", y=c.h * .25)
            c.text(c.w / 2, c.h * .76, "1,000 additional test filings", 38, MUTED)
        else:
            metric_pair(c, t, [("Jev", sum(r[0] == r[1] for r in rows[:n]), GOLD),
                               ("Venn", sum(r[0] == r[2] for r in rows[:n]), GREEN)],
                        lambda v: str(v), y=c.h * .25)
            c.text(c.w / 2, c.h * .76, "60 cookbook test filings", 38, MUTED)
        return
    if mode == "example":
        cx = c.w / 2
        c.text(cx, c.h * .14, "One filing's predictions", 44, bold=True)
        c.document(cx, c.h * .36, 1.6, BLUE)
        c.text(c.w * .25, c.h * .58, "Jev: 97%", 36, GOLD, True)
        c.text(c.w * .25, c.h * .71, "Engineering services", 36, GOLD, True)
        if t > 1:
            c.text(c.w * .75, c.h * .58, "Venn head", 36, GREEN, True)
            c.text(c.w * .75, c.h * .71, "Business Services", 36, GREEN, True)
        return
    cx = c.w / 2
    orbit(c, t, BLUE)
    c.text(cx, c.h * .17, "IT-services companies", 48, GREEN, True)
    names = ["Techteam", "Perot", "Digitalnet", "Dyntek", "Covansys"]
    for i, name in enumerate(names):
        if t < i * .7:
            continue
        p = ease((t - i * .7) / 1.2)
        if c.vertical:
            tx, ty = c.w * .14, c.h * (.26 + i * .12)
        else:
            tx, ty = 128 + i * (c.w - 256) / 4, c.h * .39 + (i % 2) * 16
        x, y = tx, ty + (1 - p) * 100 + math.sin(t + i) * 5
        c.document(x, y, .65, GREEN)
        if p > .8:
            c.text(x + 65 if c.vertical else x, y if c.vertical else y + 70,
                   name, 32, GREEN, True, anchor="lm" if c.vertical else "mm")
            if t >= 4.4:
                c.trail([(c.w * .5, y), (c.w * .75 - 140, c.h * .5)] if c.vertical else
                        [(x, y + 112), (cx, c.h * .74 - 40)], LINE, 2)
    if t >= 4.4:
        if c.vertical:
            c.text(c.w * .75, c.h * .5 - 30, "Filed under", 32, GREEN, True)
            c.text(c.w * .75, c.h * .5 + 25, "Business Services", 32, GREEN, True)
        else:
            c.text(cx, c.h * .74, "Filed under Business Services", 38, GREEN, True)


def flow(c, t, mode):
    """Two animated processing lanes; speed is illustrative, never a replay clock."""
    for lane, color in [(0, GOLD), (1, GREEN)]:
        y = c.h * (.33 + lane * .32)
        c.d.line((80, y, c.w - 80, y), fill=LINE, width=3)
        for i in range(15):
            p = (i / 15 + t * (.055 if lane == 0 else .19)) % 1
            x = 70 + p * (c.w - 140)
            c.document(x, y + math.sin(p * math.pi * 4 + t) * 13, .32, color, lean=.1 * math.sin(t + i))
        stations = [.26, .53, .78] if lane == 0 else [.53]
        for pos in stations:
            x = c.w * pos
            c.ring(x, y, 32 + 3 * math.sin(t * 4), color, 3)
            c.dot(x, y, 9, color)
            for spoke in range(4):
                angle = t * 2 + spoke * math.pi / 2
                c.trail([(x + math.cos(angle) * 17, y + math.sin(angle) * 17),
                         (x + math.cos(angle) * 29, y + math.sin(angle) * 29)], color, 3)
        c.text(85, y - 86, "Jev pipeline" if lane == 0 else "Venn", 38, color, True, anchor="lm")
    if mode == "flow":
        c.text(c.w / 2, c.h * .48, "Questions  /  Answers  /  CatBoost", 34, GOLD)
        c.text(c.w / 2, c.h * .76, "Embeddings + targets -> custom head", 38, GREEN)


def work(c, t, e, mode):
    data = e["wine"]
    if mode == "question":
        orbit(c, t, BLUE)
        c.text(c.w / 2, c.h * .15, "Wine tasting notes", 46, GREEN, True)
        c.document(c.w / 2, c.h * .43, 1.8, BLUE)
        c.text(c.w / 2, c.h * .73, "Critic's score", 44, GREEN, True)
        return
    if mode == "flow":
        flow(c, t, mode)
        return
    # Keep the work moving behind the measurement rather than cutting to cards.
    for lane, color in [(0, GOLD), (1, GREEN)]:
        yy = c.h * (.32 + lane * .34)
        for i in range(20):
            x = ((i * 89 + t * (40 if lane == 0 else 145)) % (c.w + 100)) - 50
            c.document(x, yy + math.sin(t + i) * 12, .23, color)
    if mode == "accuracy":
        scores = data["results"]["fresh_3000"]
        c.text(c.w / 2, c.h * .15, "Prediction error in score points", 42, bold=True)
        metric_pair(c, t, [("Jev", scores["five_round"], GOLD), ("Venn", scores["venn_best"], GREEN)],
                    lambda v: f"{v:.2f}")
        c.text(c.w / 2, c.h * .77, "Lower is better", 36, MUTED)
    else:
        c.text(c.w / 2, c.h * .15, "Time to build the score predictor", 42, bold=True)
        metric_pair(c, t, [("Jev: discover questions", data["jev_build"]["seconds"] / 60, GOLD),
                           ("Venn: embed and train", data["api"]["build_seconds_embed_1200_and_register"], GREEN)],
                    lambda v: f"{v:.1f} min" if v < 20 else f"{v:.0f} s")
        c.text(c.w / 2, c.h * .77, "Each time includes reading all the training notes", 34, MUTED)


def price(c, t, e, mode):
    data = e["wine"]["price"]
    if mode == "question":
        c.text(c.w / 2, c.h * .14, "New question: price", 46, GREEN, True)
        for i, (label, head, color) in enumerate([
            ("SentenceSimilarity", "Score head", BLUE), ("Classification", "Price head", GREEN),
        ]):
            x = c.w * (.25 + i * .5)
            c.text(x, c.h * .32, label, 36, color, True)
            for j in range(24):
                xx = x + (j % 6 - 2.5) * 32
                yy = c.h * .48 + (j // 6 - 1.5) * 28
                c.dot(xx, yy + math.sin(t + j) * 4, 5, color)
            c.trail([(x, c.h * .58), (x, c.h * .65)], color, 3)
            c.text(x, c.h * .74, head, 38, color, True)
        return
    orbit(c, t * 1.8, GREEN, min(c.w * .42, 450))
    angle = t * .8
    cx, cy = c.w / 2, c.h * .46
    for i in range(140):
        a = i * 2.39996
        r = 60 + math.sqrt(i / 140) * min(c.w * .4, 420)
        x, y = cx + math.cos(a) * r, cy + math.sin(a) * r * .48
        highlight = abs(((a - angle + math.pi) % (2 * math.pi)) - math.pi) < .18
        c.dot(x, y, 4 if highlight else 2, GREEN if highlight else LINE)
    c.trail([(cx, cy), (cx + math.cos(angle) * c.w * .35, cy + math.sin(angle) * c.w * .18)], GREEN, 3)
    if mode == "time":
        local = data["inventory"]["venn_family"]
        metric_pair(c, t, [("Jev direct", data["jev_direct_scaled_seconds"] / 60, GOLD),
                           ("Venn fit + predict", local["train_seconds"] + local["predict_seconds"], GREEN)],
                    lambda v: f"~{v:.0f} min" if v > 20 else f"{v:.0f} s")
        c.text(cx, c.h * .14, "Predict prices across the inventory", 42, INK, True)
        c.text(cx, c.h * .25, f"{e['wine']['inventory_rows']:,} notes", 40, INK, True)
        c.text(cx, c.h * .77, "Cached Classification embeddings", 34, GREEN)
    else:
        metric_pair(c, t, [("Jev API", data["jev_direct_total_usd"], GOLD),
                           ("Venn API", data["venn_api_usd"], GREEN)], lambda v: f"${v:.2f}" if v else "$0")
        for i in range(12):
            x = c.w * .25 + math.sin(i * 2.4) * 95
            y = (t * 95 + i * 61) % (c.h * .6) + c.h * .13
            c.ring(x, y, 7, GOLD, 2)


def finish(c, t, kind):
    orbit(c, t, GREEN)
    cx, cy = c.w / 2, c.h * .44
    if kind == "scoreboard":
        for i in range(35):
            p = (i / 35 + t * .14) % 1
            lane = i % 2
            x = mix(cx, c.w * (.2 if lane == 0 else .8), p)
            y = c.h * .18 + p * c.h * .5
            c.document(x, y, .24, GOLD if lane == 0 else GREEN)
        if c.vertical:
            c.text(cx, c.h * .2, "Jev: reading the item", 38, GOLD, True)
            c.text(cx, c.h * .65, "Venn: history and local prediction", 36, GREEN, True)
        else:
            c.text(c.w * .25, c.h * .2, "Jev", 52, GOLD, True)
            c.text(c.w * .75, c.h * .2, "Venn", 52, GREEN, True)
            c.text(c.w * .25, c.h * .73, "Grape identification and bug routing", 32, GOLD)
            c.text(c.w * .75, c.h * .73, "History and local prediction", 34, GREEN)
    elif kind == "claim":
        c.pulse(cx, cy, t, GREEN, 170)
        for i, word in enumerate(["History", "Custom heads", "Reuse"]):
            a = i * math.tau / 3 - math.pi / 2 + math.sin(t * .35) * .12
            x = cx + math.cos(a) * (165 if c.vertical else 310)
            y = cy + math.sin(a) * (220 if c.vertical else 140)
            c.text(x, y, word, 42, GREEN, True)
    else:
        c.pulse(cx, cy, t, GREEN, 145)
        if c.vertical:
            c.text(cx, cy - 85, "Build decisions", 50, GREEN, True)
            c.text(cx, cy + 10, "from your own data", 50, GREEN, True)
        else:
            c.text(cx, cy - 30, "Build decisions from your own data", 62, GREEN, True)
        c.text(cx, cy + 165, "pip install venn-vdm", 34, mono=True)
        c.text(cx, c.h * .77, "github.com/markrussinovich/venn-vdm", 32, MUTED)


def frame(scene, elapsed, e, vertical=False):
    absolute = scene["start"] + elapsed
    c = Canvas(vertical, absolute)
    kind, mode = scene["kind"], scene.get("mode")
    if kind == "setup":
        setup(c, elapsed)
    elif kind == "selection":
        selection(c, elapsed, scene["model"], scene.get("reuse", False))
    elif kind == "code":
        code_scene(c, elapsed, e)
    elif kind in {"conveyor", "hero", "question"}:
        conveyor(c, elapsed, kind != "conveyor")
        if kind == "question":
            c.pulse(c.w / 2, c.h * .43, elapsed, GREEN, 160)
        if kind == "conveyor":
            c.text(c.w / 2, c.h * .15, f"{e['history']['published']['rows']:,}", 86, GREEN, True)
    elif kind == "title":
        title(c, elapsed)
    elif kind == "history":
        history(c, elapsed, e, mode)
    elif kind == "sec":
        sec(c, elapsed, e, mode)
    elif kind == "work":
        work(c, elapsed, e, mode)
    elif kind == "price":
        price(c, elapsed, e, mode)
    else:
        finish(c, elapsed, kind)
    # Title and end typography are the scene, not a duplicated subtitle.
    if kind not in {"title", "end"}:
        c.caption(scene["card"])
    c.check_text_overlap()
    if len(c.numbers) > 3:
        raise ValueError(f"Too many numbers: {scene['card']}: {c.numbers}")
    return c.image


def timestamp(seconds, sep):
    ms = round(seconds * 1000)
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02}{sep}{ms % 1000:03}"


def captions(scenes, target):
    srt = "\n".join(f"{i}\n{timestamp(s['start'], ',')} --> {timestamp(s['end'], ',')}\n{s['card']}\n"
                    for i, s in enumerate(scenes, 1))
    target.with_suffix(".srt").write_text(srt)
    target.with_suffix(".vtt").write_text("WEBVTT\n\n" + re.sub(r"(\d\d:\d\d:\d\d),(\d{3})", r"\1.\2", srt))


# Section energy by scene kind: 0 pads only, 1 adds arpeggio and a light kick, 2 adds bass and backbeat,
# 3 adds hats and the lead melody.
MUSIC_ENERGY = {"title": 0, "setup": 0, "selection": 1, "history": 2, "sec": 3, "work": 3, "code": 2,
                "price": 3, "scoreboard": 2, "end": 1}


def music(path, seconds, scenes):
    """Original synth score in A minor (Am-F-C-G at 118 BPM): warm detuned pads, a stereo arpeggio, a pulsing
    bass, a soft kick/clap groove and a pentatonic lead that build with the story. Scene cuts get short
    rising swells instead of impacts, retrieval beats get in-key bell chimes, and the archive wipe drops
    the music out. Deterministic; no samples or third-party recordings."""
    rate = 44100
    n = int(seconds * rate)
    music_bus, drum_bus = np.zeros((n, 2)), np.zeros((n, 2))
    rng = np.random.default_rng(16)
    beat = 60 / 118
    bar = 4 * beat

    def hz(note):
        return 440 * 2 ** ((note - 69) / 12)

    def add(bus, at, signal, pan=.5):
        i = int(at * rate)
        if i >= n or i + len(signal) <= 0:
            return
        j = min(n, i + len(signal))
        part = signal[:j - i] if i >= 0 else signal[-i:j - i]
        i = max(i, 0)
        bus[i:j, 0] += part * np.sqrt(1 - pan)
        bus[i:j, 1] += part * np.sqrt(pan)

    def voice(freq, length, partials, attack, decay=0., release=.05, vibrato=0.):
        t = np.arange(int(length * rate)) / rate
        phase = 2 * np.pi * freq * (t + (vibrato / (2 * np.pi * 5.5)) * np.sin(2 * np.pi * 5.5 * t))
        wave_ = sum(a * np.sin(k * phase) for k, a in partials)
        envelope = np.minimum(t / attack, 1) * np.exp(-t * decay) * np.clip((length - t) / release, 0, 1)
        return wave_ * envelope

    def smooth(x, width):
        kernel = np.ones(width) / width
        return np.convolve(x, kernel, mode="same")

    def energy(at):
        for scene in scenes:
            if scene["start"] <= at < scene["end"]:
                return MUSIC_ENERGY.get(scene["kind"], 2)
        return 1

    chords = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]  # Am F C G
    melody = [76, None, 74, 72, 69, None, 72, 74, 76, None, 79, 76, 74, None, 72, None]
    saw = [(k, 1 / k) for k in range(1, 8)]
    pluck = [(1, 1), (2, .45), (3, .25), (4, .12)]
    flute = [(1, 1), (2, .18), (3, .11), (5, .04)]
    kicks = []
    bars = int(np.ceil(seconds / bar))
    for b in range(bars):
        at = b * bar
        level = energy(at + .01)
        chord = chords[b % 4]
        for note in chord:  # Pads: three detuned voices spread across the stereo field.
            for detune, pan in ((-.0025, .2), (0, .5), (.0025, .8)):
                add(music_bus, at, .0075 * voice(hz(note) * (1 + detune), bar + .35, saw, .35, release=.45), pan)
        if level >= 1:  # Arpeggio in sixteenths, alternating sides.
            tones = chord + [c + 12 for c in chord]
            for step, index in enumerate([0, 2, 4, 1, 3, 5, 4, 2] * 2):
                add(music_bus, at + step * beat / 4,
                    (.016 if level == 1 else .02) * voice(hz(tones[index] + 12), beat / 4 * 1.6, pluck, .003, 9),
                    .3 if step % 2 else .7)
        for k in range(4):
            if level >= 2 or (level == 1 and k % 2 == 0):
                kicks.append(at + k * beat)
            if level >= 2:  # Bass in eighths, the root with an octave pop on the off-beat.
                for half in (0, 1):
                    note = chord[0] - 24 + (12 if half and k % 2 else 0)
                    add(music_bus, at + k * beat + half * beat / 2,
                        .06 * voice(hz(note), beat / 2 * .95, [(1, 1), (2, .5), (3, .2)], .004, 5, .03))
            if level >= 2 and k % 2 == 1:  # Clap: a warm body tone and a short, filtered noise layer.
                noise = rng.normal(0, 1, int(.18 * rate))
                body = noise - smooth(noise, 9)
                t = np.arange(len(noise)) / rate
                clap = .022 * body * np.exp(-t * 26) + .02 * np.sin(2 * np.pi * 185 * t) * np.exp(-t * 30)
                add(drum_bus, at + k * beat, clap, .45)
            if level >= 3:  # Soft off-beat hats.
                noise = rng.normal(0, 1, int(.05 * rate))
                t = np.arange(len(noise)) / rate
                add(drum_bus, at + k * beat + beat / 2, .006 * np.diff(noise, prepend=0) * np.exp(-t * 70), .62)
        if level >= 3:  # Lead melody over two-bar phrases, with an echo on the opposite side.
            phrase = melody[(b % 2) * 8:(b % 2) * 8 + 8]
            for step, note in enumerate(phrase):
                if note is None:
                    continue
                line = .02 * voice(hz(note), beat * .9, flute, .025, 1.2, .12, vibrato=.004)
                add(music_bus, at + step * beat / 2, line, .42)
                add(music_bus, at + step * beat / 2 + beat * .75, .4 * line, .78)
    for at in kicks:  # Round kick: pitch sweep without a noise click.
        t = np.arange(int(.32 * rate)) / rate
        add(drum_bus, at, .16 * np.sin(2 * np.pi * (48 * t + 70 * .03 * (1 - np.exp(-t / .03)))) * np.exp(-t * 9))
    times = np.arange(n) / rate
    if kicks:  # Gentle sidechain pump on the music under each kick.
        pump = np.ones(n)
        for at in kicks:
            i = int(at * rate)
            j = min(n, i + int(beat * rate))
            pump[i:j] = np.minimum(pump[i:j], 1 - .3 * np.exp(-(times[i:j] - at) * 7))
        music_bus *= pump[:, None]
    for scene in scenes:
        start = scene["start"]
        if start > 0:  # A short rising swell into each cut, then a soft in-key shimmer.
            length = int(.7 * rate)
            noise = smooth(rng.normal(0, 1, length), 6)
            ramp = np.linspace(0, 1, length) ** 2.5
            add(music_bus, start - .7, .03 * noise * ramp, .5)
            chord = chords[int(start // bar) % 4]
            for i, note in enumerate(chord):
                add(music_bus, start, .009 * voice(hz(note + 24), 1.2, [(1, 1), (2.76, .3)], .01, 3.5), .25 + .25 * i)
        if scene.get("mode") in {"retrieve", "neighbors"}:  # Retrieval: ascending bell chimes in A minor pentatonic.
            for i, note in enumerate([81, 84, 86, 88, 91]):
                at = start + (1 + i * .7 if scene["mode"] == "neighbors" else .5 + i * .6)
                add(music_bus, at, .03 * voice(hz(note), .9, [(1, 1), (2.76, .35), (5.4, .12)], .002, 6), .3 + .1 * i)
        if scene.get("mode") == "wipe":
            mask = (times >= start + 3) & (times < scene["end"] - .3)
            music_bus[mask] *= .12
            drum_bus[mask] *= 0
    final = max(0., seconds - 3.2)  # Resolve on a held A minor chord.
    for note in (45, 57, 60, 64, 69):
        add(music_bus, final, .012 * voice(hz(note), 3.2, saw[:4], .2, .4, 1.5), .35 + (note % 5) * .07)
    mix = music_bus + drum_bus
    mix *= (np.minimum(times / .8, 1) * np.minimum((seconds - times) / 2, 1))[:, None]
    mix = np.tanh(mix * 1.6) / 1.6
    mix *= .89 / max(np.abs(mix).max(), 1e-9)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes((mix * 32767).astype("<i2").tobytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_cut(name, scenes, e, out, preview=False):
    vertical = name == "vertical"
    stem = "venn-same-memory-" + ("vertical-review" if vertical else "review")
    target = out / (stem + ".mp4")
    width, height = (720, 1280) if vertical else (1280, 720)
    output_size = "1080:1920" if vertical else "1920:1080"
    duration = scenes[-1]["end"]
    captions(scenes, out / (stem + "-captions"))
    samples = []
    for scene in scenes:
        length = scene["end"] - scene["start"]
        for fraction in (.05, .5, .95):
            image = frame(scene, length * fraction, e, vertical)
            if fraction == .5:
                samples.append(image)
    poster = frame(shot(0, 4, "title", "Decisions over stored embeddings"), 1.5, e, vertical)
    poster.resize((1080, 1920) if vertical else (1920, 1080)).save(out / (stem + "-poster.jpg"), quality=92)
    thumb_w = 240 if vertical else 384
    thumb_h = round(height / width * thumb_w)
    board = Image.new("RGB", (4 * thumb_w, math.ceil(len(samples) / 4) * thumb_h), BG)
    for i, sample in enumerate(samples):
        board.paste(sample.resize((thumb_w, thumb_h)), (i % 4 * thumb_w, i // 4 * thumb_h))
    board.save(out / (stem + "-storyboard.png"))
    if preview:
        return None
    import imageio_ffmpeg

    audio = out / (stem + "-music.wav")
    music(audio, duration, scenes)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(FPS),
               "-i", "-", "-i", str(audio), "-vf", f"scale={output_size}:flags=lanczos",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-threads", "4",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
               "-t", str(duration), str(target)]
    with (out / (stem + "-encode.log")).open("w") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for scene in scenes:
                print(name, scene["start"], scene["card"], flush=True)
                for index in range(round((scene["end"] - scene["start"]) * FPS)):
                    t = index / FPS
                    image = frame(scene, t, e, vertical)
                    process.stdin.write(image.tobytes())
            process.stdin.close()
            if process.wait():
                raise RuntimeError(f"Encoding failed; see {log.name}")
        except BaseException:
            process.kill()
            process.wait()
            raise
    audio.unlink()
    return {"file": target.name, "sha256": sha(target), "bytes": target.stat().st_size,
            "seconds": duration, "width": 1080 if vertical else 1920, "height": 1920 if vertical else 1080, "fps": FPS}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = load_evidence()
    cuts = timelines(evidence)
    media = [render_cut(name, scenes, evidence, args.output, args.preview) for name, scenes in cuts.items()]
    manifest = {
        "title": "Decisions over stored embeddings", "revision": "Two-minute-thirty landscape cut; slower explanations preserved",
        "opening_example": {
            "state": EXAMPLE_STATE, "question": EXAMPLE_QUESTION, "options": EXAMPLE_OPTIONS,
            "status": "Authored schematic; the illustrative selection is not a recorded Jev or Venn prediction.",
            "jev_reference": "https://docs.typesafe.ai/primitives/choice.md",
            "architecture_reference": "https://docs.typesafe.ai/concepts/system-one.md",
        },
        "status": "Local review draft; clearances granted by owner; do not publish without explicit instruction.",
        "narration": False,
        "music": "Original synth score in A minor (Am-F-C-G, 118 BPM): pads, arpeggio, bass, soft kick/clap groove and a pentatonic lead that build with the sections; rising swells at cuts and in-key bell chimes for retrieval; no samples or third-party recordings.",
        "speech_playback": False, "source_narratives_or_wine_notes": False,
        "on_screen_fine_print": False, "minimum_text_size_at_1080_output": 48,
        "layout": "Padded answer-box containment, metric-column bounds, reserved narration area and per-frame text-overlap checks",
        "narration_style": "One cool-blue regular-weight caption block, distinct from graphic labels, without sentence-ending periods",
        "disclosure": "On methods page only: Microsoft employee; personal research. Owner confirmed clearances on 2026-11-07; publication on hold.",
        "animation": "Continuous 30 fps record conveyors, camera-depth motion, data morphs, retrieval trails, archive wipe and processing lanes. Slower architecture and filing-label reveals; no frozen post-intro frames.",
        "timing": "All animation speeds are editorial, not real-time execution or measurement recordings.",
        "pacing": "Landscape trimmed to 150 seconds by removing the 3-second summary animation and shortening the end card by 1 second; vertical remains 60 seconds. Slower architecture stages and the 10-second code highlight are unchanged.",
        "audio_section": False,
        "evidence": {f"{beat}-eval.json": sha(EVIDENCE / f"{beat}-eval.json")
                     for beat in ("history", "sec", "wine")},
        "head_execution": {
            "file": "same-memory/head-execution.json", "sha256": sha(EVIDENCE / "head-execution.json"),
            "source": "examples/same_memory_video/head_demo.py",
            "source_sha256": evidence["head"]["source_sha256"],
            "replay": "Verbatim executed source; replay compressed to fit the edit, without a real-time clock.",
        },
        "renderer_sha256": sha(Path(__file__)), "media": [m for m in media if m], "cuts": cuts,
        "limitations": [
            "Owner confirmed clearances; public upload and publication remain on hold.",
            "SEC primary gates failed; confirmation is report-only, not a win.",
            "Wipe is conceptual; no measured empty-archive 0.50 result is claimed.",
            "Separate historical build timings; workflow motion is illustrative. Code is an executed, compressed replay, not live footage.",
            "Price uses cached Classification vectors, not the winning score head's SentenceSimilarity vectors.",
            "Notebooks verify exported point estimates, not end-to-end experiments or historical timings.",
            "Fine print and disclosures remain on the methods page at the user's request.",
        ],
    }
    (args.output / "venn-same-memory-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
