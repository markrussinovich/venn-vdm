import importlib.util
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_source = Path(__file__).resolve().parents[1] / "examples/same_memory_video/render.py"
_spec = importlib.util.spec_from_file_location("same_memory_render", _source)
render = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render)


@pytest.fixture(scope="module", autouse=True)
def require_renderer_fonts():
    try:
        render.font(36)
        render.font(36, mono=True)
    except RuntimeError:
        pytest.skip("Motion renderer requires the documented Lato/DejaVu system fonts")


@pytest.mark.parametrize("kind,mode", [
    ("history", "retrieve"),
    ("sec", "confirmation"),
    ("work", "flow"),
])
def test_foreground_keeps_moving_after_the_entrance(kind, mode):
    evidence = render.load_evidence()
    scene = render.shot(0, 9, kind, "Keep the story moving.", mode=mode)
    a = np.asarray(render.frame(scene, 6, evidence))[180:550, 80:1200].astype(int)
    b = np.asarray(render.frame(scene, 6.5, evidence))[180:550, 80:1200].astype(int)
    changed = np.max(np.abs(a - b), axis=2) > 20
    # Background dust alone cannot satisfy this: the records and processing must move.
    assert changed.sum() > 1500


def test_cut_has_no_fine_print_or_publication_overlays(monkeypatch):
    evidence = render.load_evidence()
    seen = []
    original = render.Canvas.text

    def capture(self, x, y, value, size=36, *args, **kwargs):
        seen.append((value, size))
        return original(self, x, y, value, size, *args, **kwargs)

    monkeypatch.setattr(render.Canvas, "text", capture)
    for name, scenes in render.timelines(evidence).items():
        for scene in scenes:
            assert not re.search(r"[.!?]\s+\S", scene["card"])
            assert not scene["card"].endswith(".")
            render.frame(scene, (scene["end"] - scene["start"]) * .9, evidence, name == "vertical")
    assert seen
    assert min(size for _, size in seen) >= 32
    for value, _ in seen:
        assert not value.endswith((".", '."'))
        assert not any(word in value.lower() for word in (
            "approval pending", "clearance", "review draft", "microsoft", "promising", "not conclusive",
            "rmse", "rsme", "auroc", "jev reads.", "venn remembers."))


@pytest.mark.parametrize("vertical", [False, True])
def test_opening_explains_the_same_selection_in_separate_scenes(monkeypatch, vertical):
    seen = []
    original = render.Canvas.text

    def capture(self, x, y, value, *args, **kwargs):
        seen.append(value)
        return original(self, x, y, value, *args, **kwargs)

    monkeypatch.setattr(render.Canvas, "text", capture)
    evidence = render.load_evidence()
    scenes = render.timelines(evidence)["vertical" if vertical else "landscape"][:3]
    render.frame(scenes[0], 0, evidence, vertical)
    assert any("Jev" in value for value in seen) and any("Venn" in value for value in seen)
    for scene, model in zip(scenes[1:], ("Jev", "Venn")):
        seen.clear()
        for elapsed in (0, 2, 5, 8.5):
            render.frame(scene, elapsed, evidence, vertical)
        assert f"How {model} chooses" in seen
        other = "Venn" if model == "Jev" else "Jev"
        assert f"How {other} chooses" not in seen
        assert {render.EXAMPLE_STATE, render.EXAMPLE_QUESTION, *render.EXAMPLE_OPTIONS}.issubset(seen)
        assert "Selected answer" in seen
        assert ("Decision model" if model == "Jev" else "Stored embeddings") in seen
        assert "Embeddings represent meaning as numbers" not in seen
        assert "Returns a choice and probabilities" not in seen


def test_score_comparison_explains_error_in_points(monkeypatch):
    seen = []
    original = render.Canvas.text

    def capture(self, x, y, value, *args, **kwargs):
        seen.append(value)
        return original(self, x, y, value, *args, **kwargs)

    monkeypatch.setattr(render.Canvas, "text", capture)
    evidence = render.load_evidence()
    render.frame(render.shot(0, 6, "work", "Compare score prediction errors.", mode="accuracy"), 3, evidence)
    scores = evidence["wine"]["results"]["fresh_3000"]
    assert {"Prediction error in score points", "Lower is better",
            f"{scores['five_round']:.2f}", f"{scores['venn_best']:.2f}"}.issubset(seen)


@pytest.mark.parametrize("kind,heading,assumption", [
    ("work", "Time to build the score predictor", "includes reading all the training notes"),
    ("price", "Predict prices across the inventory", "excludes the already-completed embedding step"),
])
def test_timing_comparisons_distinguish_workloads_and_embedding_scope(monkeypatch, kind, heading, assumption):
    seen = {}
    original = render.Canvas.text

    def capture(self, x, y, value, *args, **kwargs):
        seen[value] = x
        return original(self, x, y, value, *args, **kwargs)

    monkeypatch.setattr(render.Canvas, "text", capture)
    evidence = render.load_evidence()
    scene = next(s for s in render.timelines(evidence)["landscape"]
                 if s["kind"] == kind and s.get("mode") == "time")
    render.frame(scene, 3, evidence)
    assert heading in seen and assumption in " ".join(seen)
    if kind == "work":
        assert f"Both learn from the same {evidence['head']['examples']:,} training notes" in seen
    else:
        assert f"{evidence['wine']['inventory_rows']:,} notes" in seen
        assert "Cached Classification embeddings" in seen


def test_code_scene_displays_the_executed_function_verbatim(monkeypatch):
    seen = []
    original = render.Canvas.text

    def capture(self, x, y, value, *args, **kwargs):
        seen.append(value)
        return original(self, x, y, value, *args, **kwargs)

    monkeypatch.setattr(render.Canvas, "text", capture)
    evidence = render.load_evidence()
    scene = render.shot(0, 14, "code", "Fit a custom head")
    render.frame(scene, 8, evidence)
    assert "Custom head registered" not in seen
    seen.clear()
    render.frame(scene, 12, evidence)
    lines = render.head_code()
    assert [value for value in seen if value in lines] == lines
    assert "Custom head registered" in seen


@pytest.mark.parametrize("vertical", [False, True])
def test_answer_labels_fit_inside_padded_boxes(monkeypatch, vertical):
    seen = []
    original = render.Canvas.text

    def capture(self, x, y, value, size=36, *args, **kwargs):
        if value in render.EXAMPLE_OPTIONS and size == 36:
            seen.append(kwargs.get("bounds"))
        return original(self, x, y, value, size, *args, **kwargs)

    monkeypatch.setattr(render.Canvas, "text", capture)
    evidence = render.load_evidence()
    scenes = render.timelines(evidence)["vertical" if vertical else "landscape"][:3]
    for scene in scenes:
        render.frame(scene, 5.5, evidence, vertical)
    assert len(seen) == 9
    assert all(bounds is not None for bounds in seen)
    canvas = render.Canvas(vertical, 0)
    with pytest.raises(ValueError, match="outside its container"):
        canvas.text(canvas.w / 2, 200, "Replacement", 36, bold=True,
                    bounds=(canvas.w / 2 - 98, 170, canvas.w / 2 + 98, 230))


@pytest.mark.parametrize("vertical", [False, True])
def test_architecture_reveals_follow_the_data_path(monkeypatch, vertical):
    seen = []
    original = render.Canvas.text

    def capture(self, x, y, value, *args, **kwargs):
        seen.append(value)
        return original(self, x, y, value, *args, **kwargs)

    monkeypatch.setattr(render.Canvas, "text", capture)
    evidence = render.load_evidence()
    scene = render.timelines(evidence)["vertical" if vertical else "landscape"][2]
    steps = [
        (0, render.EXAMPLE_STATE, render.EXAMPLE_QUESTION),
        (1.1, render.EXAMPLE_STATE, render.EXAMPLE_QUESTION),
        (1.5, render.EXAMPLE_QUESTION, "Possible answers"),
        (3, "Possible answers", "Reader"),
        (4.5, "Reader", "Stored embeddings"),
        (5.5, "Stored embeddings", "Head"),
        (7, "Head", "Selected answer"),
        (8.5, "Selected answer", None),
    ]
    for elapsed, present, absent in steps:
        seen.clear()
        render.frame(scene, elapsed, evidence, vertical)
        assert present in seen
        assert absent not in seen


@pytest.mark.parametrize("vertical", [False, True])
def test_narration_is_one_distinct_block_below_the_graphics(monkeypatch, vertical):
    calls = []
    original = render.Canvas.text

    def capture(self, x, y, value, size=36, color=render.INK, bold=False, **kwargs):
        calls.append((y, value, size, color, bold, kwargs.get("bounds")))
        return original(self, x, y, value, size, color, bold, **kwargs)

    monkeypatch.setattr(render.Canvas, "text", capture)
    evidence = render.load_evidence()
    scene = render.timelines(evidence)["vertical" if vertical else "landscape"][2]
    render.frame(scene, 5.5, evidence, vertical)
    narration = [call for call in calls if call[3] == render.CAPTION]
    graphic = [call for call in calls if call[3] != render.CAPTION]
    assert " ".join(call[1] for call in narration) == scene["card"]
    assert len({(call[2], call[3], call[4]) for call in narration}) == 1
    assert all(not call[4] and call[5] is not None for call in narration)
    assert max(call[0] for call in graphic) < min(call[0] for call in narration)


@pytest.mark.parametrize("vertical", [False, True])
def test_multiline_titles_keep_the_same_style(monkeypatch, vertical):
    calls = {}
    original = render.Canvas.text

    def capture(self, x, y, value, size=36, color=render.INK, bold=False, **kwargs):
        calls[value] = (size, color, bold)
        return original(self, x, y, value, size, color, bold, **kwargs)

    monkeypatch.setattr(render.Canvas, "text", capture)
    evidence = render.load_evidence()
    render.frame(render.shot(0, 4, "title", "Poster"), 1, evidence, vertical)
    assert calls["Decisions over"] == calls["stored embeddings"]
    if vertical:
        render.frame(render.shot(0, 4, "setup", "Setup"), 1, evidence, vertical)
        assert calls["How Venn compares"] == calls["with Jev"]
        render.frame(render.shot(0, 4, "end", "End"), 1, evidence, vertical)
        assert calls["Build decisions"] == calls["from your own data"]


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("mode", ["primary", "refund"])
def test_complaint_dots_leave_metric_text_clear(monkeypatch, vertical, mode):
    boxes, dots = [], []
    original_text, original_dot = render.Canvas.text, render.Canvas.dot

    def capture_text(self, *args, **kwargs):
        original_text(self, *args, **kwargs)
        boxes.append(self.text_boxes[-1][1])

    def capture_dot(self, x, y, radius=5, *args, **kwargs):
        dots.append((x, y, radius))
        original_dot(self, x, y, radius, *args, **kwargs)

    monkeypatch.setattr(render.Canvas, "text", capture_text)
    monkeypatch.setattr(render.Canvas, "dot", capture_dot)
    scene = render.shot(0, 6, "history", "Compare historical outcome rankings", mode=mode)
    render.frame(scene, 3, render.load_evidence(), vertical)
    assert dots and boxes
    for x, y, radius in dots:
        assert not any(x + radius + 8 > a and x - radius - 8 < c
                       and y + radius + 8 > b and y - radius - 8 < d
                       for a, b, c, d in boxes)


def test_changed_code_requires_a_new_execution_record(monkeypatch):
    monkeypatch.setattr(render, "sha", lambda _path: "changed source")
    with pytest.raises(ValueError, match="Execute the displayed"):
        render.load_evidence()


@pytest.mark.parametrize("vertical", [False, True])
def test_complaint_wipe_collapses_lookup_and_restore_spreads_it(monkeypatch, vertical):
    xs = []
    original = render.Canvas.dot

    def capture(self, x, y, radius=5, *args, **kwargs):
        if radius == 3.7:
            xs.append(x)
        return original(self, x, y, radius, *args, **kwargs)

    monkeypatch.setattr(render.Canvas, "dot", capture)
    evidence = render.load_evidence()
    render.frame(render.shot(0, 5, "history", "Remove this complaint archive", mode="wipe"),
                 4, evidence, vertical)
    assert len(xs) > 10 and np.ptp(xs) == 0
    xs.clear()
    render.frame(render.shot(0, 3, "history", "Restore this complaint archive", mode="restore"),
                 2.5, evidence, vertical)
    assert len(xs) > 10 and np.ptp(xs) > 100


def test_sec_convention_gallery_has_no_hero_query(monkeypatch):
    heroes, text = [], []
    original = render.Canvas.document
    original_text = render.Canvas.text

    def capture(self, *args, **kwargs):
        heroes.append(kwargs.get("hero", False))
        return original(self, *args, **kwargs)

    def capture_text(self, x, y, value, *args, **kwargs):
        text.append(value)
        return original_text(self, x, y, value, *args, **kwargs)

    monkeypatch.setattr(render.Canvas, "document", capture)
    monkeypatch.setattr(render.Canvas, "text", capture_text)
    render.frame(render.shot(0, 8, "sec", "Venn learns industry labels from past filings", mode="neighbors"),
                 6, render.load_evidence())
    assert len(heroes) == 5 and not any(heroes)
    assert {"IT-services companies", "Filed under Business Services"}.issubset(text)


def test_price_bridge_connects_each_existing_cache_to_its_own_head(monkeypatch):
    positions = {}
    original = render.Canvas.text

    def capture(self, x, y, value, *args, **kwargs):
        positions[value] = x
        return original(self, x, y, value, *args, **kwargs)

    monkeypatch.setattr(render.Canvas, "text", capture)
    render.frame(render.shot(0, 5, "price", "Price uses a different stored cache", mode="question"),
                 3, render.load_evidence())
    assert positions["SentenceSimilarity"] == positions["Score head"]
    assert positions["Classification"] == positions["Price head"]
    assert positions["Score head"] != positions["Price head"]


@pytest.mark.parametrize("cut", ["landscape", "vertical"])
def test_audio_beat_and_audio_claims_are_absent_from_the_rendered_cut(monkeypatch, cut):
    seen = []
    original = render.Canvas.text

    def capture(self, x, y, value, *args, **kwargs):
        seen.append(value)
        return original(self, x, y, value, *args, **kwargs)

    monkeypatch.setattr(render.Canvas, "text", capture)
    evidence = render.load_evidence()
    for scene in render.timelines(evidence)[cut]:
        assert scene["kind"] != "sound"
        render.frame(scene, scene["end"] - scene["start"] - .2, evidence, cut == "vertical")
    assert not re.search(r"\b(audio|sound|voices?|speakers?)\b", " ".join(seen), re.I)


def test_encoder_keeps_animating_and_cuts_without_overlaid_shots(tmp_path, monkeypatch):
    frames = []
    encoder = SimpleNamespace(
        stdin=SimpleNamespace(write=frames.append, close=lambda: None),
        wait=lambda: 0,
        kill=lambda: None,
    )

    def fake_encoder(command, **_kwargs):
        Path(command[-1]).write_bytes(b"encoded fixture")
        return encoder

    def fake_frame(scene, elapsed, _evidence, _vertical=False):
        color = 40 if scene["kind"] == "history" else 160
        return render.Image.new("RGB", (4, 4), (color, round(elapsed * 10), 0))

    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", SimpleNamespace(get_ffmpeg_exe=lambda: "ffmpeg"))
    monkeypatch.setattr(render.subprocess, "Popen", fake_encoder)
    monkeypatch.setattr(render, "music", lambda path, *_args: path.write_bytes(b"audio fixture"))
    monkeypatch.setattr(render, "frame", fake_frame)
    monkeypatch.setattr(render, "FPS", 2)
    scenes = [render.shot(0, 4, "history", "First shot."),
              render.shot(4, 8, "work", "Next shot.")]
    render.render_cut("landscape", scenes, {}, tmp_path)
    assert len(frames) == 16
    assert frames[7] == render.Image.new("RGB", (4, 4), (40, 35, 0)).tobytes()
    assert frames[8] == render.Image.new("RGB", (4, 4), (160, 0, 0)).tobytes()
