"""A screen-only ViZDoom player driven by Venn's outcome API and an app-owned FAISS memory.

Flow per step: grab each game's screen, embed it (natively through Venn on CUDA, or with a
portable image reader), ask ``OutcomeService`` which move had the best recorded return for
similar screens, press that button. When a game ends, discounted returns for its
(screen, move) pairs are written back to the memory. The reader is frozen throughout.

With ``--head awr`` the memory is also compiled into a small head (``OutcomeService.register_family``)
that answers the same question without a lookup; ``--head-every N`` re-registers it during training
and serves it from then on. The head is the only trained part: 49,153 parameters with text option
vectors, saved as ``head.pt``.

    python examples/arena.py --smoke-test --output /tmp/venn-arena-smoke
    python examples/arena.py --reader-mode portable --device cpu --model sentence-transformers/clip-ViT-B-32 \\
        --revision FULL_40_CHARACTER_COMMIT --layout whole --games 4 --decisions 20000 --level basic \\
        --download-reader --output /tmp/venn-arena-clip

See docs/arena-example.md for the native CUDA configuration used in the video.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time

import faiss
import numpy as np

from venn import OutcomeService
from venn.adapters import FaissOutcomeAdapter
from venn.heads.outcome import MINIMUM_CASES
from venn.profiles import IMAGE_TAGS, VENN_1, VectorTag


def fingerprint(vector):
    return hashlib.sha256(np.asarray(vector, dtype=np.float16).tobytes()).digest()


def returns(rewards, gamma):
    values = np.empty(len(rewards), dtype=np.float32)
    total = 0.
    for index in range(len(rewards) - 1, -1, -1):
        total = float(rewards[index]) + gamma * total
        values[index] = total
    return values


class Memory:
    def __init__(self, tag, options, index_device="cpu"):
        self.tag, self.options = tag, list(options)
        self.parts, self.resources = {}, None
        if index_device != "cpu":
            if not hasattr(faiss, "StandardGpuResources"):
                raise ValueError("GPU indexes require a separately installed FAISS GPU build")
            self.resources = faiss.StandardGpuResources()
        for option in self.options:
            index = faiss.IndexFlatL2(tag.dimensions)
            if self.resources is not None:
                index = faiss.index_cpu_to_gpu(self.resources, int(index_device.split(":")[1]), index)
            self.parts[option] = {"index": index, "ids": [], "outcomes": [], "exact_positions": {}}
        self.adapter = FaissOutcomeAdapter(self.parts, tag=tag, query_dtype="float16")

    @property
    def size(self):
        return sum(len(part["ids"]) for part in self.parts.values())

    def observe(self, vectors, actions, rewards, gamma):
        updates = {}
        for vector, action, value in zip(vectors, actions, returns(rewards, gamma)):
            identity = (str(action), fingerprint(vector))
            if identity not in updates or value > updates[identity][1]:
                updates[identity] = (vector, float(value))
        additions = {}
        for (action, key), (vector, value) in updates.items():
            part = self.parts[action]
            position = part["exact_positions"].get(key)
            if position is None:
                position = len(part["ids"])
                part["exact_positions"][key] = position
                part["ids"].append(position)
                part["outcomes"].append(value)
                additions.setdefault(action, []).append(np.asarray(vector, np.float16).astype(np.float32))
            elif value > part["outcomes"][position]:
                part["outcomes"][position] = value
        for action, vectors in additions.items():
            self.parts[action]["index"].add(np.asarray(vectors, dtype=np.float32))

    def save(self, output, config):
        arrays = {}
        for option, part in self.parts.items():
            arrays[f"vectors_{option}"] = part["index"].reconstruct_n(0, len(part["ids"]))
            arrays[f"outcomes_{option}"] = np.asarray(part["outcomes"], dtype=np.float32)
        np.savez(output / "memory.npz", **arrays)
        (output / "memory.json").write_text(json.dumps(
            {"tag": self.tag.to_dict(), "options": self.options, "config": config}, indent=2) + "\n")

    def load(self, directory):
        info = json.loads((directory / "memory.json").read_text())
        self.tag.check(info["tag"])
        if info["options"] != self.options:
            raise ValueError("Memory action layout changed; start a new store")
        with np.load(directory / "memory.npz", allow_pickle=False) as arrays:
            for option in self.options:
                vectors, values = arrays[f"vectors_{option}"], arrays[f"outcomes_{option}"]
                if (vectors.ndim != 2 or vectors.shape[1] != self.tag.dimensions
                        or len(vectors) != len(values) or values.ndim != 1
                        or not np.isfinite(vectors).all() or not np.isfinite(values).all()):
                    raise ValueError("Invalid saved memory")
                for vector, value in zip(vectors, values):
                    self.observe([vector], [option], [value], 1.)


class PortableReader:
    def __init__(self, model_id, revision, device, dtype, layout, download=False):
        import torch
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Pin --revision to a full 40-character model commit")
        snapshot = snapshot_download(model_id, revision=revision, local_files_only=not download)
        self.model = SentenceTransformer(snapshot, device=device, model_kwargs={"dtype": getattr(torch, dtype)}).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.layout, self.gemma = layout, model_id == VENN_1.reader
        dimensions = int(self.model.get_sentence_embedding_dimension())
        protocol = {"model": model_id, "layout": layout, "dtype": dtype,
                    "pipeline": "PIL crops; model image processor; normalized fields then concatenation",
                    "gemma_image_soft_tokens": 70 if self.gemma else None}
        digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:24]
        self.tag = VectorTag("experimental:" + digest, revision, "arena-pil-" + layout,
                             dimensions * (3 if layout == "thirds" else 1),
                             ("left", "center", "right") if layout == "thirds" else ("image",))

    def encode(self, frames):
        from PIL import Image

        items = []
        for frame in frames:
            image = Image.fromarray(frame)
            width, height = image.size
            cuts = ((0, width),) if self.layout == "whole" else (
                (0, width // 3), (width // 3, width * 2 // 3), (width * 2 // 3, width))
            if self.layout == "thirds" and width == 320:
                cuts = ((0, 106), (106, 213), (213, 320))
            for left, right in cuts:
                crop = image.crop((left, 0, right, height))
                items.append({"text": "<|image|>", "image": [crop]} if self.gemma else crop)
        kwargs = {"processing_kwargs": {"image": {"max_soft_tokens": 70}}} if self.gemma else {}
        values = self.model.encode(items, batch_size=32, normalize_embeddings=True,
                                   convert_to_numpy=True, **kwargs)
        values = np.asarray(values, np.float32).reshape(len(frames), -1)
        if not np.isfinite(values).all():
            raise ValueError("Reader returned invalid embeddings; use an image-capable model")
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


class Video:
    def __init__(self, path, fps):
        import imageio_ffmpeg
        import subprocess

        self.process = subprocess.Popen([
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", "320x240", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    def add(self, frame):
        self.process.stdin.write(np.asarray(frame, np.uint8).tobytes())

    def close(self):
        self.process.stdin.close()
        if self.process.wait():
            raise RuntimeError("Evaluation recording failed")


def make_game(args, slot, episode, evaluating=False):
    import vizdoom as vzd

    level = Path(args.level)
    config = level if level.is_file() else Path(vzd.scenarios_path) / (args.level + ".cfg")
    if not config.is_file():
        raise ValueError("Use a bundled scenario name or an existing ViZDoom .cfg path")
    scenario_offset = {"basic": 0, "defend_the_center": 1000000}.get(
        args.level, int(hashlib.sha256(args.level.encode()).hexdigest()[:8], 16))
    seed = (10102026 + scenario_offset + (args.seed + (200 if evaluating else 0)) * 10000000
            + slot * 10007 + episode * 1000003) % (2**31 - 1)
    game = vzd.DoomGame()
    game.load_config(str(config.resolve()))
    game.set_doom_game_path(str(Path(vzd.install_path) / "freedoom2.wad"))
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_labels_buffer_enabled(False)
    game.set_episode_timeout(args.episode_timeout)
    game.set_seed(seed)
    game.init()
    game.set_seed(seed)
    game.new_episode()
    return game


def decide(service, handle, inputs, tag, rng, epsilon, k, exact, native=False, family=None):
    """One ``nearest_outcome`` question per batch; with ``family`` the registered head answers it alone."""
    states = inputs if native else [{"vector": vector, "tag": tag.to_dict()} for vector in inputs]
    question = {"move": {"type": "nearest_outcome", "store": handle,
                         "options": list(service.stores[handle].partitions),
                         "k": k, "min_support": 1, "exact_match": exact, "unsupported_value": 0}}
    if family is not None:
        question["move"]["family"] = family
    return service.decide_batch(states, question, rng=rng, epsilon=epsilon,
                                representation=tag.representation)


def option_vectors(reader, buttons, native, smoke):
    """Text embeddings of the button names from the same frozen reader; one-hot vectors in the smoke test."""
    if smoke:
        return None
    model = reader.reader if native else reader.model
    names = [button.replace("_", " ").lower() for button in buttons]
    values = np.asarray(model.encode(names, prompt_name="Classification", normalize_embeddings=False, convert_to_numpy=True), dtype=np.float64)
    return (values / np.linalg.norm(values, axis=1, keepdims=True)).astype(np.float32)


def register_head(service, handle, options, vectors, args, warm):
    """Compile the current memory into the ``policy`` family with the chosen objective."""
    return service.register_family("policy", from_store=handle, options=options, option_vectors=vectors,
                                   objective=args.head, k=args.k, exact_match=args.exact_match, unsupported_value=0,
                                   seed=args.seed, warm_start="policy" if warm else None)


def evaluate(args, service, handle, reader, tag, native, buttons, episodes, family=None, record=None):
    """Greedy evaluation games from fresh seeds; returns one record per game."""
    import vizdoom as vzd

    rng = np.random.default_rng(10102026 + 200)
    results = []
    for episode in range(episodes):
        game = make_game(args, episode, 0, evaluating=True)
        video = Video(record, 35 / args.frame_skip) if record is not None and episode == 0 else None
        try:
            while not game.is_episode_finished():
                frames = np.asarray([game.get_state().screen_buffer.copy()])
                if video is not None:
                    video.add(frames[0])
                inputs = frames if native else reader.encode(frames)
                response = decide(service, handle, inputs, tag, rng, [0.], args.k, args.exact_match, native, family)
                action = int(response["results"][0]["answers"]["move"]["action"])
                game.make_action([int(i == action) for i in range(len(buttons))], args.frame_skip)
            results.append({"episode": episode, "kills": float(game.get_game_variable(vzd.GameVariable.KILLCOUNT)),
                            "reward": float(game.get_total_reward())})
        finally:
            if video is not None:
                video.close()
            game.close()
    return results


def run(args):
    import vizdoom as vzd

    setup_start = time.perf_counter()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    games = []
    try:
        for slot in range(args.games):
            games.append(make_game(args, slot, 0))
        buttons = [button.name for button in games[0].get_available_buttons()]
        if not buttons or any("DELTA" in name for name in buttons):
            raise ValueError("This example supports discrete one-button moves, not continuous controls")
        if any([button.name for button in game.get_available_buttons()] != buttons for game in games):
            raise ValueError("All games must use the same action layout")
        native = args.reader_mode == "native" and not args.smoke_test
        if args.smoke_test:
            tag = VectorTag("experimental:arena-smoke", "0" * 40, "smoke-rgb-means", 3, ("image",))
            reader = None
        elif native:
            from venn.images import ImageBatchReader

            if args.download_reader:
                from huggingface_hub import snapshot_download

                snapshot_download(VENN_1.reader, revision=VENN_1.revision)
            reader = ImageBatchReader(args.device, compile_norms=args.compile_norms)
            tag = IMAGE_TAGS["image-thirds-70" if args.layout == "thirds" else "image-70"]
        else:
            reader = PortableReader(args.model, args.revision, args.device, args.dtype, args.layout, args.download_reader)
            tag = reader.tag
        memory = Memory(tag, [str(i) for i in range(len(buttons))], args.index_device)
        if args.load_memory:
            memory.load(Path(args.load_memory))
        service = OutcomeService(image_reader=reader if native else None, head_device=args.device)
        handle = service.register_store(memory.adapter)
        options = [str(i) for i in range(len(buttons))]
        vectors_for_options = option_vectors(reader, buttons, native, args.smoke_test) if args.head != "none" else None
        family, registrations = None, []
        rng = np.random.default_rng(10102026 + args.seed)
        trajectories = [[] for _ in games]
        episode_numbers = [0] * len(games)
        completed, window, curve = [], deque(maxlen=100), []
        start, counter = time.perf_counter(), 0
        setup_seconds = start - setup_start
        while counter < args.decisions:
            active = list(range(min(args.games, args.decisions - counter)))
            frames = np.asarray([games[slot].get_state().screen_buffer.copy() for slot in active])
            if args.smoke_test:
                vectors = frames.mean(axis=(1, 2)).astype(np.float32) + 1
                vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            elif not native:
                vectors = reader.encode(frames)
            epsilon = [0. if slot < min(8, args.games) else max(.05, 1 - .95 * counter / (.3 * args.decisions))
                       for slot in active]
            response = decide(service, handle, frames if native else vectors, tag, rng,
                              epsilon, args.k, args.exact_match, native, family)
            vectors = np.asarray([row["vector"] for row in response["results"]])
            for row, slot in enumerate(active):
                action = int(response["results"][row]["answers"]["move"]["action"])
                reward = games[slot].make_action([int(i == action) for i in range(len(buttons))], args.frame_skip)
                trajectories[slot].append((vectors[row], action, reward))
                if games[slot].is_episode_finished():
                    kills = float(games[slot].get_game_variable(vzd.GameVariable.KILLCOUNT))
                    completed.append({"slot": slot, "episode": episode_numbers[slot], "kills": kills,
                                      "reward": float(games[slot].get_total_reward())})
                    if slot < min(8, args.games):
                        window.append(kills)
                    if not args.smoke_test:
                        vectors_done, actions_done, rewards = zip(*trajectories[slot])
                        memory.observe(vectors_done, actions_done, rewards, args.gamma)
                    trajectories[slot] = []
                    episode_numbers[slot] += 1
                    games[slot].close()
                    games[slot] = make_game(args, slot, episode_numbers[slot])
            counter += len(active)
            if args.head_every and counter % args.head_every == 0 and memory.size >= MINIMUM_CASES:
                summary = register_head(service, handle, options, vectors_for_options, args, family is not None)
                family = "policy"
                registrations.append({"decisions": counter, "cases": summary["cases"], "seconds": summary["seconds"],
                                      "sha256": summary["sha256"]})
            if counter % (args.games * 20) == 0 or counter == args.decisions:
                curve.append({"decisions": counter, "greedy_online_kills": float(np.mean(window)) if window else None,
                              "memory_entries": memory.size, "served_by": "head" if family else "memory",
                              "wall_seconds": time.perf_counter() - start})
                (output / "progress.json").write_text(json.dumps(curve[-1], indent=2) + "\n")
        for game in games:
            game.close()
        games = []
        training_seconds = time.perf_counter() - start
        config = {**vars(args), "buttons": buttons, "tag": tag.to_dict(),
                  "reader_frozen": True, "scope": "new local run; not a reproduction of published scores",
                  "memory_restart": "only memory is portable; live game/RNG/unfinished episodes are not checkpointed"}
        save_start = time.perf_counter()
        memory.save(output, config)
        memory_save_seconds = time.perf_counter() - save_start
        before_evaluation = memory.size
        head = None
        if args.head != "none" and not args.smoke_test:
            head_start = time.perf_counter()
            summary = register_head(service, handle, options, vectors_for_options, args, family is not None)
            service.save_family("policy", output / "head.pt")
            head = {"objective": args.head, "sha256": summary["sha256"], "parameters": summary["parameters"],
                    "cases": summary["cases"], "registration_seconds": time.perf_counter() - head_start,
                    "registrations_during_training": registrations, "fit": summary["report"]["losses"],
                    "option_vectors": "text embeddings of the button names from the frozen reader"}
        evaluation_start = time.perf_counter()
        episodes = 0 if args.smoke_test else args.eval_episodes
        evaluations = evaluate(args, service, handle, reader, tag, native, buttons, episodes,
                               record=output / "evaluation-first.mp4" if args.record else None)
        if head is not None:
            head_evaluations = evaluate(args, service, handle, reader, tag, native, buttons, episodes, family="policy",
                                        record=output / "evaluation-first-head.mp4" if args.record else None)
            head["evaluation"] = head_evaluations
            head["mean_evaluation_kills"] = float(np.mean([row["kills"] for row in head_evaluations])) if head_evaluations else None
        assert memory.size == before_evaluation
        result = {"config": config, "decisions": counter, "training_wall_seconds": training_seconds,
                  "setup_seconds": setup_seconds, "memory_save_seconds": memory_save_seconds,
                  "evaluation_seconds": time.perf_counter() - evaluation_start,
                  "total_wall_seconds": time.perf_counter() - setup_start,
                  "online_training": curve, "completed_training_episodes": completed,
                  "evaluation": evaluations, "mean_evaluation_kills": float(np.mean(
                      [row["kills"] for row in evaluations])) if evaluations else None,
                  "evaluation_served_by": "memory", "head": head,
                  "smoke_test": args.smoke_test, "reader_weights_trained": False, "head_trained": head is not None}
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"output": str(output), "smoke_test": args.smoke_test,
                          "memory_entries": memory.size, "mean_evaluation_kills": result["mean_evaluation_kills"],
                          "head_mean_evaluation_kills": head["mean_evaluation_kills"] if head else None}))
        return result
    finally:
        for game in games:
            game.close()


def parser():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--output", required=True, help="New directory outside the source checkout")
    cli.add_argument("--level", default="defend_the_center", help="Bundled scenario or .cfg path")
    cli.add_argument("--device", default="cpu", help="Reader device: cpu, cuda:0, ...")
    cli.add_argument("--index-device", default="cpu", help="cpu, or cuda:N with a FAISS GPU build")
    cli.add_argument("--model", default=VENN_1.reader, help="Image-capable SentenceTransformer model ID")
    cli.add_argument("--reader-mode", choices=("portable", "native"), default="portable")
    cli.add_argument("--compile-norms", action="store_true", help="Native CUDA reader only; compilation is excluded from historical speed")
    cli.add_argument("--revision", default=None, help="Immutable 40-character model commit")
    cli.add_argument("--download-reader", action="store_true", help="Explicitly permit fetching pinned reader assets")
    cli.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    cli.add_argument("--layout", choices=("whole", "thirds"), default="thirds")
    cli.add_argument("--games", type=int, default=8)
    cli.add_argument("--decisions", type=int, default=20000, help="Total moves to make during training")
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--k", type=int, default=11)
    cli.add_argument("--gamma", type=float, default=.99)
    cli.add_argument("--frame-skip", type=int, default=4)
    cli.add_argument("--episode-timeout", type=int, default=2100)
    cli.add_argument("--eval-episodes", type=int, default=50)
    cli.add_argument("--exact-match", action=argparse.BooleanOptionalAction, default=True)
    cli.add_argument("--load-memory", help="Reuse a matching memory; start fresh game episodes")
    cli.add_argument("--head", choices=("none", "bc", "awr"), default="none",
                     help="Compile the final memory into a head with this objective and evaluate it head-only")
    cli.add_argument("--head-every", type=int, default=0,
                     help="Re-register the head every N training decisions (a multiple of --games) and serve it from then on")
    cli.add_argument("--record", action="store_true", help="Record the first final evaluation episode")
    cli.add_argument("--smoke-test", action="store_true", help="Four software-check steps without a semantic reader")
    return cli


if __name__ == "__main__":
    args = parser().parse_args()
    if args.model == VENN_1.reader and args.revision is None:
        args.revision = VENN_1.revision
    if args.smoke_test:
        args.games, args.decisions, args.eval_episodes = 1, 4, 0
    if (not 1 <= args.games <= 128 or args.decisions <= 0 or not 1 <= args.k <= 1024
            or not 0 <= args.gamma <= 1 or args.frame_skip <= 0 or args.episode_timeout <= 0
            or args.eval_episodes < 0 or (not args.smoke_test and args.revision is None)
            or args.head_every < 0 or (args.head_every and (args.head == "none" or args.head_every % args.games))
            or (args.device == "cpu" and args.dtype == "float16")
            or (args.reader_mode == "native" and not args.smoke_test and (
                not args.device.startswith("cuda:") or args.model != VENN_1.reader
                or args.revision != VENN_1.revision or args.dtype != "bfloat16"))
            or (args.compile_norms and args.reader_mode != "native")):
        raise SystemExit("Invalid configuration; pin the reader revision and use float32 on CPU")
    run(args)
