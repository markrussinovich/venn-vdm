"""Translate Decisions API requests (choice, score, noul) into native Venn questions and back."""

import json
import uuid

import numpy as np

from venn.errors import ContractError
from venn.reader import REVISION


MODEL = "venn-1"
ALIASES = (MODEL, "venn-latest", "venn-preview")
EXTENSIONS = {
    "nearest_outcome": "nearest_outcome",
    "nearest_reference": "NearestReference",
    "set_membership": "SetMembership",
    "population_shift": "PopulationShift",
    "index_apply": "IndexApply",
}


def typed_media(value):
    """Return ``True`` if ``value`` contains a typed media part such as ``{"type": "image", "image": ...}`` anywhere inside."""
    if isinstance(value, dict):
        kind = value.get("type")
        if isinstance(kind, str) and kind in ("image", "image_url", "audio", "input_audio", "image_ref") and kind in value:
            return True
        return any(typed_media(item) for item in value.values())
    return isinstance(value, list) and any(typed_media(item) for item in value)


def text(value):
    """Return ``value`` as text: strings pass through; dicts and lists become sorted JSON.

    Raises ``ContractError`` for empty strings, other types, typed media parts or nonfinite JSON.
    """
    if isinstance(value, str):
        if not value:
            raise ContractError("decision_text", "Text must not be empty")
        return value
    if not isinstance(value, (dict, list)):
        raise ContractError("decision_text", "Use text or a JSON object or array")
    if typed_media(value):
        raise ContractError("media_parts_unsupported", "Typed media parts cannot be serialized as text", 400)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        raise ContractError("decision_text", "Provide finite JSON content")


def validate_media(state, experimental_media):
    """Return ``True`` when ``state`` carries raw ``image`` or ``audio`` media.

    Raises ``ContractError`` if media is present without the ``experimental_media`` opt-in, or if typed
    media parts are used instead of one raw payload.
    """
    parts = typed_media(state)
    media = parts or isinstance(state, dict) and any(key in state for key in ("image", "audio"))
    if media and not experimental_media:
        raise ContractError("experimental_media", "Set X-Venn-Experimental: media to opt in", 400)
    if parts:
        raise ContractError("media_parts_unsupported", "Use a single raw image/audio state, not typed media parts", 400)
    return media


def translate_question(question):
    """Convert one Decisions API question into a native Venn question.

    Returns ``(native_question, mapping)``; ``mapping`` is ``None`` for Venn extension types and otherwise
    records the kind, client labels, internal labels and null (abstain) criteria needed to translate the
    answer back. Raises ``ContractError`` for malformed questions.
    """
    if not isinstance(question, dict):
        raise ContractError("question_shape", "Each question must be an object")
    kind = question.get("type")
    if not isinstance(kind, str):
        raise ContractError("question_type", "Question type must be a string")
    if kind in EXTENSIONS:
        result = {**question, "type": EXTENSIONS[kind]}
        if "store" in result and kind != "nearest_outcome":
            result["index_ref"] = result.pop("store")
        if "set" in result:
            result["members"] = result.pop("set")
        return result, None
    if kind not in ("choice", "score", "noul"):
        raise ContractError("question_type", "Use choice, score, noul or a Venn extension")
    if kind == "score" and question.get("family") and question.get("criteria") is None:
        # A Score question that names a regression family is answered by that family's head.
        return {"type": "Score", "family": question["family"]}, {"type": "score_family"}
    if "instructions" not in question:
        raise ContractError("instructions", "Instructions are required")
    instructions = text(question["instructions"])
    criteria = question.get("criteria")
    if kind == "noul":
        if criteria is None:
            criteria = {"false": f"No. {instructions}", "true": f"Yes. {instructions}"}
        if not isinstance(criteria, dict) or set(criteria) != {"false", "true"}:
            raise ContractError("criteria", "Noul criteria need true and false")
    elif kind == "score":
        if not isinstance(criteria, list) or not 1 <= len(criteria) <= 256:
            raise ContractError("criteria", "Score criteria need 1..256 ordered levels")
        criteria = {str(i): value for i, value in enumerate(criteria)}
    if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 256:
        raise ContractError("criteria", "Choice criteria need 1..256 named options")
    if not all(isinstance(label, str) and label for label in criteria):
        raise ContractError("criteria", "Criteria names must be nonempty strings")
    nulls = [label for label, value in criteria.items() if value is None]
    if nulls and (kind != "choice" or len(nulls) != 1):
        raise ContractError("criteria", "Only Choice accepts one null abstention criterion")
    labels = [label for label in criteria if label not in nulls]
    if not labels:
        raise ContractError("criteria", "Provide at least one described option")
    # Internal IDs keep arbitrary client labels, including 'none', unambiguous.
    native_labels = labels if question.get("family") else [f"option-{i}" for i in range(len(labels))]
    result = {
        "type": "Choice",
        "instructions": instructions,
        "options": [
            {"label": native, "text": text(criteria[label])}
            for native, label in zip(native_labels, labels)
        ],
    }
    if question.get("family"):
        result["family"] = question["family"]
    return result, {"type": kind, "labels": labels, "native_labels": native_labels,
                    "registered_family": bool(question.get("family")),
                    "nulls": nulls, "criteria": criteria}


def decide(service, request, experimental_media=False):
    """Answer a Decisions API request with ``service`` and return the response in Decisions format.

    Validates the model alias, 1..32 named questions and the state, embeds option criteria, then
    renormalizes each answer over the client's labels. Raises ``ContractError`` for invalid requests.
    """
    if not isinstance(request, dict) or request.get("model") not in ALIASES:
        raise ContractError("decision_model", f"Use {MODEL}", 400)
    questions = request.get("questions")
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 32:
        raise ContractError("question_limit", "Provide 1..32 named questions")
    if not all(isinstance(name, str) and name for name in questions):
        raise ContractError("question_name", "Question names must be nonempty strings")
    state = request.get("state")
    if request.get("assets"):
        raise ContractError("assets_unsupported", "Referenced image assets are not supported", 400)
    media = validate_media(state, experimental_media)
    native_state = state if isinstance(state, dict) and any(
        key in state for key in ("text", "texts", "vector", "vectors", "index_ref", "image", "audio")
    ) else text(state)
    if isinstance(native_state, dict) and (
        native_state.get("profile", MODEL) != MODEL or
        native_state.get("revision", REVISION) != REVISION
    ):
        raise ContractError("decision_profile", "Re-embed and re-register to migrate profiles", 400)
    native_questions, mappings = {}, {}
    for name, question in questions.items():
        translated, mapping = translate_question(question)
        if mapping is not None and "options" in translated:
            translated["options"] = [
                {"label": option["label"], "vector": service.vector(option["text"]).tolist()}
                for option in translated["options"]
            ]
        native_questions[name], mappings[name] = translated, mapping
    response = service.decide({"state": native_state, "questions": native_questions})
    answers = {}
    for name, raw in response["answers"].items():
        mapping = mappings[name]
        if mapping is None:
            answers[name] = {**raw, "type": questions[name]["type"]}
            continue
        if mapping["type"] == "score_family":
            if "rows" in raw:
                answers[name] = {"type": "score", "rows": raw["rows"]}
            else:
                answers[name] = {"type": "score", "score": raw["score"], "interval": raw["interval"],
                                 "interval_level": raw["interval_level"], "calibrated": raw["calibrated"],
                                 "source": raw["source"]}
            continue
        labels, native = mapping["labels"], mapping["native_labels"]
        probabilities = np.array([raw["probabilities"][label] for label in native], dtype=np.float64)
        none = float(raw["probabilities"]["none"])
        if mapping["nulls"]:
            labels = labels + mapping["nulls"]
            probabilities = np.append(probabilities, none)
        mass = float(probabilities.sum())
        if mass <= 0 or not np.isfinite(mass):
            raise ContractError("decision_distribution", "No finite option probability mass", 500)
        probabilities /= mass
        distribution = dict(zip(labels, probabilities.tolist()))
        metadata = {
            "calibrated": raw["calibrated"], "none_probability": none,
            "abstained": raw["answer"] == "none",
            "probabilities_conditioned_on_not_none": not bool(mapping["nulls"]),
        }
        kind = mapping["type"]
        if kind == "noul":
            answers[name] = {"type": kind, "noul": distribution["true"], "venn": metadata}
        elif kind == "score":
            answers[name] = {
                "type": kind, "score": float(probabilities @ np.arange(len(labels))),
                "confidence": float(probabilities.max()), "probabilities": distribution,
                "legend": mapping["criteria"],
                "venn": {**metadata, "experimental": not mapping["registered_family"]},
            }
        else:
            answers[name] = {
                "type": kind, "choice": labels[int(probabilities.argmax())],
                "confidence": float(probabilities.max()), "probabilities": distribution,
                "venn": metadata,
            }
    return {
        "id": f"venn-{uuid.uuid4().hex}", "model": MODEL, "provider": "Venn",
        "answers": answers,
        "usage": {"input_tokens": 0, "output_tokens": 0, "cost": 0},
        "venn": {
            "profile": MODEL, "reader": response["model"],
            "vector": response["vector"], "experimental_media": bool(media),
            "vector_tag": response.get("vector_tag"),
            "usage_note": "Local inference; token accounting is not instrumented",
        },
    }
