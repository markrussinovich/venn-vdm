"""Reader profiles, vector tags and the published head location.

A profile pins the reader, its revision, the vector dimension and the prompt
protocol. Vectors are only comparable within one profile and representation.
"""
from dataclasses import asdict, dataclass
import re

from venn.reader import MODEL_ID, REVISION


@dataclass(frozen=True)
class EmbedderProfile:
    """A pinned reader: ID, immutable revision, vector dimension and prompt protocol."""

    id: str
    reader: str
    revision: str
    dimensions: int
    prompt: str

    def check(self, profile, revision):
        """Raise ``ValueError`` unless ``profile`` and ``revision`` match this profile exactly."""
        if (profile, revision) != (self.id, self.revision):
            raise ValueError("Profile mismatch: re-embed and re-register; profiles are not alignments")


VENN_1 = EmbedderProfile("venn-1", MODEL_ID, REVISION, 768, "Classification")

# The published venn-1 decision head on the Hugging Face Hub, pinned to an immutable commit.
HEAD_REPO_ID = "markrussinovich/venn-1"
HEAD_REVISION = "5e512eb799058d33ca47dd1c80863e7785960160"
HEAD_SHA256 = "03c62c7fbc5f5d6b49e171084c9b64291a1201b4cf967b5abbc9ebcfa3137aa9"
HEAD_CONFIG_SHA256 = "537db9925208d18fd6ae1ff13ab417315e2f0313646a48ddc3010200b103cf95"


@dataclass(frozen=True)
class VectorTag:
    """Provenance carried by every vector: profile, revision, representation, dimension and field order.

    Profiles starting with ``experimental:`` name an alternative reader for outcome-store
    geometry only; they require a 40-character commit as the revision.
    """

    profile: str
    revision: str
    representation: str
    dimensions: int
    fields: tuple = ()

    def __post_init__(self):
        if isinstance(self.profile, str) and self.profile.startswith("experimental:"):
            if (not 14 <= len(self.profile) <= 128 or not isinstance(self.revision, str)
                    or not re.fullmatch(r"[0-9a-f]{40}", self.revision)):
                raise ValueError("Experimental readers require a named profile and immutable revision")
        else:
            VENN_1.check(self.profile, self.revision)
        if (not isinstance(self.representation, str) or not 1 <= len(self.representation) <= 128
                or type(self.dimensions) is not int or not 1 <= self.dimensions <= 16384
                or not isinstance(self.fields, tuple)
                or not all(isinstance(field, str) and field for field in self.fields)
                or len(set(self.fields)) != len(self.fields)):
            raise ValueError("Invalid vector representation tag")

    def to_dict(self):
        """Return the tag as a JSON-serializable dict with ``fields`` as a list."""
        return {**asdict(self), "fields": list(self.fields)}

    @classmethod
    def parse(cls, value):
        """Build a tag from a complete dict (or return ``value`` if it already is a tag)."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict) or set(value) != {
                "profile", "revision", "representation", "dimensions", "fields"}:
            raise ValueError("Provide the complete vector tag")
        if not isinstance(value["fields"], (list, tuple)):
            raise ValueError("Tag fields must be an ordered array")
        return cls(**{**value, "fields": tuple(value["fields"])})

    def check(self, other):
        """Raise ``ValueError`` unless ``other`` is exactly this tag."""
        if self != self.parse(other):
            raise ValueError("Vector tag mismatch: re-embed the store; layouts are not interchangeable")


TEXT_TAG = VectorTag(VENN_1.id, VENN_1.revision, "text-classification", 768)
IMAGE_TAGS = {
    "image-70": VectorTag(VENN_1.id, VENN_1.revision, "image-70", 768, ("image",)),
    "image-280": VectorTag(VENN_1.id, VENN_1.revision, "image-280", 768, ("image",)),
    "image-thirds-70": VectorTag(
        VENN_1.id, VENN_1.revision, "image-thirds-70", 2304, ("left", "center", "right")),
}
