"""Strict public enums for the valise facade.

Each enum's ``value`` is the exact canonical string the native parser already
accepts (``parse_metric`` / ``parse_lang`` / ``parse_durability`` /
``parse_rerank`` / ``parse_tf_mode`` / ``parse_design``). The facade passes
``enum.value`` to the native layer; raw strings are rejected at the typed
boundary.

Only one canonical spelling is exposed per variant. The native parsers accept
several aliases (``ip``, ``euclidean``, ``fsync``, ...), but the public
surface presents a single, strict spelling.
"""

from __future__ import annotations

import enum

__all__ = ["Metric", "Lang", "Durability", "Rerank", "TfMode", "Design"]


class _StrEnum(enum.Enum):
    """Base enum whose ``str()`` is its value, so a stray ``str()`` is harmless.

    The facade still passes ``enum.value`` explicitly; this only guards against
    accidental ``str(enum)`` interpolation in user code.
    """

    def __str__(self) -> str:  # noqa: D105 - trivial
        return str(self.value)


class Metric(_StrEnum):
    """Vector-space distance metric."""

    COSINE = "cosine"
    DOT = "dot"
    L2 = "l2"


class Lang(_StrEnum):
    """Text-space analyzer language."""

    ENGLISH = "english"
    RAW = "raw"


class Durability(_StrEnum):
    """Commit durability mode.

    ``FULL_SYNC`` maps to the Rust ``Durability::FullSync`` (the strongest
    hardware barrier, ``F_FULLFSYNC`` on macOS).
    """

    BUFFERED = "buffered"
    FULL_SYNC = "full_sync"
    SYNC_ALL = "sync_all"


class Rerank(_StrEnum):
    """Vector-search rerank fidelity. ``ACCURATE`` (full f32 rerank) is the
    default, mirroring Rust ``Rerank::Accurate``."""

    FAST = "fast"
    ACCURATE = "accurate"


class TfMode(_StrEnum):
    """Term-frequency weighting mode for TF-IDF scoring."""

    RAW = "raw"
    LOG = "log"


class Design(_StrEnum):
    """UPQ ring-design source (mirror of Rust ``UpqDesign``)."""

    EMPIRICAL = "empirical"
    RAYLEIGH = "rayleigh"
