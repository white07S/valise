# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Exception hierarchy for the valise bindings.

These are re-exported directly from the native module so that exception
*identity* is preserved: ``except valise.ValidationError`` keeps matching the
errors the native layer raises. Do not redefine these in pure Python.

``NotCalibratedError`` is raised by a vector search before the field's codec
has been calibrated (calibration happens at the first commit containing
vectors for the field, or eagerly via :class:`valise.Now`); the message names
the auto space (``~auto/{collection}/{field}``). ``SchemaMismatchError`` is
raised by a divergent re-declaration of a collection (identical re-declares
are no-ops; purely additive ones are accepted).
"""

from __future__ import annotations

from ._native import (
    BusyError,
    CorruptionError,
    NotCalibratedError,
    ValiseError,
    SchemaMismatchError,
    UnsupportedError,
    ValidationError,
)

__all__ = [
    "ValiseError",
    "CorruptionError",
    "BusyError",
    "UnsupportedError",
    "ValidationError",
    "NotCalibratedError",
    "SchemaMismatchError",
]
