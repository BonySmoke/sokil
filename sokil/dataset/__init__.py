"""
The annotation dataset: what was recorded, and what the shuttle did.

Two hand-maintained tables (clips.csv, events.csv) plus clip-level splits. See
schema.py for what each column means and why it lives where it does.
"""

from .court import UnknownVertex, resolve_vertex
from .schema import Call, Line, PointSource, columns
from .store import SPLITS, Dataset, DatasetError

__all__ = [
    "SPLITS",
    "Call",
    "Dataset",
    "DatasetError",
    "Line",
    "PointSource",
    "UnknownVertex",
    "columns",
    "resolve_vertex",
]
