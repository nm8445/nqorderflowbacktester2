"""
Base classes for research studies.

Each study is a subclass of BaseStudy.  Class-level attributes define metadata
and parameter schema.  Implement run() to add study logic; stubs return a
"not yet implemented" placeholder with the received parameters echoed back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Param:
    """Descriptor for one configurable study parameter."""

    name: str
    label: str
    ptype: str          # 'float' | 'int' | 'bool' | 'choice'
    default: Any
    description: str = ""
    mn: float | None = None
    mx: float | None = None
    step: float | None = None
    choices: list[str] | None = None


@dataclass
class StudyResult:
    """Returned by BaseStudy.run()."""

    summary: str
    tables: list[tuple[str, pd.DataFrame]] = field(default_factory=list)


class BaseStudy:
    """
    Override class attributes and implement run() in each study subclass.

    Attributes
    ----------
    id          : Short identifier, e.g. "1.1"
    name        : Human-readable name
    block       : Block label, e.g. "Block 1 — Value Area Behavior"
    description : Full study description (markdown supported)
    variables   : Ordered list of Param descriptors shown in the UI
    """

    id: str = ""
    name: str = ""
    block: str = ""
    description: str = ""
    variables: list[Param] = []

    def run(self, start_date: str, end_date: str, **params) -> StudyResult:
        """
        Execute the study over [start_date, end_date] with the given params.
        Default implementation returns a not-yet-implemented stub.
        """
        param_lines = "\n".join(
            f"  {k}: {v}" for k, v in params.items()
        ) or "  (none)"
        summary = (
            f"Study {self.id} — {self.name}\n"
            f"{'=' * 60}\n"
            f"Date range : {start_date} → {end_date}\n"
            f"Parameters :\n{param_lines}\n\n"
            f"[NOT YET IMPLEMENTED]\n"
            f"This study is registered in the runner but its analysis\n"
            f"logic has not been written yet.  Parameters and UI are\n"
            f"fully configured — implement run() in the study class to\n"
            f"activate it.\n"
        )
        return StudyResult(summary=summary)
