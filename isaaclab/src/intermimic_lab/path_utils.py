"""Helpers for resolving paths to InterMimic assets after the repo restructure."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    """Return the unique paths while preserving order."""
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.resolve()
        key = str(resolved)
        if key in seen:
            continue
        unique.append(resolved)
        seen.add(key)
    return unique


def _candidate_repo_roots() -> list[Path]:
    """Potential locations of the InterMimic repo."""
    roots: list[Path] = []
    env_root = os.environ.get("INTERMIMIC_PATH")
    if env_root:
        roots.append(Path(env_root).expanduser())
    # Fallback to repo root inferred from this module location.
    roots.append(Path(__file__).resolve().parents[3])
    return _unique_paths(roots)


def resolve_data_path(*relative_parts: str, must_exist: bool = True) -> Path:
    """Resolve a path under ``intermimic/data`` supporting legacy + new layouts.

    Args:
        *relative_parts: Path pieces relative to ``intermimic/data``.
        must_exist: Whether to raise if the path is not found.

    Returns:
        The first existing matching path, or the primary candidate if
        ``must_exist`` is False.
    """
    relative = Path(*relative_parts)
    data_dirs = [
        Path("isaacgym") / "src" / "intermimic" / "data",
        Path("intermimic") / "data",
    ]

    candidates: list[Path] = []
    for repo_root in _candidate_repo_roots():
        for data_dir in data_dirs:
            candidate = (repo_root / data_dir / relative).resolve()
            if candidate not in candidates:
                candidates.append(candidate)
            if candidate.exists():
                return candidate

    if not candidates:
        raise FileNotFoundError("Unable to construct candidate paths for InterMimic data.")

    if must_exist:
        search_paths = "\n".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Could not find InterMimic data path '{relative}'. Looked in:\n{search_paths}"
        )
    return candidates[0]
