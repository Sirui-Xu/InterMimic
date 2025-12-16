"""Helpers for resolving paths within the InterMimic repository."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def _unique(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        key = str(resolved)
        if key in seen:
            continue
        ordered.append(resolved)
        seen.add(key)
    return ordered


def _repo_root_candidates() -> list[Path]:
    env_root = os.environ.get("INTERMIMIC_PATH")
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.append(Path(__file__).resolve().parents[4])
    return _unique(candidates)


def _legacy_candidates(root: Path) -> list[Path]:
    """Legacy locations that existed before the repo cleanup."""
    return [
        root / "intermimic",
    ]


def _canonical_candidates(root: Path) -> list[Path]:
    return [
        root / "isaacgym" / "src" / "intermimic",
    ]


def resolve_repo_path(path_str: str, must_exist: bool = True) -> Path:
    """Resolve a path relative to the repository.

    Args:
        path_str: A relative path (e.g., ``intermimic/data/assets``) or absolute path.
        must_exist: Whether to raise if no candidate exists.

    Returns:
        A ``Path`` pointing to the resolved location.
    """
    path = Path(path_str)
    if path.is_absolute():
        if not must_exist or path.exists():
            return path
        raise FileNotFoundError(f"Path does not exist: {path}")

    candidates: list[Path] = []
    for root in _repo_root_candidates():
        # Direct relative path
        candidates.append((root / path).resolve())

        # If the path begins with "intermimic", also try the canonical tree
        parts = path.parts
        if parts and parts[0] == "intermimic":
            child = Path(*parts[1:]) if len(parts) > 1 else Path(".")
            candidates.append((root / "isaacgym" / "src" / "intermimic" / child).resolve())

    if not must_exist:
        return candidates[0]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve path '{path_str}'. Looked in:\n" + "\n".join(str(c) for c in candidates)
    )


def resolve_data_path(*relative_parts: str, must_exist: bool = True) -> Path:
    """Resolve a path inside ``intermimic/data`` accounting for legacy + canonical layouts."""
    relative = Path(*relative_parts)
    candidates: list[Path] = []
    for root in _repo_root_candidates():
        for base in _canonical_candidates(root):
            candidates.append((base / "data" / relative).resolve())
        for base in _legacy_candidates(root):
            candidates.append((base / "data" / relative).resolve())

    if not must_exist:
        return candidates[0]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find data path '{relative}'. Looked in:\n" + "\n".join(str(c) for c in candidates)
    )
