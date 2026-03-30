from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _normalize_rel(repo_root: Path, path_str: str) -> Path:
    raw = (path_str or "").strip().replace("\\", "/")
    if not raw or raw.startswith("..") or "/../" in f"/{raw}/":
        raise ValueError(f"Invalid path: {path_str}")
    candidate = (repo_root / raw).resolve()
    root_resolved = repo_root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Path outside repo: {path_str}") from exc
    return candidate


def load_resume_entries(repo_root: Path) -> List[Dict[str, Any]]:
    """Load resume entries from Resume/resume_manifest.json if present; else scan Resume/ recursively."""
    resume_dir = repo_root / "Resume"
    if not resume_dir.exists():
        return []

    manifest_path = resume_dir / "resume_manifest.json"
    if manifest_path.exists():
        return _from_manifest(repo_root, manifest_path)

    return _scan_folder(repo_root, resume_dir)


def _from_manifest(repo_root: Path, manifest_path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries_raw = payload.get("entries") or []
    default_path = (payload.get("default_path") or "").strip().replace("\\", "/")

    entries: List[Dict[str, Any]] = []
    for idx, row in enumerate(entries_raw):
        path_str = (row.get("path") or row.get("relative_path") or "").strip().replace("\\", "/")
        if not path_str:
            continue
        abs_path = _normalize_rel(repo_root, path_str)
        if not abs_path.is_file():
            continue
        suffix = abs_path.suffix.lower()
        if suffix not in {".pdf", ".docx"}:
            continue
        tags = row.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        tags = [str(t).strip() for t in tags if str(t).strip()]
        rel = str(abs_path.relative_to(repo_root)).replace("\\", "/")
        parent = abs_path.parent.name if abs_path.parent != (repo_root / "Resume") else "root"
        entries.append(
            {
                "id": f"entry_{idx}",
                "relative_path": rel,
                "tags": list(dict.fromkeys(tags + [parent.lower(), abs_path.stem.lower(), "resume"])),
                "folder_segment": parent,
            }
        )

    if default_path:
        try:
            abs_default = _normalize_rel(repo_root, default_path)
        except ValueError:
            abs_default = None
        if abs_default and abs_default.is_file():
            rel_d = str(abs_default.relative_to(repo_root)).replace("\\", "/")
            if not any(e["relative_path"] == rel_d for e in entries):
                entries.insert(
                    0,
                    {
                        "id": "entry_default",
                        "relative_path": rel_d,
                        "tags": ["default", "resume"],
                        "folder_segment": abs_default.parent.name,
                    },
                )

    for i, e in enumerate(entries):
        e["id"] = f"entry_{i}"
    return entries


def _scan_folder(repo_root: Path, resume_dir: Path) -> List[Dict[str, Any]]:
    found: List[Path] = []
    for p in resume_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".pdf", ".docx"}:
            continue
        if p.name.lower() == "resume_manifest.json":
            continue
        found.append(p)

    found.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    entries: List[Dict[str, Any]] = []
    for idx, abs_path in enumerate(found):
        rel = str(abs_path.relative_to(repo_root)).replace("\\", "/")
        rel_under = abs_path.relative_to(resume_dir)
        parts = list(rel_under.parts[:-1])
        tags: List[str] = [abs_path.stem.lower(), "resume"]
        for part in parts:
            cleaned = part.lower().replace(" ", "_")
            if cleaned not in tags:
                tags.append(cleaned)
        parent = abs_path.parent.name if abs_path.parent != resume_dir else "root"
        entries.append(
            {
                "id": f"entry_{idx}",
                "relative_path": rel,
                "tags": tags,
                "folder_segment": parent,
            }
        )
    return entries
