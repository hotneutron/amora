"""Source-backed generation profiles, independent of a deployed TPU instance."""
from __future__ import annotations

import json
import re
from importlib.resources import files

from .artifacts import digest


def profiles():
    entries = {}
    for path in files(__package__).joinpath("profiles").iterdir():
        if path.name.endswith(".json"):
            profile = json.loads(path.read_text())
            required = {"generation", "revision", "aliases", "sources", "facts"}
            if not required <= profile.keys() or not profile["sources"]:
                raise ValueError(f"invalid architecture profile: {path.name}")
            profile["profile_digest"] = digest(profile)
            entries[profile["generation"]] = profile
    return entries


def _key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def resolve_profile(device_kind, requested=None, chip_version=None):
    catalog = profiles()
    aliases = {_key(alias): name for name, p in catalog.items()
               for alias in [name, *p["aliases"]]}
    observed = {aliases[_key(value)] for value in (device_kind, chip_version)
                if value is not None and _key(value) in aliases}
    if len(observed) > 1:
        raise ValueError("profile_conflict: runtime identities disagree")
    selected = aliases.get(_key(requested)) if requested else None
    if requested and selected is None:
        raise ValueError(f"unknown architecture profile: {requested}")
    if selected and observed and selected not in observed:
        raise ValueError("profile_conflict: requested profile disagrees with runtime")
    name = selected or next(iter(observed), None)
    return {"generation": name or "unknown", "profile": catalog.get(name),
            "resolution": "explicit" if selected else "runtime" if name else "unresolved",
            "raw_device_kind": device_kind, "raw_chip_version": chip_version}
