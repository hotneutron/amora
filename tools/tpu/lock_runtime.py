#!/usr/bin/env python3
"""Hash a resolved wheelhouse into an offline pip requirements lock."""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, default=Path("wheelhouse"))
    parser.add_argument("--lock", type=Path, default=Path("runtime-lock.txt"))
    parser.add_argument("--manifest", type=Path, default=Path("wheel-manifest.json"))
    args = parser.parse_args()
    if args.lock.exists() or args.manifest.exists():
        parser.error("refusing to overwrite an existing runtime lock/manifest")
    records, packages = [], set()
    for path in sorted(args.wheelhouse.glob("*.whl")):
        with zipfile.ZipFile(path) as archive:
            name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
            metadata = BytesParser().parsebytes(archive.read(name))
        package = metadata["Name"].lower().replace("_","-")
        if package in packages: parser.error(f"multiple wheels for {package}; select one resolved platform wheel set")
        packages.add(package)
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8*1024*1024): hasher.update(chunk)
        records.append({"name":package,"version":metadata["Version"],"wheel":path.name,"sha256":hasher.hexdigest()})
    if not records: parser.error("wheelhouse contains no wheels")
    args.lock.parent.mkdir(parents=True,exist_ok=True)
    args.manifest.parent.mkdir(parents=True,exist_ok=True)
    with args.lock.open("x") as handle:
        for record in sorted(records,key=lambda item:item["name"]):
            handle.write(f"{record['name']}=={record['version']} --hash=sha256:{record['sha256']}\n")
    with args.manifest.open("x") as handle: json.dump(records,handle,indent=2)
    print(json.dumps({"lock":str(args.lock),"manifest":str(args.manifest),"wheels":len(records)}))


if __name__ == "__main__":
    raise SystemExit(main())
