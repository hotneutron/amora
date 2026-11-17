#!/usr/bin/env python3
"""Run Python with a recorded, host-specific TPU topology configuration."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ALLOWED = {"TPU_SKIP_MDS_QUERY", "TPU_HOST_BOUNDS", "TPU_CHIPS_PER_HOST_BOUNDS",
           "TPU_TOPOLOGY", "TPU_WORKER_ID", "TPU_WORKER_HOSTNAMES", "TPU_VISIBLE_CHIPS",
           "TPU_PROCESS_BOUNDS", "TPU_CHIPS_PER_PROCESS_BOUNDS", "TPU_TOPOLOGY_ALT", "TPU_TOPOLOGY_WRAP",
           "TPU_LOG_DIR"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("hostname") != socket.gethostname():
        parser.error("instance configuration belongs to a different host")
    settings = config.get("environment", {})
    if not isinstance(settings, dict) or not set(settings) <= ALLOWED or any(not isinstance(v,str) for v in settings.values()):
        parser.error("invalid TPU environment configuration")
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if not arguments: parser.error("Python command arguments are required")
    env = dict(os.environ, **settings, AMORA_TPU_INSTANCE_CONFIG=str(args.config.resolve()))
    return subprocess.run([args.python, *arguments], env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
