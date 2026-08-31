#!/usr/bin/env python3
"""Mirror a remote run locally, then require matching SHA-256 checksums."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.research_protocol import sha256_file, write_json


def remote_checksums(host: str, directory: str) -> dict[str, str]:
    command = (
        f"cd {shlex.quote(directory)} && "
        "find . -type f -print0 | sort -z | xargs -0 sha256sum"
    )
    output = subprocess.check_output(["ssh", host, command], text=True)
    checksums = {}
    for line in output.splitlines():
        checksum, relative = line.split(maxsplit=1)
        checksums[relative.lstrip("*").removeprefix("./")] = checksum
    return checksums


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="gpu5090")
    parser.add_argument("--remote-dir", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"/?[A-Za-z0-9._/-]+", args.remote_dir):
        raise ValueError("Remote path contains unsupported characters")
    if "ESAI_RA_ZHANTORE" not in args.remote_dir or args.remote_dir.rstrip("/").endswith("ESAI_RA_ZHANTORE"):
        raise ValueError("Remote path must be a scoped project directory inside ESAI_RA_ZHANTORE")
    local_dir = args.local_dir.resolve()
    if PROJECT_ROOT not in local_dir.parents:
        raise ValueError("Local mirror must be inside this repository")
    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync", "-az", "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=",
            f"{args.host}:{args.remote_dir.rstrip('/')}/", f"{local_dir}/",
        ],
        check=True,
    )
    remote = remote_checksums(args.host, args.remote_dir)
    local = {
        str(path.relative_to(local_dir)): sha256_file(path)
        for path in sorted(local_dir.rglob("*"))
        if path.is_file() and path.name != "mirror-manifest.json"
    }
    if remote != local:
        missing = sorted(remote.keys() - local.keys())
        extra = sorted(local.keys() - remote.keys())
        mismatched = sorted(key for key in remote.keys() & local.keys() if remote[key] != local[key])
        raise RuntimeError(
            f"Mirror verification failed: missing={missing}, extra={extra}, mismatched={mismatched}"
        )
    write_json(local_dir / "mirror-manifest.json", {
        "host_alias": args.host,
        "remote_directory": args.remote_dir,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(local),
        "checksums": local,
    })
    print(json.dumps({"verified": True, "file_count": len(local)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
