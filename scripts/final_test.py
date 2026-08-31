#!/usr/bin/env python3
"""Consume the series' one-shot test capability for a locked champion."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.research_protocol import authorize_final_test, load_protocol


def remove_option(command: list[str], option: str, *, takes_value: bool = True) -> list[str]:
    while option in command:
        index = command.index(option)
        del command[index:index + 2 if takes_value else index + 1]
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-config", type=Path, required=True)
    parser.add_argument("--champion-manifest", type=Path, required=True)
    args = parser.parse_args()
    protocol, protocol_hash = load_protocol(args.research_config)
    state_dir = (
        PROJECT_ROOT / "results" / "autoresearch" / protocol["series"]["id"] / "final-test"
    )
    champion = authorize_final_test(
        args.champion_manifest,
        protocol=protocol,
        protocol_hash=protocol_hash,
        final_state_dir=state_dir,
    )
    checkpoint = Path(champion["selected_checkpoint"])
    command = list(champion["command"])
    if command[0] != sys.executable:
        command[0] = sys.executable
    for option, takes_value in (
        ("--robustness-input", True),
        ("--checkpoint-path", True),
        ("--final-test-manifest", True),
        ("--temperature", True),
        ("--eval-only", False),
    ):
        remove_option(command, option, takes_value=takes_value)
    command.extend([
        "--eval-only",
        "--checkpoint-path", str(checkpoint),
        "--final-test-manifest", str(args.champion_manifest),
        "--temperature", str(champion["temperature"]),
    ])
    environment = os.environ.copy()
    environment["CEFR_FINAL_TEST_AUTH"] = hashlib.sha256(
        f"{protocol['series']['id']}:{protocol_hash}".encode("utf-8")
    ).hexdigest()
    return subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
