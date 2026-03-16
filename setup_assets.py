#!/usr/bin/env python3
# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
# Download external Nexerra runtime assets into their expected repo paths.
# ---
# This script is intended for artifacts that are too large to version in Git,
# such as checkpoints, latent banks, and tokenized dataset bundles
# ---------------------------------------

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


REPO_ROOT = Path(__file__).resolve().parent

# --- Default external assets expected for inference/runtime ---
DEFAULT_FILES = [
    {"dest": "artifacts/ckpt/vae/no_prop_vae_epoch_120.pt"},
    {"dest": "artifacts/ckpt/flow/otcfm_step_180000.pt"},
    {"dest": "artifacts/latent_banks/latent_bank.pt"},
    {"dest": "artifacts/latent_banks/latent_bank_len.pt"},
    {"dest": "data/processed/tokenized_dataset.pkl"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Download Nexerra runtime assets from Zenodo")
    
    parser.add_argument(
        "--base-url",
        default = os.environ.get("NEXERRA_ASSET_BASE_URL"),
        help = ("Base URL that serves the asset files. Example: "
                "https://zenodo.org/records/<record-id>/files"),
    )
    
    parser.add_argument(
        "--manifest",
        type = Path,
        help = ("Optional JSON manifest overriding the default asset list. "
            "Entries must include 'dest' and may include 'remote_name' or 'url'."),
    )
   
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files even if the destination already exists.",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved transfers without downloading anything.",
    )
    
    parser.add_argument(
        "--timeout",
        type = int,
        default = 500,
        help = "Network timeout in seconds per file.",
    )
    return parser.parse_args()


def load_manifest(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return list(DEFAULT_FILES)
    data = json.loads(path.read_text(encoding = "utf-8"))
    if not isinstance(data, list):
        raise ValueError("Manifest must be a JSON list.")
    files: list[dict[str, str]] = []
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict) or "dest" not in entry:
            raise ValueError(f"Manifest entry {idx} must be an object with a 'dest' field.")
        files.append({str(k): str(v) for k, v in entry.items()})
    return files


def resolve_url(base_url: str | None, entry: dict[str, str]) -> str:
    if "url" in entry:
        return entry["url"]
    if not base_url:
        raise ValueError("A --base-url (or NEXERRA_ASSET_BASE_URL) is required unless manifest entries include explicit 'url' values.")
    remote_name = entry.get("remote_name") or Path(entry["dest"]).name
    return f"{base_url.rstrip('/')}/{urllib.parse.quote(remote_name)}"


def ensure_dest_ready(dest: Path, overwrite: bool) -> bool:
    dest.parent.mkdir(parents = True, exist_ok = True)
    if dest.is_file() and not overwrite:
        print(
            f"{Colors.CYAN}Skipping existing file:{Colors.RESET} {dest} "
            f"(use {Colors.BOLD}--overwrite{Colors.RESET} to replace)"
        )
        return False
    return True


def download_file(url: str, dest: Path, timeout: int) -> None:
    print(f"{Colors.BLUE}Downloading:{Colors.RESET} {url}")
    print(f"{Colors.BLUE}Destination:{Colors.RESET} {dest}")
    with urllib.request.urlopen(url, timeout=timeout) as response, dest.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk: break
            out.write(chunk)
    print(f"{Colors.GREEN}Ready:{Colors.RESET} {dest}\n")


def main() -> int:
    args = parse_args()
    try:
        files = load_manifest(args.manifest)
    except Exception as exc:
        print(f"{Colors.RED}Manifest error:{Colors.RESET} {exc}", file=sys.stderr)
        return 2

    print(f"{Colors.HEADER}{Colors.BOLD}=== Nexerra Asset Setup ==={Colors.RESET}")
    if args.base_url:
        print(f"{Colors.BOLD}Base URL:{Colors.RESET} {args.base_url}")
    elif not any("url" in entry for entry in files):
        print(
            f"{Colors.RED}Error:{Colors.RESET} no --base-url provided and manifest entries do not define explicit URLs.",
            file=sys.stderr,
        )
        return 2
    print()

    failures = 0
    for entry in files:
        dest = REPO_ROOT / entry["dest"]
        try:
            url = resolve_url(args.base_url, entry)
            if not ensure_dest_ready(dest, args.overwrite):
                continue
            if args.dry_run:
                print(f"{Colors.YELLOW}Dry run:{Colors.RESET} {url} -> {dest}")
                continue
            download_file(url, dest, args.timeout)
        except urllib.error.HTTPError as exc:
            failures += 1
            print(f"{Colors.RED}HTTP error:{Colors.RESET} {exc.code} for {entry['dest']} ({exc.reason})", file=sys.stderr)
        except urllib.error.URLError as exc:
            failures += 1
            print(f"{Colors.RED}URL error:{Colors.RESET} {entry['dest']} ({exc.reason})", file=sys.stderr)
        except Exception as exc:
            failures += 1
            print(f"{Colors.RED}Error:{Colors.RESET} {entry['dest']} ({exc})", file=sys.stderr)

    if failures:
        print(f"{Colors.YELLOW}{failures} file(s) failed.{Colors.RESET}", file=sys.stderr)
        return 1

    print(f"{Colors.GREEN}{Colors.BOLD} All requested assets are ready.{Colors.RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
