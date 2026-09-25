"""Re-sync the vendored SteerViT inference files from upstream at a given git ref.

Fetches ``src/steervit/<file>`` for every file listed in ``.upstream-sync.yml`` from the upstream
repository (raw GitHub URLs, standard library only), applies the two local changes (provenance
header, package-relative imports) and writes the result into the vendor folder, printing which
files changed so the diff can be reviewed before committing.

    python tools/sync_vendor.py d385b7c3            # write files
    python tools/sync_vendor.py d385b7c3 --check    # only report drift, exit 1 if any
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR = REPO_ROOT / "cuvis_ai_steervit" / "_vendor" / "steervit"
UPSTREAM = "https://raw.githubusercontent.com/manugaurdl/SteerViT/{ref}/src/steervit/{name}"
FILES = ("__init__.py", "model.py", "backbone.py", "crossattention.py", "utils.py")
HEADER = (
    "# Vendored from https://github.com/manugaurdl/SteerViT (src/steervit/{name})\n"
    "# at commit {ref}; see .upstream-sync.yml for provenance and NOTICE in this folder for\n"
    "# the licence.\n"
    "# Local changes: this header and package-relative imports; nothing else.\n"
)


def fetch(ref: str, name: str) -> str:
    """Download one upstream file as text."""
    with urllib.request.urlopen(UPSTREAM.format(ref=ref, name=name), timeout=60) as resp:  # noqa: S310
        return resp.read().decode("utf-8")


def adapt(text: str, ref: str, name: str) -> str:
    """Apply the local changes: header + relative imports."""
    text = re.sub(r"^from steervit\.(\w+) import", r"from .\1 import", text, flags=re.M)
    if "from steervit" in text or "import steervit" in text:
        raise RuntimeError(f"{name}: an absolute steervit import survived the rewrite")
    return HEADER.format(name=name, ref=ref[:8]) + text


def strip_header(text: str) -> str:
    """Drop the provenance header so files from different refs compare on content."""
    lines = text.splitlines(keepends=True)
    while lines and lines[0].startswith("# "):
        lines.pop(0)
    return "".join(lines)


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref", help="upstream git ref (commit, tag or branch)")
    parser.add_argument("--check", action="store_true", help="report drift only, write nothing")
    args = parser.parse_args()
    changed = []
    for name in FILES:
        new = adapt(fetch(args.ref, name), args.ref, name)
        path = VENDOR / name
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if strip_header(old) != strip_header(new):
            changed.append(name)
            if not args.check:
                path.write_text(new, encoding="utf-8", newline="\n")
    verb = "differ from" if args.check else "rewritten from"
    print(f"{len(changed)} file(s) {verb} upstream {args.ref}: {', '.join(changed) or '-'}")
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
