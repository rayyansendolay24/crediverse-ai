"""
study/seed_articles.py
───────────────────────────
Bulk-loads curated study articles into study.db from plain text files,
so you don't have to call add_article() by hand for every article.

Expected folder layout (create this yourself, next to the project root):

    study_articles/
      human/
        story1.txt
        story2.txt
      ai/
        story1.txt
        story2.txt

Each .txt file's FIRST LINE is treated as the title, and everything
after a blank line is the body text. Example file content:

    Local council approves new bus route

    The city council voted 6-2 on Tuesday to approve a new bus route
    connecting the east side to downtown, after months of public
    hearings...

Files are matched by name across human/ and ai/ so pairs on the same
story get the same `topic` tag (the filename stem) - useful if you
built this set as same-story human/AI pairs, which is the strongest
design for the study (see README "Running the reader study").
Unmatched files are still loaded, just without a shared topic tag.

Usage:
    python -m study.seed_articles study_articles/
    python -m study.seed_articles study_articles/ --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from study.database import add_article, init_db


def _parse_file(path: Path) -> tuple[str, str]:
    raw = path.read_text(encoding="utf-8").strip()
    parts = raw.split("\n\n", 1)
    if len(parts) == 2:
        title, body = parts[0].strip(), parts[1].strip()
    else:
        # No blank-line separator found - fall back to first line as title.
        lines = raw.split("\n", 1)
        title = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else raw
    return title, body


def seed_from_folder(root: Path, dry_run: bool = False) -> int:
    init_db()
    count = 0
    for origin in ("human", "ai"):
        folder = root / origin
        if not folder.is_dir():
            print(f"  [seed] skipping '{origin}' - no folder at {folder}")
            continue
        for path in sorted(folder.glob("*.txt")):
            title, body = _parse_file(path)
            if len(body.split()) < 30:
                print(f"  [seed] WARNING: '{path}' has very little body text ({len(body.split())} words) - loading anyway")
            topic = path.stem
            if dry_run:
                print(f"  [dry-run] would add {origin:5s} topic={topic!r} title={title!r} ({len(body)} chars)")
            else:
                article_id = add_article(title=title, body_text=body, actual_origin=origin, topic=topic)
                print(f"  [seed] added {origin:5s} topic={topic!r} -> {article_id}")
            count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="Path to the study_articles/ folder (containing human/ and ai/ subfolders)")
    parser.add_argument("--dry-run", action="store_true", help="Preview what would be loaded without writing to the database")
    args = parser.parse_args()

    root = Path(args.folder)
    if not root.is_dir():
        print(f"Error: '{root}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    n = seed_from_folder(root, dry_run=args.dry_run)
    print(f"\n{'Would load' if args.dry_run else 'Loaded'} {n} article(s).")
