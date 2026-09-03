"""
Standalone CLI ingestion — useful for quickly building a persistent, shared
collection (e.g. for local manual testing) without going through the
Streamlit UI's per-session collections.

Usage:
    python -m scripts.seed https://example.com/docs/page1 https://example.com/docs/page2
    python -m scripts.seed --collection demo https://example.com
"""

from __future__ import annotations

import argparse
import sys

from src.rag import pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urls", nargs="+", help="One or more URLs to ingest")
    parser.add_argument("--collection", default="cli-seed", help="Collection id to ingest into")
    args = parser.parse_args()

    print(f"Ingesting {len(args.urls)} URL(s) into collection '{args.collection}'...")
    outcomes = pipeline.ingest_urls(args.urls, collection_id=args.collection)

    failures = 0
    for outcome in outcomes:
        if outcome.success:
            print(f"  OK   {outcome.source} -> {outcome.chunk_count} chunks")
        else:
            failures += 1
            print(f"  FAIL {outcome.source}: {outcome.error}")

    print(f"\nDone. {len(outcomes) - failures}/{len(outcomes)} sources ingested.")
    return 1 if failures == len(outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
