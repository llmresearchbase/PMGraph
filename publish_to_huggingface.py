#!/usr/bin/env python3
"""
Publish this release package to Hugging Face Hub as PMGraph v2. Not run
automatically by anything else in this package -- a deliberate, standalone
step.

What --execute does, in order:
  1. Refuses to run if placeholder markers remain in the docs (see
     PLACEHOLDER_MARKERS), unless --force is passed.
  2. Makes ONE commit to main that adds the v2 files and deletes every
     other file left over from the prior release (except .gitattributes).
     The dataset card (HF's README.md) is built from hf_card_header.md +
     DATA_CARD.md; this package's reproduction README is uploaded as
     REPRODUCE.md.
  3. Tags that commit with `--tag` (default v2).

Authentication: an HF token with write access, via HF_TOKEN or a prior
`hf auth login`. Never hardcode a token in this file.

Usage:
  python publish_to_huggingface.py --repo-id shawntzx/PMGraph            # dry run
  python publish_to_huggingface.py --repo-id shawntzx/PMGraph --execute
"""
import argparse
import csv
import json
import os
import sys

PLACEHOLDER_MARKERS = ["PROPOSED, NOT YET CONFIRMED", "[TODO"]
CHECKED_FILES = ("LICENSE", "DATA_CARD.md", "CHANGELOG.md", "README.md", "hf_card_header.md")

# local path -> path in the HF repo
DOC_PATHS = {
    "DATA_CARD.md": "DATA_CARD.md", "LICENSE": "LICENSE",
    "PROVENANCE_AND_ETHICS.md": "PROVENANCE_AND_ETHICS.md", "CHANGELOG.md": "CHANGELOG.md",
    "README.md": "REPRODUCE.md", "requirements.txt": "requirements.txt",
}
# Large/binary data artifacts -- the actual dataset payload
DATA_PATHS = [
    "Cleaned_Master_CSV_v3.csv", "hetero_graphs/Graph_A.pt",
    "frozen_split_GraphA_seed42.pt", "material_task_split.json", "material_edges.pt",
    "manufacturer_features_no_material.pt", "manufacturer_image_release_asset.pt",
    "manufacturer_validity_mask.pt", "excluded_entries_v1.json",
    "manufacturer_coverage_flags.pt", "manufacturer_order_for_material_task.json",
]


def check_not_still_placeholder(force):
    issues = []
    for fname in CHECKED_FILES:
        text = open(fname, encoding="utf-8").read()
        for marker in PLACEHOLDER_MARKERS:
            if marker in text:
                issues.append(f"{fname} still contains '{marker}'")
    if issues and not force:
        print("REFUSING to publish -- unresolved placeholders:")
        for i in issues:
            print(f"  - {i}")
        print("Resolve these (or rerun with --force if you are certain) before publishing.")
        sys.exit(1)
    elif issues:
        print("--force passed: proceeding despite unresolved placeholders:")
        for i in issues:
            print(f"  - {i}")


def build_dataset_card():
    """HF renders the repo's README.md as the dataset card: YAML front matter and
    the v2 notice from hf_card_header.md, followed by DATA_CARD.md."""
    header = open("hf_card_header.md", encoding="utf-8").read()
    _, front, notice = header.split("---\n", 2)
    # Declare every CSV column as string: the viewer otherwise infers types from the
    # first rows (e.g. Business_Description is empty there) and fails on later rows.
    with open("Cleaned_Master_CSV_v3.csv", encoding="utf-8-sig", newline="") as f:
        columns = next(csv.reader(f))
    front += "dataset_info:\n- config_name: directory\n  features:\n"
    front += "".join(f"  - name: {json.dumps(c)}\n    dtype: string\n" for c in columns)
    card = open("DATA_CARD.md", encoding="utf-8").read()
    assert "\n## Summary" in card, "DATA_CARD.md layout changed; update build_dataset_card()"
    card = card.replace("\n## Summary", "\n" + notice.strip() + "\n\n## Summary", 1)
    return f"---\n{front}---\n\n{card}".encode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="e.g. shawntzx/PMGraph")
    ap.add_argument("--execute", action="store_true", help="actually upload; omit for a dry run")
    ap.add_argument("--force", action="store_true", help="publish even if placeholder markers remain (not recommended)")
    ap.add_argument("--private", action="store_true", help="create the repo as private if it does not exist yet")
    ap.add_argument("--tag", default="v2", help="tag for the new release commit")
    args = ap.parse_args()

    from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
    from huggingface_hub.utils import RepositoryNotFoundError
    api = HfApi(token=os.environ.get("HF_TOKEN"))  # falls back to `hf auth login`

    missing = [p for p in list(DOC_PATHS) + DATA_PATHS + ["hf_card_header.md"] if not os.path.exists(p)]
    if missing:
        print("ERROR -- expected files not found in the current directory:", missing)
        sys.exit(1)
    card = build_dataset_card()

    try:
        remote_files = api.list_repo_files(args.repo_id, repo_type="dataset")
        refs = api.list_repo_refs(args.repo_id, repo_type="dataset")
        tags = {t.name for t in refs.tags}
    except RepositoryNotFoundError:
        remote_files, tags = [], set()

    new_paths = {"README.md"} | set(DOC_PATHS.values()) | set(DATA_PATHS)
    to_delete = sorted(f for f in remote_files if f not in new_paths and f != ".gitattributes")

    print(f"Target repo: {args.repo_id}  ({'exists' if remote_files else 'will be created'})")
    print(f"Existing tags: {sorted(tags) or 'none'}")
    print("Will upload:")
    print("  README.md  (dataset card, built from hf_card_header.md + DATA_CARD.md)")
    for local, remote in DOC_PATHS.items():
        print(f"  {remote}" + (f"  (from {local})" if local != remote else ""))
    for p in DATA_PATHS:
        print(f"  {p}  ({os.path.getsize(p) / 1e6:.1f} MB)")
    print(f"Will delete from main ({len(to_delete)}): {to_delete}")
    print(f"Will tag the new commit as '{args.tag}'.")

    if args.tag in tags:
        print(f"ERROR -- tag '{args.tag}' already exists on {args.repo_id}; pass a different --tag.")
        sys.exit(1)
    check_not_still_placeholder(args.force)
    if not args.execute:
        print("\nDry run only -- nothing uploaded. Pass --execute to actually publish.")
        return

    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)

    ops = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card)]
    ops += [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=local) for local, remote in DOC_PATHS.items()]
    ops += [CommitOperationAdd(path_in_repo=p, path_or_fileobj=p) for p in DATA_PATHS]
    ops += [CommitOperationDelete(path_in_repo=p) for p in to_delete]
    print(f"uploading {len(ops)} operations in one commit ...")
    commit = api.create_commit(args.repo_id, operations=ops, repo_type="dataset",
                               commit_message="PMGraph v2 release")
    api.create_tag(args.repo_id, tag=args.tag, revision=commit.oid, repo_type="dataset",
                   tag_message="PMGraph v2")
    print(f"\nDone. Published to https://huggingface.co/datasets/{args.repo_id} (tag {args.tag})")


if __name__ == "__main__":
    main()
