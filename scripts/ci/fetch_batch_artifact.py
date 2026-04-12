#!/usr/bin/env python3
"""Download and verify a staged batch artifact from a prior GitHub Actions run.

Usage:
  python scripts/ci/fetch_batch_artifact.py \
    --repo owner/repo \
    --run-id 123456789 \
    --artifact-name benchmark-verified-ntp3-vr1-20260412 \
    --batch-id ntp3-vr1-20260412
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEST_ROOT = PROJECT_ROOT
GITHUB_API = "https://api.github.com"


class ArtifactFetchError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def github_request(url: str, token: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "panya-manee-artifact-fetcher",
        },
    )


def fetch_json(url: str, token: str) -> dict:
    req = github_request(url, token)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def fetch_bytes(url: str, token: str) -> bytes:
    req = github_request(url, token)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def find_artifact_id(repo: str, run_id: str, artifact_name: str, token: str) -> int:
    data = fetch_json(f"{GITHUB_API}/repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100", token)
    matches = [
        artifact for artifact in data.get("artifacts", [])
        if artifact.get("name") == artifact_name and not artifact.get("expired", False)
    ]
    if not matches:
        raise ArtifactFetchError(
            f"artifact '{artifact_name}' not found for run {run_id} in {repo}"
        )
    if len(matches) > 1:
        raise ArtifactFetchError(
            f"artifact '{artifact_name}' is ambiguous for run {run_id} in {repo}"
        )
    return int(matches[0]["id"])


def extract_zip_bytes(zip_bytes: bytes, dest_root: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(zip_bytes)
        tmp_path = Path(tmp.name)
    try:
        with zipfile.ZipFile(tmp_path) as zf:
            zf.extractall(dest_root)
    finally:
        tmp_path.unlink(missing_ok=True)


def materialize_benchmark_responses(batch_id: str, extracted_root: Path, dest_root: Path) -> Path:
    matches = list(extracted_root.rglob(f"batch_manifest_{batch_id}.json"))
    if not matches:
        raise ArtifactFetchError(
            f"downloaded artifact does not contain batch_manifest_{batch_id}.json"
        )
    if len(matches) > 1:
        raise ArtifactFetchError(
            f"downloaded artifact contains multiple batch manifests for {batch_id}"
        )

    source_responses_dir = matches[0].parent
    dest_responses_dir = dest_root / "benchmark_responses"
    dest_responses_dir.mkdir(parents=True, exist_ok=True)

    for path in source_responses_dir.iterdir():
        if path.is_file():
            target = dest_responses_dir / path.name
            target.write_bytes(path.read_bytes())

    return dest_responses_dir


def clear_existing_batch_files(batch_id: str, dest_root: Path) -> None:
    responses_dir = dest_root / "benchmark_responses"
    if not responses_dir.exists():
        return
    patterns = [
        f"responses_{batch_id}-*.jsonl",
        f"repeat_summary_{batch_id}.json",
        f"verification_report_{batch_id}.json",
        f"artifact_manifest_{batch_id}.txt",
        f"batch_manifest_{batch_id}.json",
    ]
    for pattern in patterns:
        for path in responses_dir.glob(pattern):
            path.unlink(missing_ok=True)


def verify_manifest(batch_id: str, artifact_name: str, dest_root: Path) -> dict:
    manifest_path = dest_root / "benchmark_responses" / f"batch_manifest_{batch_id}.json"
    if not manifest_path.exists():
        raise ArtifactFetchError(f"downloaded artifact missing batch manifest: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("batch_id") != batch_id:
        raise ArtifactFetchError(
            f"batch manifest batch_id mismatch: expected {batch_id}, got {manifest.get('batch_id')}"
        )
    if manifest.get("artifact_name") != artifact_name:
        raise ArtifactFetchError(
            f"batch manifest artifact_name mismatch: expected {artifact_name}, got {manifest.get('artifact_name')}"
        )

    files = manifest.get("files", [])
    if not files:
        raise ArtifactFetchError("batch manifest contains no files")

    for entry in files:
        rel = entry.get("path")
        expected_sha = entry.get("sha256")
        if not rel or not expected_sha:
            raise ArtifactFetchError(f"invalid manifest entry: {entry}")
        path = dest_root / rel
        if not path.exists():
            raise ArtifactFetchError(f"manifest file missing after download: {rel}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ArtifactFetchError(
                f"sha256 mismatch for {rel}: expected {expected_sha}, got {actual_sha}"
            )

    return manifest


def fetch_batch_artifact(
    repo: str,
    run_id: str,
    artifact_name: str,
    batch_id: str,
    dest_root: Path | None = None,
    token: str | None = None,
) -> dict:
    dest_root = (dest_root or DEFAULT_DEST_ROOT).resolve()
    token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        raise ArtifactFetchError("GITHUB_TOKEN or GH_TOKEN is required")

    artifact_id = find_artifact_id(repo, run_id, artifact_name, token)
    zip_bytes = fetch_bytes(
        f"{GITHUB_API}/repos/{repo}/actions/artifacts/{artifact_id}/zip",
        token,
    )

    clear_existing_batch_files(batch_id, dest_root)
    with tempfile.TemporaryDirectory() as tmpdir:
        extracted_root = Path(tmpdir)
        extract_zip_bytes(zip_bytes, extracted_root)
        materialize_benchmark_responses(batch_id, extracted_root, dest_root)
    manifest = verify_manifest(batch_id, artifact_name, dest_root)

    return {
        "repo": repo,
        "run_id": run_id,
        "artifact_id": artifact_id,
        "artifact_name": artifact_name,
        "batch_id": batch_id,
        "file_count": manifest.get("file_count", len(manifest.get("files", []))),
        "manifest_path": str(dest_root / "benchmark_responses" / f"batch_manifest_{batch_id}.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and verify a benchmark batch artifact")
    parser.add_argument("--repo", required=True, help="GitHub repo in owner/name form")
    parser.add_argument("--run-id", required=True, help="Workflow run ID that produced the artifact")
    parser.add_argument("--artifact-name", required=True, help="Exact artifact name")
    parser.add_argument("--batch-id", required=True, help="Batch ID expected inside the artifact")
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=None,
        help="Extraction root (default: project root)",
    )
    args = parser.parse_args()

    result = fetch_batch_artifact(
        repo=args.repo,
        run_id=args.run_id,
        artifact_name=args.artifact_name,
        batch_id=args.batch_id,
        dest_root=args.dest_root,
    )

    print(f"Fetched batch artifact: {result['artifact_name']}")
    print(f"  Repo: {result['repo']}")
    print(f"  Run ID: {result['run_id']}")
    print(f"  Artifact ID: {result['artifact_id']}")
    print(f"  Batch: {result['batch_id']}")
    print(f"  Files: {result['file_count']}")
    print(f"  Manifest: {result['manifest_path']}")


if __name__ == "__main__":
    main()
