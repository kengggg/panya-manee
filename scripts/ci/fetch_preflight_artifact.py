#!/usr/bin/env python3
"""Download and materialize a benchmark preflight record artifact from GitHub Actions.

Usage:
  python scripts/ci/fetch_preflight_artifact.py \
    --repo owner/repo \
    --preflight-date 20260412 \
    --output preflight_records/preflight_20260412.json
"""
from __future__ import annotations

import argparse
import os
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "preflight_records"
GITHUB_API = "https://api.github.com"


class PreflightFetchError(RuntimeError):
    pass


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


def github_request(url: str, token: str, *, accept: str = "application/vnd.github+json") -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": accept,
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "panya-manee-preflight-fetcher",
        },
    )


def fetch_json(url: str, token: str) -> dict:
    req = github_request(url, token)
    with urllib.request.urlopen(req, timeout=60) as resp:
        import json

        return json.load(resp)


def fetch_bytes(url: str, token: str) -> bytes:
    req = github_request(url, token)
    opener = urllib.request.build_opener(NoRedirectHandler())
    try:
        with opener.open(req, timeout=120) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code not in (301, 302, 303, 307, 308):
            raise
        redirect_url = exc.headers.get("Location")
        if not redirect_url:
            raise
        redirected_req = urllib.request.Request(
            redirect_url,
            headers={"User-Agent": "panya-manee-preflight-fetcher"},
        )
        with urllib.request.urlopen(redirected_req, timeout=120) as resp:
            return resp.read()


def iter_artifacts(repo: str, token: str):
    page = 1
    while True:
        data = fetch_json(
            f"{GITHUB_API}/repos/{repo}/actions/artifacts?per_page=100&page={page}",
            token,
        )
        artifacts = data.get("artifacts", [])
        if not artifacts:
            break
        for artifact in artifacts:
            yield artifact
        if len(artifacts) < 100:
            break
        page += 1


def choose_latest_artifact(artifacts: list[dict], artifact_name: str) -> dict:
    matches = [
        artifact
        for artifact in artifacts
        if artifact.get("name") == artifact_name and not artifact.get("expired", False)
    ]
    if not matches:
        raise PreflightFetchError(f"artifact '{artifact_name}' not found")
    matches.sort(
        key=lambda artifact: (
            artifact.get("updated_at", ""),
            artifact.get("created_at", ""),
            int(artifact.get("id", 0)),
        ),
        reverse=True,
    )
    return matches[0]


def extract_zip_bytes(zip_bytes: bytes, dest_root: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(zip_bytes)
        tmp_path = Path(tmp.name)
    try:
        with zipfile.ZipFile(tmp_path) as zf:
            zf.extractall(dest_root)
    finally:
        tmp_path.unlink(missing_ok=True)


def materialize_preflight_record(preflight_date: str, extracted_root: Path, output_path: Path) -> Path:
    candidates = list(extracted_root.rglob(f"preflight_{preflight_date}.json"))
    if not candidates:
        raise PreflightFetchError(
            f"downloaded artifact does not contain preflight_{preflight_date}.json"
        )
    candidates.sort(key=lambda path: (len(path.parts), str(path)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(candidates[0].read_bytes())
    return output_path


def fetch_preflight_artifact(
    repo: str,
    preflight_date: str,
    output_path: Path | None = None,
    token: str | None = None,
) -> dict:
    output_path = (output_path or (DEFAULT_OUTPUT_DIR / f"preflight_{preflight_date}.json")).resolve()
    if output_path.exists():
        return {
            "repo": repo,
            "artifact_name": f"preflight-{preflight_date}",
            "artifact_id": None,
            "workflow_run_id": None,
            "output_path": str(output_path),
            "reused_local": True,
        }

    token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        raise PreflightFetchError("GITHUB_TOKEN or GH_TOKEN is required")

    artifact_name = f"preflight-{preflight_date}"
    artifact = choose_latest_artifact(list(iter_artifacts(repo, token)), artifact_name)
    artifact_id = int(artifact["id"])
    zip_bytes = fetch_bytes(
        f"{GITHUB_API}/repos/{repo}/actions/artifacts/{artifact_id}/zip",
        token,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        extracted_root = Path(tmpdir)
        extract_zip_bytes(zip_bytes, extracted_root)
        materialize_preflight_record(preflight_date, extracted_root, output_path)

    return {
        "repo": repo,
        "artifact_name": artifact_name,
        "artifact_id": artifact_id,
        "workflow_run_id": artifact.get("workflow_run", {}).get("id"),
        "output_path": str(output_path),
        "reused_local": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a benchmark preflight record artifact")
    parser.add_argument("--repo", required=True, help="GitHub repo in owner/name form")
    parser.add_argument("--preflight-date", required=True, help="Preflight date YYYYMMDD")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: preflight_records/preflight_YYYYMMDD.json)",
    )
    args = parser.parse_args()

    result = fetch_preflight_artifact(
        repo=args.repo,
        preflight_date=args.preflight_date,
        output_path=args.output,
    )

    if result["reused_local"]:
        print(f"Preflight record already present: {result['output_path']}")
    else:
        print(f"Fetched preflight artifact: {result['artifact_name']}")
        print(f"  Repo: {result['repo']}")
        print(f"  Artifact ID: {result['artifact_id']}")
        print(f"  Workflow run ID: {result['workflow_run_id']}")
        print(f"  Output: {result['output_path']}")


if __name__ == "__main__":
    main()
