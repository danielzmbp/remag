#!/usr/bin/env python3
"""Publish a REMAG release as a new version of its Zenodo record."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from packaging.version import InvalidVersion, Version

ZENODO_URL = "https://zenodo.org/api"
SANDBOX_URL = "https://sandbox.zenodo.org/api"
DEFAULT_RECORD_ID = "16762341"
REQUEST_TIMEOUT = 60
UPLOAD_TIMEOUT = 600


class ZenodoError(RuntimeError):
    """Raised when Zenodo cannot complete a release operation."""


class ZenodoClient:
    """Small client for the Zenodo records and deposit APIs."""

    def __init__(self, token, use_sandbox=False, session=None):
        self.base_url = SANDBOX_URL if use_sandbox else ZENODO_URL
        self.session = session or requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}"})

    def _request(self, method, url, expected_statuses=(200,), **kwargs):
        timeout = kwargs.pop("timeout", REQUEST_TIMEOUT)
        response = self.session.request(method, url, timeout=timeout, **kwargs)
        if response.status_code not in expected_statuses:
            detail = response.text.strip()[:500]
            raise ZenodoError(
                f"Zenodo {method} {url} returned {response.status_code}: {detail}"
            )
        return response

    def latest_record(self, seed_record_id):
        """Resolve any record in the version chain to the latest public record."""
        seed = self._request("GET", f"{self.base_url}/records/{seed_record_id}").json()
        latest_url = seed.get("links", {}).get("latest")
        if not latest_url:
            raise ZenodoError(
                f"Zenodo record {seed_record_id} has no latest-version link"
            )
        return self._request("GET", latest_url).json()

    def new_version_draft(self, latest_record):
        """Create or retrieve the draft following the latest published record."""
        record_id = latest_record["id"]
        deposition_url = f"{self.base_url}/deposit/depositions/{record_id}"
        deposition = self._request("GET", deposition_url).json()
        if deposition.get("state") != "done":
            raise ZenodoError(
                f"Latest Zenodo deposition {record_id} is not published "
                f"(state={deposition.get('state')!r})"
            )

        response = self._request(
            "POST",
            f"{deposition_url}/actions/newversion",
            expected_statuses=(200, 201),
        ).json()
        draft_url = response.get("links", {}).get("latest_draft")
        if not draft_url:
            raise ZenodoError("Zenodo did not return a latest-draft link")
        return self._request("GET", draft_url).json()

    def update_metadata(self, deposition_id, metadata):
        return self._request(
            "PUT",
            f"{self.base_url}/deposit/depositions/{deposition_id}",
            json={"metadata": metadata},
        ).json()

    def replace_files(self, deposition, release_files):
        deposition_id = deposition["id"]
        for file_info in deposition.get("files", []):
            filename = file_info.get("filename", file_info.get("key", "unknown"))
            print(f"Removing inherited file: {filename}")
            self._request(
                "DELETE",
                f"{self.base_url}/deposit/depositions/{deposition_id}/files/"
                f"{file_info['id']}",
                expected_statuses=(204, 404),
            )

        bucket_url = deposition.get("links", {}).get("bucket")
        if not bucket_url:
            raise ZenodoError(f"Zenodo deposition {deposition_id} has no file bucket")

        for file_path in release_files:
            print(f"Uploading {file_path.name}")
            with file_path.open("rb") as release_file:
                self._request(
                    "PUT",
                    f"{bucket_url}/{file_path.name}",
                    expected_statuses=(200, 201),
                    data=release_file,
                    timeout=UPLOAD_TIMEOUT,
                )

    def publish(self, deposition_id):
        return self._request(
            "POST",
            f"{self.base_url}/deposit/depositions/{deposition_id}/actions/publish",
            expected_statuses=(200, 202),
        ).json()


def normalize_version(tag):
    """Return a package version from a GitHub tag or ref."""
    version = tag.removeprefix("refs/tags/").removeprefix("v").strip()
    if not version:
        raise ZenodoError("No release version was provided")
    try:
        Version(version)
    except InvalidVersion as error:
        raise ZenodoError(f"Invalid release version {version!r}") from error
    return version


def distribution_files(dist_path=Path("dist")):
    """Validate and return the wheel and source distribution for the release."""
    files = sorted(path for path in dist_path.iterdir() if path.is_file())
    wheels = [path for path in files if path.suffix == ".whl"]
    source_distributions = [path for path in files if path.name.endswith(".tar.gz")]
    if len(files) != 2 or len(wheels) != 1 or len(source_distributions) != 1:
        raise ZenodoError(
            "Expected dist/ to contain exactly one wheel and one .tar.gz source "
            f"distribution; found {[path.name for path in files]}"
        )
    return files


def build_metadata(version, metadata_path=Path(".zenodo.json")):
    """Load shared metadata and add fields specific to this release."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["title"] = f"{metadata['title']} v{version}"
    metadata["version"] = version
    metadata["publication_date"] = datetime.now(timezone.utc).date().isoformat()
    return metadata


def published_version(record):
    version = record.get("metadata", {}).get("version")
    return normalize_version(str(version)) if version else None


def ensure_release_is_newer(version, latest_record):
    """Return False for a safe rerun and reject an older release."""
    latest_version = published_version(latest_record)
    if latest_version == version:
        return False
    if latest_version and Version(version) <= Version(latest_version):
        raise ZenodoError(
            f"Release {version} is not newer than Zenodo's latest version "
            f"{latest_version}"
        )
    return True


def main():
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        raise ZenodoError("ZENODO_TOKEN is not set")

    tag = os.environ.get("GITHUB_REF_NAME") or os.environ.get("GITHUB_REF", "")
    version = normalize_version(tag)
    use_sandbox = os.environ.get("ZENODO_SANDBOX", "false").lower() == "true"
    record_id = os.environ.get("ZENODO_RECORD_ID", DEFAULT_RECORD_ID)
    if use_sandbox and "ZENODO_RECORD_ID" not in os.environ:
        raise ZenodoError("ZENODO_RECORD_ID must be set when using the sandbox")

    files = distribution_files()
    metadata = build_metadata(version)
    client = ZenodoClient(token, use_sandbox=use_sandbox)

    latest_record = client.latest_record(record_id)
    if not ensure_release_is_newer(version, latest_record):
        print(f"Zenodo already contains REMAG {version}; nothing to do")
        return 0

    latest_version = published_version(latest_record) or "unknown"
    print(
        f"Creating REMAG {version} after Zenodo record {latest_record['id']} "
        f"(version {latest_version})"
    )
    deposition = client.new_version_draft(latest_record)
    client.update_metadata(deposition["id"], metadata)
    client.replace_files(deposition, files)
    published = client.publish(deposition["id"])

    doi = published.get("doi", "unknown")
    doi_url = published.get("doi_url") or published.get("links", {}).get("doi", "")
    print(f"Published REMAG {version} to Zenodo")
    print(f"DOI: {doi}")
    print(f"URL: {doi_url}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, requests.RequestException, ZenodoError) as error:
        print(f"Zenodo upload failed: {error}", file=sys.stderr)
        raise SystemExit(1)
