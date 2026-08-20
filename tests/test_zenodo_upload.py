import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / ".github" / "scripts" / "zenodo_upload.py"
if not SCRIPT_PATH.exists():
    pytest.skip(
        "Zenodo workflow script is not included in the sdist", allow_module_level=True
    )

SPEC = importlib.util.spec_from_file_location("zenodo_upload", SCRIPT_PATH)
zenodo_upload = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(zenodo_upload)


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.headers = {}
        self.requests = []

    def request(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        return next(self.responses)


def test_latest_record_and_new_version_follow_the_existing_chain():
    session = FakeSession(
        [
            FakeResponse(200, {"links": {"latest": "https://latest-record"}}),
            FakeResponse(200, {"id": 20, "metadata": {"version": "0.4.3"}}),
            FakeResponse(200, {"id": 20, "state": "done"}),
            FakeResponse(201, {"links": {"latest_draft": "https://latest-draft"}}),
            FakeResponse(200, {"id": 21, "state": "inprogress"}),
        ]
    )
    client = zenodo_upload.ZenodoClient("token", session=session)

    latest = client.latest_record("10")
    draft = client.new_version_draft(latest)

    assert latest["id"] == 20
    assert draft["id"] == 21
    assert session.requests[0][1].endswith("/records/10")
    assert session.requests[2][1].endswith("/deposit/depositions/20")
    assert session.requests[3][1].endswith("/deposit/depositions/20/actions/newversion")


def test_same_version_is_an_idempotent_rerun():
    latest = {"metadata": {"version": "0.4.4"}}
    assert not zenodo_upload.ensure_release_is_newer("0.4.4", latest)


def test_older_version_is_rejected():
    latest = {"metadata": {"version": "0.4.4"}}
    with pytest.raises(zenodo_upload.ZenodoError, match="not newer"):
        zenodo_upload.ensure_release_is_newer("0.4.3", latest)


def test_distribution_files_requires_one_wheel_and_one_sdist(tmp_path):
    wheel = tmp_path / "remag-0.4.4-py3-none-any.whl"
    sdist = tmp_path / "remag-0.4.4.tar.gz"
    wheel.touch()
    sdist.touch()

    assert zenodo_upload.distribution_files(tmp_path) == [wheel, sdist]


def test_metadata_comes_from_zenodo_json():
    metadata = zenodo_upload.build_metadata("0.4.4")

    assert metadata["title"].endswith("v0.4.4")
    assert metadata["version"] == "0.4.4"
    assert "HyenaDNA" in metadata["description"]
    assert "HDBSCAN" not in metadata["description"]
