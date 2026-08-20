from importlib.metadata import version

import remag
from remag import cli


def test_runtime_version_matches_distribution_metadata():
    expected_version = version("remag")

    assert remag.__version__ == expected_version
    assert cli.__version__ == expected_version
