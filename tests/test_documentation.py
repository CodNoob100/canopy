"""Keep user-facing documentation links and claim boundaries reviewable."""

from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "MOTIVATION.md",
    ROOT / "docs" / "cli.md",
    ROOT / "docs" / "reproducibility.md",
    ROOT / "docs" / "research-evidence.md",
)
LOCAL_LINK = re.compile(r"!?\[[^\]]+\]\(([^)]+)\)")


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda path: path.name)
def test_local_documentation_links_exist(document):
    """Guard against stale references to code paths, docs and figures."""
    for raw in LOCAL_LINK.findall(document.read_text(encoding="utf-8")):
        target = raw.split("#", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        resolved = (document.parent / target).resolve()
        assert resolved.is_relative_to(ROOT), f"Path leaves repository: {raw}"
        assert resolved.exists(), f"Broken link in {document}: {raw}"


def test_readme_and_motivation_do_not_claim_exact_architecture_counts():
    """An architectural upper bound must not be marketed as a realized count."""
    text = "\n".join(p.read_text(encoding="utf-8").lower() for p in DOCUMENTS[:2])
    assert "calculates the architecture's exact theoretical expressivity" not in text
    assert "first training-free nas metric" not in text
    assert "independently reproduced results" in DOCUMENTS[0].read_text(encoding="utf-8")

