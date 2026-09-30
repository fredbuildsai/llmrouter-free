"""Documentation is checked like code: arc42 completeness, working links, and real test references."""
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
ARCH = ROOT / "docs" / "architecture"

ARC42_SECTIONS = [
    "01-introduction-and-goals", "02-constraints", "03-context-and-scope", "04-solution-strategy",
    "05-building-block-view", "06-runtime-view", "07-deployment-view", "08-crosscutting-concepts",
    "09-architecture-decisions", "10-quality-requirements", "11-risks-and-technical-debt", "12-glossary",
]


def test_all_twelve_arc42_sections_exist_and_are_not_stubs():
    for section in ARC42_SECTIONS:
        text = (ARCH / f"{section}.md").read_text()
        assert text.startswith("# "), section
        assert len(text.split()) > 60, f"{section} looks like a stub"


def test_architecture_index_links_every_section():
    index = (ARCH / "README.md").read_text()
    for section in ARC42_SECTIONS:
        assert f"({section}.md)" in index


def test_relative_markdown_links_resolve():
    for md in [ROOT / "README.md", *ARCH.glob("*.md")]:
        for target in re.findall(r"\]\(([^)#\s]+\.md)(?:#[^)]*)?\)", md.read_text()):
            if target.startswith("http"):
                continue
            assert (md.parent / target).exists(), f"{md.name} links to missing {target}"


def test_tests_cited_by_the_quality_scenarios_exist():
    quality = (ARCH / "10-quality-requirements.md").read_text()
    cited = set(re.findall(r"`(test_\w+\.py)`", quality))
    assert cited, "quality table should cite verifying tests"
    for name in cited:
        assert (ROOT / "tests" / name).exists(), f"quality doc cites missing {name}"


def test_readme_documents_installing_from_github():
    readme = (ROOT / "README.md").read_text()
    assert "git+https://github.com/fredbuildsai/llmrouter-free.git" in readme
