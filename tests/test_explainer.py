"""Task 9: the "What is this?" explainer window's pure content -- NO Tk here
(demo/explainer.py's ExplainerWindow class does the Tk/Toplevel building;
this file only tests explainer_sections()/clipboard_text(), importable and
testable with no Tk root at all, same "no Tk in pure logic" convention as
tests/test_lattice_app_logic.py / tests/test_frontier.py).

Step 1 (brief): explainer_sections() must return a non-empty list of
(title, body) pairs with EVERY body non-empty -- "a placeholder section is
worse than an absent one." Step 5: clipboard_text() must contain every
section title, so Copy-to-clipboard can never silently omit one.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO = REPO_ROOT / "demo"
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

import explainer as ex  # noqa: E402


def test_explainer_sections_is_non_empty():
    sections = ex.explainer_sections()
    assert len(sections) > 0


def test_explainer_sections_are_title_body_pairs():
    sections = ex.explainer_sections()
    for s in sections:
        assert isinstance(s, tuple) and len(s) == 2
        title, body = s
        assert isinstance(title, str) and isinstance(body, str)


def test_every_explainer_section_title_is_non_empty():
    for title, _ in ex.explainer_sections():
        assert title.strip(), f"empty title in {ex.explainer_sections()!r}"


def test_every_explainer_section_body_is_non_empty():
    """The brief's own framing: 'a placeholder section is worse than an
    absent one' -- a body that is only whitespace is exactly that."""
    for title, body in ex.explainer_sections():
        assert body.strip(), f"section {title!r} has an empty/placeholder body"


def test_explainer_covers_every_topic_the_brief_names():
    """what a p-bit is; what sampling from an energy landscape means and
    how it differs from computing an answer; why temperature matters; what
    mediator spins are and why they exist; what the hardware gates check;
    and why some readouts honestly say `unavailable` -- six required
    topics, checked by keyword against the combined section text (title
    lower-cased + body lower-cased) so wording can vary but the TOPIC must
    be present somewhere."""
    sections = ex.explainer_sections()
    blob = "\n".join(f"{t}\n{b}" for t, b in sections).lower()
    required_keywords = [
        ("p-bit", ["p-bit"]),
        ("sampling vs computing", ["sampl", "energy landscape"]),
        ("temperature", ["temperature"]),
        ("mediator spins", ["mediator"]),
        ("hardware gates", ["gate"]),
        ("unavailable readouts", ["unavailable"]),
    ]
    for topic, keywords in required_keywords:
        assert any(k in blob for k in keywords), f"no section covers {topic!r}"


def test_unavailable_section_explains_it_is_not_a_bug():
    """The brief calls this topic out as the one most likely to be
    misread: 'it is the thing a visitor is most likely to misread as a bug
    rather than as the tool declining to guess.' The section covering it
    must say so, not just define the word unavailable."""
    sections = ex.explainer_sections()
    unavailable_bodies = [b for t, b in sections if "unavailable" in b.lower()]
    assert unavailable_bodies, "no section body even mentions 'unavailable'"
    combined = " ".join(unavailable_bodies).lower()
    assert "bug" in combined or "guess" in combined or "fabricat" in combined


def test_clipboard_text_contains_every_section_title():
    """Step 5 -- Copy-to-clipboard can never silently omit a section."""
    sections = ex.explainer_sections()
    text = ex.clipboard_text()
    for title, _ in sections:
        assert title in text, f"clipboard_text is missing section title {title!r}"


def test_clipboard_text_contains_every_section_body():
    sections = ex.explainer_sections()
    text = ex.clipboard_text()
    for _, body in sections:
        assert body in text, "clipboard_text is missing a section body"


def test_clipboard_text_is_a_single_nonempty_string():
    text = ex.clipboard_text()
    assert isinstance(text, str)
    assert text.strip()


def test_explainer_diagram_functions_return_images_of_consistent_size():
    """Static PNGs (numpy/PIL), not live renders -- per the tokens spec's
    own buildability note ('explainer diagrams (static, not live)'). Every
    diagram function listed alongside a section must return a real
    PIL.Image the same nominal size (so the window layout can lay them out
    predictably), never None / a placeholder."""
    from PIL import Image
    diagrams = ex.explainer_diagrams()
    assert len(diagrams) > 0
    sizes = set()
    for img in diagrams:
        assert isinstance(img, Image.Image)
        sizes.add(img.size)
    assert len(sizes) == 1, f"diagrams are not a consistent size: {sizes}"


def test_explainer_diagrams_count_matches_sections_count():
    assert len(ex.explainer_diagrams()) == len(ex.explainer_sections())
