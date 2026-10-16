"""
Tests for the chat page's optional warm look (src/assistant/theme.py). Pure
presentation, so these only guard what must not silently go wrong: the privacy
notice, that the warm look cannot leak into the normal page, and the hooks
app.py relies on.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("gradio")

from src.assistant import theme  # noqa: E402

# Selectors allowed OUTSIDE the body.warm scope: the switch's own bar, and the rule that hides warm-only blocks
# until the look is switched on.
UNSCOPED_ALLOWED = ("#style-bar", "body:not(.warm) .warm-only")
SCOPED_PREFIXES = ("body.warm", ":root body.warm")


def _selectors(css: str) -> list[str]:
    """Every selector in the CSS, including those inside @media blocks."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    found = []
    for head in re.findall(r"([^{}]+)\{", css):
        head = head.strip()
        if head.startswith("@"):
            continue
        found += [part.strip() for part in head.split(",")]
    return found


def test_the_privacy_notice_keeps_every_point_the_readme_promises():
    text = theme.PRIVACY_HTML
    for must_say in ("personal data", "Gemini", "Google Cloud Run", "IP addresses", "Google Fonts", "#data-and-privacy"):
        assert must_say in text


def test_no_warm_rule_can_leak_into_the_normal_page():
    selectors = _selectors(theme.CSS)
    assert selectors
    for selector in selectors:
        assert selector.startswith(SCOPED_PREFIXES + UNSCOPED_ALLOWED), f"unscoped rule: {selector}"


def test_scoping_prefixes_every_selector_and_keeps_media_queries():
    scoped = theme._scope("a, .b { color: red } @media (max-width: 10px) { .c { color: blue } }")
    assert "body.warm a, body.warm .b" in scoped
    assert "@media (max-width: 10px)" in scoped and "body.warm .c" in scoped


def test_the_warm_variables_come_from_the_theme_and_win_over_the_dark_variant():
    variables = theme._warm_variables()
    assert "--body-background-fill" in variables
    assert ":root body.warm.dark" in theme.CSS  # the same colours when Gradio has switched to its dark variant


def test_the_switch_remembers_the_choice_under_one_storage_key():
    assert theme.STORAGE_KEY in theme.TOGGLE_JS and "classList.toggle('warm')" in theme.TOGGLE_JS
    assert theme.STORAGE_KEY in theme.RESTORE_JS and "classList.add('warm')" in theme.RESTORE_JS


def test_the_palette_is_the_documented_one():
    assert (theme.NAVY, theme.SAFFRON, theme.TERRACOTTA, theme.TEAL, theme.CREAM) == (
        "#1c2647", "#f7b033", "#dd675e", "#85cbd0", "#fffcec",
    )
    for colour in (theme.NAVY, theme.SAFFRON, theme.CREAM):
        assert colour in theme.CSS


def test_the_page_pieces_app_py_relies_on_exist():
    assert "#main" in theme.CSS and "#chat-card" in theme.CSS and "#style-toggle" in theme.CSS
    assert 'class="site-header"' in theme.HEADER_HTML and 'class="intro"' in theme.INTRO_HTML
    assert "fonts.googleapis.com" in theme.HEAD  # the fonts the privacy notice mentions


def test_the_page_carries_no_third_party_name_or_branding():
    everything = (theme.HEADER_HTML + theme.INTRO_HTML + theme.PRIVACY_HTML + theme.CSS + theme.HEAD).lower()
    for word in ("andaluz", "squarespace", "volume tc", "halyard"):
        assert word not in everything
