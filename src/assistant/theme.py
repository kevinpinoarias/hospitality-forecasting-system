"""
An optional "warm" look for the chat page: deep navy, saffron, terracotta, soft
teal and cream, with a serif for headings and a clean sans for text. Pure
presentation - no behaviour lives here.

The page opens in Gradio's normal look. A small button switches the warm look
on and off (and remembers the choice in the visitor's browser). Every warm rule
is scoped under the `warm` class on <body>, so with the look off nothing here
changes the page.

app.py passes CSS and HEAD to Gradio's `launch()` and uses the HTML pieces and
the two JS snippets. The colours are the light variables of the THEME below,
re-emitted under `body.warm` (see `_warm_variables`), so they win over Gradio's
own dark variant too and the warm look is the same on a dark system.
"""

from __future__ import annotations

import re

import gradio as gr

NAVY = "#1c2647"
SAFFRON = "#f7b033"
TERRACOTTA = "#dd675e"
TEAL = "#85cbd0"
CREAM = "#fffcec"

REPO_URL = "https://github.com/kevinpinoarias/hospitality-forecasting-system"
STORAGE_KEY = "forecast-page-style"

# Fonts come from Google Fonts - the same place Gradio's own default theme loads
# its fonts from. Mentioned in the README's "Data and privacy" section.
HEAD = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700'
    '&family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;1,9..144,500&display=swap" rel="stylesheet">'
)

_SAFFRON_SCALE = gr.themes.Color(
    c50="#fff8e6", c100="#ffefc2", c200="#fde196", c300="#fbd06a", c400="#f9c04e", c500=SAFFRON,
    c600="#dd9a1f", c700="#b87d14", c800="#8c5f10", c900="#66440c", c950="#432c07",
)
_NAVY_SCALE = gr.themes.Color(
    c50="#f4f5f9", c100="#e6e9f1", c200="#cdd2e2", c300="#a9b1cb", c400="#7d88ae", c500="#5a668f",
    c600="#434e73", c700="#333c5c", c800="#262e4d", c900=NAVY, c950="#121936",
)

THEME = gr.themes.Base(
    primary_hue=_SAFFRON_SCALE,
    secondary_hue=_SAFFRON_SCALE,
    neutral_hue=_NAVY_SCALE,
    font=[gr.themes.GoogleFont("DM Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill=CREAM,
    body_text_color=NAVY,
    background_fill_primary="#ffffff",
    background_fill_secondary=CREAM,
    block_background_fill="#ffffff",
    block_border_color=NAVY,
    block_border_width="1.5px",
    block_shadow="0 6px 24px rgba(28, 38, 71, 0.08)",
    button_primary_background_fill=SAFFRON,
    button_primary_background_fill_hover="#f9c04e",
    button_primary_text_color=NAVY,
    button_primary_border_color=SAFFRON,
    button_secondary_background_fill="transparent",
    button_secondary_background_fill_hover=SAFFRON,
    button_secondary_text_color=NAVY,
    button_secondary_border_color=NAVY,
    input_background_fill="#ffffff",
    input_border_color=NAVY,
    input_border_color_focus=SAFFRON,
    input_shadow_focus=f"0 0 0 3px {SAFFRON}55",
    link_text_color=TERRACOTTA,
    link_text_color_hover=NAVY,
)

# Used only if Gradio's internal theme-CSS helper ever disappears: the variables that matter most.
_FALLBACK_VARIABLES = (
    f"--body-background-fill: {CREAM}; --body-text-color: {NAVY}; --background-fill-primary: #ffffff; "
    f"--background-fill-secondary: {CREAM}; --block-background-fill: #ffffff; --block-border-color: {NAVY}; "
    f"--input-background-fill: #ffffff; --input-border-color: {NAVY}; --button-primary-background-fill: {SAFFRON}; "
    f"--button-primary-text-color: {NAVY}; --link-text-color: {TERRACOTTA};"
)


def _warm_variables() -> str:
    """The theme's light CSS variables (everything in its :root block)."""
    try:
        match = re.search(r":root\s*\{(.*?)\n\}", THEME._get_theme_css(), re.S)
        if match and "--body-background-fill" in match.group(1):
            return match.group(1)
    except Exception:  # an internal helper; never let it take the page down
        pass
    return _FALLBACK_VARIABLES


# Written WITHOUT the body.warm prefix; _scope() adds it, so the look cannot leak into the normal page.
_WARM_RULES = """
.gradio-container { max-width: none !important; padding: 0 !important; background: var(--cream) !important; }
.gradio-container .main.fillable, .gradio-container .main.app { padding: 0 !important; max-width: none !important; }
.gradio-container .html-container { padding: 0 !important; }
footer { display: none !important; }
#main { max-width: 920px; margin: 0 auto; padding: 8px 20px 36px; gap: 18px; }

/* top strip and header bar */
.top-strip, .top-strip * { color: var(--navy) !important; }
.top-strip { background: var(--saffron); text-align: center; font-size: 13px; font-weight: 500; padding: 7px 14px; letter-spacing: .2px; }
.site-header { background: var(--navy); display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 14px 28px; }
.brand, .brand * { color: var(--cream) !important; }
.brand { display: flex; align-items: center; gap: 12px; }
.brand svg { width: 34px; height: 34px; flex: none; }
.brand-name { font-family: var(--serif); font-size: 26px; font-weight: 500; letter-spacing: -.4px; line-height: 1.1; }
.brand-name em { color: var(--saffron) !important; font-style: italic; }
.pill { background: var(--saffron); color: var(--navy) !important; font-weight: 700; text-decoration: none !important;
        padding: 11px 22px; border-radius: 14px; font-size: 15px; white-space: nowrap; transition: transform .15s, background .15s; }
.pill:hover { background: #f9c04e; transform: translateY(-1px); }

/* the plain title and description are the first two blocks of the chat; the header and intro above replace them */
#chat-card > div > div:nth-child(-n+2) { display: none !important; }

/* intro */
.intro { text-align: center; padding: 34px 8px 4px; }
.intro h1, .intro p { color: var(--navy) !important; }
.intro h3 { color: var(--terracotta) !important; }
.intro h1 { font-family: var(--serif); font-weight: 500; font-size: clamp(34px, 6vw, 56px); letter-spacing: -1.2px; line-height: 1.05; margin: 0; }
.intro h3 { font-family: var(--serif); font-style: italic; font-weight: 500; font-size: clamp(20px, 3.4vw, 28px); margin: 8px 0 14px; letter-spacing: -.3px; }
.intro p { max-width: 680px; margin: 0 auto; font-size: 16.5px; line-height: 1.6; }

/* chat card */
#chat-card { background: #fff; border: 1.5px solid var(--navy); border-radius: 24px; padding: 14px; box-shadow: 0 10px 34px rgba(28, 38, 71, .10); }
#chat-card .block, #chat-card .form, #chat-card .wrap { border-radius: 18px; }
#chat-card .chatbot, #chat-card .bubble-wrap { background: var(--cream) !important; border-radius: 18px; }
#chat-card .message { border-radius: 20px !important; font-size: 15.5px; line-height: 1.55; box-shadow: none; }
#chat-card .message.bot { background: var(--navy) !important; color: var(--cream) !important; border: none !important; }
#chat-card .message.bot *, #chat-card .message.bot p, #chat-card .message.bot li { color: var(--cream) !important; }
#chat-card .message.bot a { color: var(--saffron) !important; }
#chat-card .message.bot strong { color: var(--saffron) !important; }
#chat-card .message.bot code { background: rgba(255, 252, 236, .14) !important; color: var(--teal) !important; }
#chat-card .message.bot table, #chat-card .message.bot td, #chat-card .message.bot th { border-color: rgba(255, 252, 236, .35) !important; }
#chat-card .message.user { background: var(--saffron) !important; color: var(--navy) !important; border: none !important; font-weight: 500; }
#chat-card .message.user * { color: var(--navy) !important; }
#chat-card textarea { font-size: 16px; }
#chat-card button.primary, #chat-card .submit-button { background: var(--saffron) !important; color: var(--navy) !important; font-weight: 700; border-radius: 14px !important; }
#chat-card .examples button, #chat-card .example { border: 1.5px solid var(--navy) !important; border-radius: 999px !important;
    background: transparent !important; color: var(--navy) !important; font-weight: 500; padding: 8px 16px; transition: background .15s; }
#chat-card .examples button:hover, #chat-card .example:hover { background: var(--saffron) !important; }

/* the style switch itself, in the warm look */
#style-toggle { border: 1.5px solid var(--navy) !important; border-radius: 999px !important; background: transparent !important; color: var(--navy) !important; font-weight: 500; }
#style-toggle:hover { background: var(--saffron) !important; }

/* privacy band */
.privacy, .privacy .inner { color: var(--cream) !important; }
.privacy { background: var(--navy); padding: 22px 28px; font-size: 14px; line-height: 1.6; }
.privacy .inner { max-width: 920px; margin: 0 auto; }
.privacy strong { color: var(--saffron) !important; font-family: var(--serif); font-weight: 600; font-size: 16px; }
.privacy a { color: var(--saffron) !important; }

@media (max-width: 640px) {
  .site-header { flex-direction: column; align-items: flex-start; padding: 14px 18px; }
  #main { padding: 6px 12px 28px; }
  .privacy { padding: 18px; }
}
"""

_PREFIX = "body.warm "


def _scope(css: str) -> str:
    """Put `body.warm ` in front of every selector (inside @media too)."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out, i = [], 0
    while True:
        start = css.find("{", i)
        if start == -1:
            break
        head = css[i:start].strip()
        depth, j = 1, start + 1
        while depth and j < len(css):
            depth += {"{": 1, "}": -1}.get(css[j], 0)
            j += 1
        inner = css[start + 1:j - 1]
        if head.startswith("@media"):
            out.append(f"{head} {{ {_scope(inner)} }}")
        else:
            selectors = ", ".join(_PREFIX + s.strip() for s in head.split(","))
            out.append(f"{selectors} {{ {inner.strip()} }}")
        i = j
    return "\n".join(out)


CSS = (
    # always on: the classes the app uses to show or hide a block per look, and the switch's bar
    # (written with :not(.warm), not as a plain default plus an override: Gradio adds a second, more specific copy
    # of every rule, which would beat the override)
    "body:not(.warm) .warm-only { display: none !important; }\n"
    "#style-bar { max-width: 920px; margin: 0 auto; justify-content: flex-end; padding: 8px 20px 0; }\n"
    # the warm look: the theme's light colours, then our rules, all under body.warm
    f":root body.warm, :root body.warm.dark {{ color-scheme: light; --navy: {NAVY}; --saffron: {SAFFRON}; "
    f"--terracotta: {TERRACOTTA}; --teal: {TEAL}; --cream: {CREAM}; "
    f"--serif: 'Fraunces', Georgia, 'Times New Roman', serif; {_warm_variables()} }}\n"
    "body.warm { background: var(--cream); color: var(--navy); font-family: var(--font); }\n"
    + _scope(_WARM_RULES)
)

# The switch toggles the class and remembers the choice; the other snippet runs once when the page opens.
TOGGLE_JS = (
    "() => { const on = document.body.classList.toggle('warm'); "
    f"try {{ localStorage.setItem('{STORAGE_KEY}', on ? 'warm' : 'classic'); }} catch (e) {{}} }}"
)
RESTORE_JS = (
    f"() => {{ try {{ if (localStorage.getItem('{STORAGE_KEY}') === 'warm') document.body.classList.add('warm'); }} "
    "catch (e) {} }"
)

_STAR = (
    '<svg viewBox="0 0 40 40" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">'
    '<g fill="none" stroke="#f7b033" stroke-width="2.4" stroke-linecap="round">'
    '<circle cx="20" cy="20" r="6.5" fill="#f7b033"/>'
    '<path d="M20 4v6M20 30v6M4 20h6M30 20h6M8.7 8.7l4.2 4.2M27.1 27.1l4.2 4.2M31.3 8.7l-4.2 4.2M12.9 27.1l-4.2 4.2"/>'
    "</g></svg>"
)

HEADER_HTML = (
    '<div class="top-strip">Live demo &middot; every figure comes from a real forecasting model, never invented</div>'
    f'<header class="site-header"><div class="brand">{_STAR}'
    '<span class="brand-name">Forecasting <em>Assistant</em></span></div>'
    f'<a class="pill" href="{REPO_URL}" target="_blank" rel="noopener">How it works</a></header>'
)

INTRO_HTML = (
    '<section class="intro"><h1>Ask about the forecast</h1><h3>in plain language</h3>'
    "<p>Ask about the sales forecast &mdash; e.g. &ldquo;what's the forecast for next Saturday?&rdquo;, "
    "&ldquo;is anything special about this weekend?&rdquo;, &ldquo;what if it rains?&rdquo;, or &ldquo;how accurate "
    "was last week's forecast?&rdquo;. Answers are grounded in a real, live forecasting API &mdash; never invented.</p></section>"
)

PRIVACY_HTML = (
    '<div class="privacy"><div class="inner"><strong>Privacy.</strong> Please don\'t type personal data. Your messages '
    "are sent to Google's Gemini API to write the answers, and this app keeps them only in the running chat's memory. "
    "The hosting service (Google Cloud Run) keeps standard request logs, which can include IP addresses, and this page "
    f'loads its fonts from Google Fonts. See &ldquo;Data and privacy&rdquo; in the <a href="{REPO_URL}#data-and-privacy" '
    'target="_blank" rel="noopener">project README</a>.</div></div>'
)
