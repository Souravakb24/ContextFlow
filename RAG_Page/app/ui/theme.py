"""
theme.py — ContextFlow Gradio theme: white background, dark purple foreground.
"""

import gradio as gr

# ── Colour palette ────────────────────────────────────────────────────────────
PURPLE_DARK   = "#2D1B69"
PURPLE_MID    = "#4A2C8F"
PURPLE_LIGHT  = "#7B52C1"
PURPLE_PALE   = "#EDE7F6"
WHITE         = "#FFFFFF"
OFF_WHITE     = "#F8F7FC"
BORDER        = "#D5CBF0"
TEXT_DARK     = "#1A1A2E"
TEXT_MID      = "#4A4A6A"
TEXT_LIGHT    = "#7A7A9A"
SUCCESS       = "#27AE60"
WARNING       = "#E67E22"
ERROR         = "#E74C3C"


def make_theme() -> gr.Theme:
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#EDE7F6", c100="#D1C4E9", c200="#B39DDB", c300="#9575CD",
            c400="#7E57C2", c500="#4A2C8F", c600="#3B1F7A", c700="#2D1B69",
            c800="#1E1254", c900="#12093D", c950="#080520",
        ),
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace"],
    ).set(
        body_background_fill=WHITE,
        body_text_color=TEXT_DARK,
        block_background_fill=WHITE,
        block_border_color=BORDER,
        block_label_text_color=PURPLE_MID,
        block_title_text_color=PURPLE_DARK,
        input_background_fill=OFF_WHITE,
        input_border_color=BORDER,
        input_placeholder_color=TEXT_LIGHT,
        button_primary_background_fill=PURPLE_MID,
        button_primary_background_fill_hover=PURPLE_DARK,
        button_primary_text_color=WHITE,
        button_secondary_background_fill=WHITE,
        button_secondary_background_fill_hover=PURPLE_PALE,
        button_secondary_text_color=PURPLE_MID,
        button_secondary_border_color=PURPLE_LIGHT,
    )


# ── Custom CSS ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
/* ── Global ───────────────────────────────────────────────────── */
* { box-sizing: border-box; }

body, .gradio-container {
    background: #FFFFFF !important;
    font-family: 'Inter', sans-serif;
    color: #1A1A2E;
}

/* ── Top bar ──────────────────────────────────────────────────── */
.topbar {
    background: linear-gradient(135deg, #2D1B69 0%, #4A2C8F 60%, #7B52C1 100%);
    padding: 16px 28px;
    border-radius: 12px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 20px rgba(45, 27, 105, 0.25);
}

.topbar-title {
    color: #FFFFFF;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    margin: 0;
}

.topbar-subtitle {
    color: #D1C4E9;
    font-size: 0.85rem;
    margin: 2px 0 0 0;
}

/* ── Landing page ─────────────────────────────────────────────── */
.landing-card {
    background: linear-gradient(160deg, #F8F7FC 0%, #EDE7F6 100%);
    border: 1.5px solid #D5CBF0;
    border-radius: 16px;
    padding: 40px;
    text-align: center;
    max-width: 640px;
    margin: 40px auto;
    box-shadow: 0 8px 32px rgba(45, 27, 105, 0.10);
}

.landing-title {
    color: #2D1B69;
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 6px;
    letter-spacing: -0.02em;
}

.landing-tagline {
    color: #7B52C1;
    font-size: 1.05rem;
    margin-bottom: 28px;
}

/* ── Section labels ───────────────────────────────────────────── */
.section-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #7B52C1;
    margin-bottom: 6px;
}

/* ── Chunk cards ──────────────────────────────────────────────── */
.chunk-card {
    background: #FFFFFF;
    border: 1.5px solid #D5CBF0;
    border-left: 4px solid #4A2C8F;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
    transition: box-shadow 0.2s;
}

.chunk-card:hover {
    box-shadow: 0 4px 16px rgba(45, 27, 105, 0.12);
}

.chunk-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.chunk-doc-name {
    font-weight: 600;
    font-size: 0.85rem;
    color: #2D1B69;
}

.chunk-score-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    color: white;
}

.score-high   { background: #27AE60; }
.score-medium { background: #E67E22; }
.score-low    { background: #E74C3C; }

.chunk-text {
    font-size: 0.83rem;
    color: #4A4A6A;
    line-height: 1.5;
    max-height: 80px;
    overflow-y: auto;
    border-top: 1px solid #EDE7F6;
    padding-top: 6px;
    margin-top: 4px;
}

.chunk-meta {
    font-size: 0.75rem;
    color: #7A7A9A;
    margin-top: 4px;
}

/* ── PDF viewer panel ─────────────────────────────────────────── */
.viewer-panel {
    border-left: 2px solid #EDE7F6;
    padding-left: 16px;
}

.page-nav {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
}

/* ── Status box ───────────────────────────────────────────────── */
.status-box {
    background: #F8F7FC;
    border: 1.5px solid #D5CBF0;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.83rem;
    color: #4A4A6A;
    font-family: 'JetBrains Mono', monospace;
    min-height: 60px;
    max-height: 160px;
    overflow-y: auto;
}

/* ── Chat bubbles ─────────────────────────────────────────────── */
.user-bubble {
    background: linear-gradient(135deg, #4A2C8F, #2D1B69);
    color: white;
    border-radius: 16px 16px 4px 16px;
    padding: 10px 14px;
    margin: 4px 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 0.9rem;
}

.bot-bubble {
    background: #F8F7FC;
    border: 1.5px solid #D5CBF0;
    color: #1A1A2E;
    border-radius: 4px 16px 16px 16px;
    padding: 10px 14px;
    margin: 4px 0;
    max-width: 85%;
    font-size: 0.9rem;
}

/* ── Library drawer ───────────────────────────────────────────── */
.library-drawer {
    background: #F8F7FC;
    border: 1.5px solid #D5CBF0;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 12px;
}

.pdf-lib-card {
    background: white;
    border: 1.5px solid #D5CBF0;
    border-left: 4px solid #7B52C1;
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 6px;
    font-size: 0.83rem;
}

.pdf-lib-card.active {
    border-left-color: #2D1B69;
    background: #EDE7F6;
}

/* ── Progress spinner label ───────────────────────────────────── */
.progress-label {
    color: #7B52C1;
    font-size: 0.82rem;
    font-style: italic;
    margin-top: 4px;
}

/* ── Scrollbar styling ────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #F8F7FC; }
::-webkit-scrollbar-thumb { background: #D5CBF0; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #7B52C1; }

/* ── Button overrides ─────────────────────────────────────────── */
.gr-button-primary {
    background: linear-gradient(135deg, #4A2C8F, #2D1B69) !important;
    border: none !important;
    font-weight: 600 !important;
}

.gr-button-secondary {
    border-color: #D5CBF0 !important;
    color: #4A2C8F !important;
}

/* ── Annotated image caption ──────────────────────────────────── */
.proof-caption {
    font-size: 0.78rem;
    color: #7A7A9A;
    text-align: center;
    margin-top: 4px;
}
"""
