"""
app.py — Full Gradio layout for ContextFlow.
Two-page design: Landing (PDF upload) ↔ Main app (chat + viewer + proof).
"""

from __future__ import annotations

import gradio as gr

from app.config import cfg
from app.logger import get_logger
from app.ui.theme import make_theme, CUSTOM_CSS
from app.ui.components import pdf_library_html, db_stats_html
from app.ui import handlers

log = get_logger(__name__)

N_CHUNK_SLOTS = 5  # max retrieved chunks shown


def build_app() -> gr.Blocks:
    theme = make_theme()

    with gr.Blocks(title="ContextFlow") as demo:

        # ── Shared state ──────────────────────────────────────────────────────
        session_id_state   = gr.State("")
        hits_state         = gr.State([])
        library_open_state = gr.State(False)

        # ══════════════════════════════════════════════════════════════════════
        # LANDING PAGE
        # ══════════════════════════════════════════════════════════════════════
        with gr.Column(visible=True, elem_id="landing-page") as landing_page:
            gr.HTML("""
            <div class="landing-card">
              <div class="landing-title">ContextFlow</div>
              <div class="landing-tagline">Multimodal RAG — Upload · Index · Retrieve · Understand</div>
            </div>
            """)

            with gr.Column(elem_classes=["landing-card"]):
                gr.HTML('<div class="section-label">Upload PDF Documents</div>')
                pdf_upload_landing = gr.File(
                    label="",
                    file_count="multiple",
                    file_types=[".pdf"],
                    height=120,
                )
                force_reingest_cb = gr.Checkbox(
                    label="Force re-ingest (ignore cache)",
                    value=False,
                )
                process_btn = gr.Button(
                    "🚀  Process & Launch",
                    variant="primary",
                    size="lg",
                )
                landing_status = gr.Markdown(
                    value="",
                    elem_classes=["status-box"],
                )

        # ══════════════════════════════════════════════════════════════════════
        # MAIN APP PAGE
        # ══════════════════════════════════════════════════════════════════════
        with gr.Column(visible=False, elem_id="main-page") as main_page:

            # ── Top bar ───────────────────────────────────────────────────────
            with gr.Row():
                with gr.Column(scale=1):
                    back_btn = gr.Button("← Back", variant="secondary", size="sm")
                with gr.Column(scale=8):
                    gr.HTML("""
                    <div class="topbar">
                      <div>
                        <p class="topbar-title">ContextFlow</p>
                        <p class="topbar-subtitle">Multimodal Retrieval-Augmented Generation</p>
                      </div>
                    </div>
                    """)
                with gr.Column(scale=1):
                    library_toggle_btn = gr.Button("📚 Library", variant="secondary", size="sm")

            # ── Library drawer ────────────────────────────────────────────────
            with gr.Row(visible=False) as library_drawer:
                with gr.Column(scale=5):
                    gr.HTML('<div class="section-label">Indexed Documents</div>')
                    library_html = gr.HTML(pdf_library_html([]))

                with gr.Column(scale=3):
                    gr.HTML('<div class="section-label">Add More PDFs</div>')
                    add_pdf_upload = gr.File(
                        label="", file_count="multiple", file_types=[".pdf"], height=80
                    )
                    add_force_cb = gr.Checkbox(label="Force re-ingest", value=False)
                    add_btn    = gr.Button("⚡ Index & Add", variant="primary", size="sm")
                    add_status = gr.Markdown("", elem_classes=["status-box"])

                with gr.Column(scale=2):
                    db_stats_html_comp = gr.HTML(db_stats_html(0, 0))
                    gr.HTML('<div class="section-label" style="margin-top:12px;">Manage</div>')
                    doc_selector  = gr.Dropdown(
                        choices=[" "],
                        value=" ",
                        label="Select document",
                        interactive=True,
                    )
                    delete_btn    = gr.Button("🗑 Delete Document", variant="secondary", size="sm")
                    manage_status = gr.Markdown("")

            # ── Main body ──────────────────────────────────────────────────────
            with gr.Row():

                # ── LEFT: Chat + Chunks ───────────────────────────────────────
                with gr.Column(scale=6):
                    gr.HTML('<div class="section-label">💬 Chat</div>')
                    chatbot = gr.Chatbot(
                        label="",
                        height=480,
                        show_label=False,
                        layout="bubble",
                        allow_file_downloads=False,
                    )

                    # Query meta (sub-queries shown above input)
                    query_meta_disp = gr.HTML("")

                    with gr.Row():
                        filter_doc_dd = gr.Dropdown(
                            choices=[],
                            value=[],
                            label="Filter to document(s)  [leave empty = all]",
                            multiselect=True,
                            scale=3,
                            interactive=True,
                        )
                        top_k_slider = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="Top-K results",
                            scale=2,
                        )

                    with gr.Row():
                        query_input = gr.Textbox(
                            placeholder="Ask a question about your documents …",
                            label="",
                            lines=2,
                            max_lines=5,
                            scale=8,
                            show_label=False,
                        )
                        send_btn = gr.Button("Send ➤", variant="primary", scale=1)

                    with gr.Row():
                        clear_btn  = gr.Button("🗑 Clear Chat", variant="secondary", size="sm")
                        status_bar = gr.Markdown("", elem_classes=["progress-label"])

                    # ── Retrieved chunks ──────────────────────────────────────
                    gr.HTML('<div class="section-label" style="margin-top:12px;">🔗 Retrieved Chunks</div>')
                    chunk_slots = []  # (row, html_comp, view_btn)
                    for i in range(N_CHUNK_SLOTS):
                        with gr.Row(visible=False) as crow:
                            with gr.Column(scale=9):
                                chunk_html = gr.HTML("")
                            with gr.Column(scale=1, min_width=130):
                                view_btn = gr.Button("📄 View in PDF", variant="secondary", size="sm")
                        chunk_slots.append((crow, chunk_html, view_btn))

                # ── RIGHT: PDF Viewer + Proof Window ─────────────────────────
                with gr.Column(scale=4, elem_classes=["viewer-panel"]):
                    gr.HTML('<div class="section-label">📄 PDF Viewer + Proof Window</div>')
                    proof_header = gr.HTML("")

                    # Gallery shows annotated pages (all retrieved pages)
                    proof_gallery = gr.Gallery(
                        label="",
                        show_label=False,
                        columns=1,
                        height=1500,
                        object_fit="contain",
                        preview=True,
                        allow_preview=True,
                    )

        # ══════════════════════════════════════════════════════════════════════
        # EVENT WIRING
        # ══════════════════════════════════════════════════════════════════════

        # Landing → ingest → main app
        # Generator streams progress; .then() fires page switch after generator exhausts
        process_btn.click(
            fn=handlers.handle_upload_and_ingest,
            inputs=[pdf_upload_landing, force_reingest_cb],
            outputs=[
                landing_status,
                library_html,
                db_stats_html_comp,
                doc_selector,
                filter_doc_dd,
            ],
        ).then(
            fn=handlers.handle_switch_to_main,
            outputs=[main_page, landing_page],
        )

        # Add more PDFs from library drawer (stays on main page — no page swap needed)
        add_btn.click(
            fn=handlers.handle_add_pdfs,
            inputs=[add_pdf_upload, add_force_cb],
            outputs=[add_status, library_html, db_stats_html_comp, doc_selector, filter_doc_dd],
        )

        # Back button
        back_btn.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
            outputs=[landing_page, main_page],
        )

        # Library toggle
        library_toggle_btn.click(
            fn=handlers.handle_toggle_library,
            inputs=[library_open_state],
            outputs=[library_drawer, library_open_state],
        )

        _chunk_htmls = [ch for _, ch, _ in chunk_slots]
        _crows       = [crow for crow, _, _ in chunk_slots]

        _query_outputs = [
            chatbot, *_chunk_htmls, query_meta_disp,
            proof_gallery, status_bar, session_id_state,
            *_crows,
        ]

        # Query (send button)
        send_btn.click(
            fn=handlers.handle_query,
            inputs=[query_input, top_k_slider, filter_doc_dd, session_id_state, chatbot],
            outputs=_query_outputs,
        )

        # Query (Enter key)
        query_input.submit(
            fn=handlers.handle_query,
            inputs=[query_input, top_k_slider, filter_doc_dd, session_id_state, chatbot],
            outputs=_query_outputs,
        )

        # Clear chat
        clear_btn.click(
            fn=handlers.handle_clear_chat,
            inputs=[session_id_state],
            outputs=[chatbot, *_chunk_htmls, query_meta_disp, proof_gallery, status_bar, *_crows],
        )

        # Per-chunk "View in PDF" buttons
        for i, (crow, chunk_html, view_btn) in enumerate(chunk_slots):
            view_btn.click(
                fn=handlers.handle_jump_to_chunk,
                inputs=[gr.State(i), session_id_state],
                outputs=[proof_gallery],
            )

        # Delete document
        delete_btn.click(
            fn=handlers.handle_delete_document,
            inputs=[doc_selector],
            outputs=[manage_status, library_html, db_stats_html_comp, doc_selector],
        )

        # Refresh library on load — one call, 4 outputs
        demo.load(
            fn=handlers.handle_refresh_library,
            outputs=[library_html, db_stats_html_comp, doc_selector, filter_doc_dd],
        )

    return demo
