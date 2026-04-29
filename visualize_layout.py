"""
Layout detection visualizer for docling.

Usage:
    python visualize_layout.py <path_to_pdf> [--output-dir <dir>] [--scale <float>]

Outputs:
  - A PNG per page with colored bounding boxes for every detected layout element
  - A summary table printed to stdout listing all detected labels and counts
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


LABEL_COLORS = {
    "text":                  (70,  130, 180),   # steel blue
    "section_header":        (255, 140,   0),   # dark orange
    "title":                 (220,  20,  60),   # crimson
    "list_item":             (60,  179, 113),   # medium sea green
    "caption":               (147, 112, 219),   # medium purple
    "footnote":              (188, 143, 143),   # rosy brown
    "page_header":           (128, 128,   0),   # olive
    "page_footer":           (128, 128,   0),   # olive
    "table":                 (255,  69,   0),   # orange red
    "document_index":        (255,  99,  71),   # tomato
    "picture":               (30,  144, 255),   # dodger blue
    "formula":               (255,  20, 147),   # deep pink
    "code":                  (0,   206, 209),   # dark turquoise
    "checkbox_selected":     (50,  205,  50),   # lime green
    "checkbox_unselected":   (169, 169, 169),   # dark gray
    "form":                  (255, 215,   0),   # gold
    "key_value_region":      (255, 165,   0),   # orange
    "paragraph":             (100, 149, 237),   # cornflower blue
}

FALLBACK_COLOR = (128, 128, 128)  # gray for any unknown label


def get_color(label_name: str):
    return LABEL_COLORS.get(label_name.lower(), FALLBACK_COLOR)


def draw_layout_on_image(page_image, clusters, page_w: float, page_h: float):
    """Draw colored bboxes + labels onto a copy of page_image."""
    from PIL import Image, ImageDraw, ImageFont

    img = page_image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    img_w, img_h = img.size
    scale_x = img_w / page_w
    scale_y = img_h / page_h

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", 13)
        except OSError:
            font = ImageFont.load_default()

    for cluster in clusters:
        label_name = cluster.label.value if hasattr(cluster.label, "value") else str(cluster.label)
        color = get_color(label_name)
        conf = cluster.confidence

        x0 = cluster.bbox.l * scale_x
        y0 = cluster.bbox.t * scale_y
        x1 = cluster.bbox.r * scale_x
        y1 = cluster.bbox.b * scale_y

        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

        # Filled semi-transparent rectangle
        draw.rectangle([(x0, y0), (x1, y1)], fill=(*color, 45), outline=(*color, 230), width=2)

        # Label text with background
        label_text = f"{label_name} {conf:.2f}"
        text_bbox = draw.textbbox((x0, y0 - 16), label_text, font=font)
        # Clamp text above box, fallback to inside if too close to top
        text_y = y0 - 16 if y0 > 18 else y0 + 2
        text_bbox = draw.textbbox((x0, text_y), label_text, font=font)
        draw.rectangle(
            [(text_bbox[0] - 2, text_bbox[1] - 1), (text_bbox[2] + 2, text_bbox[3] + 1)],
            fill=(*color, 210),
        )
        draw.text((x0, text_y), label_text, fill=(255, 255, 255, 255), font=font)

    composed = Image.alpha_composite(img, overlay)
    return composed.convert("RGB")


def build_legend_image(label_counter: Counter):
    """Build a small legend strip listing all detected labels and their counts."""
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = font_small = ImageFont.load_default()

    row_h = 26
    swatch = 18
    pad = 10
    width = 320
    height = pad + row_h * len(label_counter) + pad

    img = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    y = pad
    for label_name, count in sorted(label_counter.items()):
        color = get_color(label_name)
        draw.rectangle([(pad, y), (pad + swatch, y + swatch - 2)], fill=color, outline=(80, 80, 80))
        draw.text((pad + swatch + 8, y), f"{label_name}  ×{count}", font=font_small, fill=(30, 30, 30))
        y += row_h

    return img


def visualize_pdf(pdf_path: Path, output_dir: Path, scale: float = 2.0):
    print(f"Converting: {pdf_path}")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False                  # skip OCR, layout only
    pipeline_options.do_table_structure = False       # skip table structure
    pipeline_options.do_formula_enrichment = False    # skip VLM formula step
    pipeline_options.generate_page_images = True      # need page images
    pipeline_options.images_scale = scale

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(pdf_path)

    if not result.pages:
        print("ERROR: No pages found in conversion result.", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    global_counter: Counter = Counter()

    for page in result.pages:
        page_no = page.page_no
        assert page.size is not None, f"Page {page_no} has no size"

        page_image = page.get_image(scale=scale)
        if page_image is None:
            print(f"  [page {page_no}] WARNING: could not get image, skipping.")
            continue

        clusters = []
        if page.predictions.layout is not None:
            clusters = page.predictions.layout.clusters

        if not clusters:
            print(f"  [page {page_no}] No layout clusters detected.")

        page_counter: Counter = Counter()
        for c in clusters:
            label_name = c.label.value if hasattr(c.label, "value") else str(c.label)
            page_counter[label_name] += 1
            global_counter[label_name] += 1

        # Draw
        annotated = draw_layout_on_image(
            page_image, clusters, page.size.width, page.size.height
        )

        # Attach legend beside the annotated page
        legend = build_legend_image(page_counter)
        legend_h = legend.height
        page_h = annotated.height
        combined_h = max(page_h, legend_h)
        combined = __import__("PIL").Image.new("RGB", (annotated.width + legend.width + 10, combined_h), (255, 255, 255))
        combined.paste(annotated, (0, 0))
        combined.paste(legend, (annotated.width + 10, 0))

        out_path = output_dir / f"layout_page_{page_no:03d}.png"
        combined.save(out_path)
        print(f"  [page {page_no}] {sum(page_counter.values())} detections → {out_path}")
        for label, cnt in sorted(page_counter.items()):
            print(f"           {label:<30} {cnt:>3}x")

    print("\n=== Global summary ===")
    for label, cnt in sorted(global_counter.items()):
        print(f"  {label:<30} {cnt:>3}x")
    print(f"\nOutputs saved to: {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Visualize docling layout detections on a PDF.")
    parser.add_argument("pdf", type=Path, help="Path to input PDF file")
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=None,
        help="Directory to save output PNGs (default: <pdf_stem>_layout_vis/ next to the PDF)"
    )
    parser.add_argument(
        "--scale", "-s", type=float, default=2.0,
        help="Image scale factor for rendering (default: 2.0 = ~144 DPI)"
    )
    args = parser.parse_args()

    pdf_path = args.pdf.resolve()
    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or (pdf_path.parent / f"{pdf_path.stem}_layout_vis")
    visualize_pdf(pdf_path, output_dir, scale=args.scale)


if __name__ == "__main__":
    main()
#python visualize_layout.py two_sides.pdf --output-dir /storage/sourava/RAG_Pipeline/Demo/output1/two_sides/bbox_visualization
