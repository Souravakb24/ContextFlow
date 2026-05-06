import json
import sys
import os
from pathlib import Path
from PIL import Image, ImageDraw

# PDF standard page size in points (used when layout JSON is unavailable)
PDF_PAGE_W = 612.0
PDF_PAGE_H = 792.0


def get_pdf_page_dims(layout_json_path: str) -> tuple[float, float]:
    """Return (pdf_w, pdf_h) in points from the layout JSON, falling back to standard letter."""
    try:
        with open(layout_json_path) as f:
            d = json.load(f)
        return float(d["width"]), float(d["height"])
    except Exception:
        return PDF_PAGE_W, PDF_PAGE_H


def pdf_to_img_coords(x0, y0, x1, y1, pdf_w, pdf_h, img_w, img_h):
    """Convert PDF bottom-left-origin coords to image top-left-origin pixel coords."""
    sx = img_w / pdf_w
    sy = img_h / pdf_h
    left  = x0 * sx
    right = x1 * sx
    # flip Y: PDF y=0 is bottom, image y=0 is top
    top = img_h - y0 * sy
    bot = img_h - y1 * sy
    return min(left, right), min(top, bot), max(left, right), max(top, bot)


def draw_bboxes_on_pages(json_path: str, output_dir: str):
    with open(json_path) as f:
        data = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Collect per (doc, page_no): list of bboxes and the image/layout paths
    page_bboxes:    dict[tuple, list] = {}
    page_image_map: dict[tuple, str]  = {}
    page_layout_map: dict[tuple, str] = {}

    for result in data.get("results", []):
        doc          = result.get("document_name", "")
        page_images  = result.get("files", {}).get("page_images", [])
        layout_jsons = result.get("files", {}).get("layout_json", [])
        provenance   = result.get("provenance_parsed", [])

        # page_no -> image path
        img_by_page: dict[int, str] = {}
        for img_path in page_images:
            stem = Path(img_path).stem          # e.g. "page_118"
            try:
                pno = int(stem.split("_")[-1])
                img_by_page[pno] = img_path
            except ValueError:
                pass

        # page_no -> layout json path
        layout_by_page: dict[int, str] = {}
        for lj_path in layout_jsons:
            stem = Path(lj_path).stem           # e.g. "page_118_layout"
            parts = stem.split("_")
            try:
                # stem is "page_<n>_layout"
                pno = int(parts[1])
                layout_by_page[pno] = lj_path
            except (ValueError, IndexError):
                pass

        for prov in provenance:
            pno = prov["page_no"]
            key = (doc, pno)
            page_bboxes.setdefault(key, []).append(
                (prov["x0"], prov["y0"], prov["x1"], prov["y1"])
            )
            if pno in img_by_page:
                page_image_map[key] = img_by_page[pno]
            if pno in layout_by_page:
                page_layout_map[key] = layout_by_page[pno]

    for (doc, pno), bboxes in page_bboxes.items():
        img_path = page_image_map.get((doc, pno))
        if not img_path or not os.path.exists(img_path):
            print(f"  [skip] image not found for {doc} page {pno}")
            continue

        layout_path = page_layout_map.get((doc, pno), "")
        pdf_w, pdf_h = get_pdf_page_dims(layout_path)

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        draw = ImageDraw.Draw(img)

        for (x0, y0, x1, y1) in bboxes:
            left, top, right, bot = pdf_to_img_coords(
                x0, y0, x1, y1, pdf_w, pdf_h, img_w, img_h
            )
            draw.rectangle([left, top, right, bot], outline="red", width=3)

        out_name = f"{doc}__page_{pno}.png"
        img.save(out / out_name)
        print(f"  saved: {out_name}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python pdf_layout_proof.py <results.json> <output_dir>")
        sys.exit(1)

    draw_bboxes_on_pages(sys.argv[1], sys.argv[2])
    print("Done.")
