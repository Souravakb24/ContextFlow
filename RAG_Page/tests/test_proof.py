"""
test_proof.py — Tests for bbox coordinate conversion and overlay drawing.
"""

import pytest
from PIL import Image


def test_pdf_to_img_coords_basic():
    from app.utils.coord_utils import pdf_to_img_coords

    # PDF: 595 x 842 pts. Image: 595 x 842 px (1:1 scale, no scaling)
    # PDF bbox bottom-left origin: x0=10, y0=20, x1=100, y1=80
    # Image coords (top-left origin):
    #   left  = 10, right = 100
    #   top   = 842 - 80 = 762, bot = 842 - 20 = 822
    l, t, r, b = pdf_to_img_coords(10, 20, 100, 80, 595.0, 842.0, 595, 842, padding=0)
    assert l == 10
    assert r == 100
    assert t == 762
    assert b == 822


def test_pdf_to_img_coords_clamped():
    from app.utils.coord_utils import pdf_to_img_coords
    # bbox going out of bounds with padding
    l, t, r, b = pdf_to_img_coords(0, 0, 595, 842, 595.0, 842.0, 595, 842, padding=10)
    assert l == 0
    assert t == 0
    assert r == 595
    assert b == 842


def test_pdf_to_img_coords_scaling():
    from app.utils.coord_utils import pdf_to_img_coords
    # Image is 2× the PDF dimensions
    l, t, r, b = pdf_to_img_coords(10, 20, 100, 80, 595.0, 842.0, 1190, 1684, padding=0)
    assert l == 20
    assert r == 200
    # Y flipped and scaled ×2
    assert t == (842 - 80) * 2
    assert b == (842 - 20) * 2


def test_normalize_bbox_x0_form():
    from app.utils.coord_utils import normalize_bbox
    b = {"x0": 10.5, "y0": 20.3, "x1": 100.7, "y1": 80.1}
    n = normalize_bbox(b)
    assert n["x0"] == 10.5
    assert n["y1"] == 80.1


def test_normalize_bbox_l_form():
    from app.utils.coord_utils import normalize_bbox
    b = {"l": 10.0, "t": 20.0, "r": 100.0, "b": 80.0}
    n = normalize_bbox(b)
    assert n["x0"] == 10.0
    assert n["y0"] == 20.0
    assert n["x1"] == 100.0
    assert n["y1"] == 80.0


def test_normalize_bbox_none():
    from app.utils.coord_utils import normalize_bbox
    assert normalize_bbox(None) is None
    assert normalize_bbox({}) is None
    assert normalize_bbox("bad") is None


def test_draw_bboxes_on_page(tmp_path):
    from app.core.proof import draw_bboxes_on_page

    # Create a blank 595×842 white image
    img_path = tmp_path / "page_1.png"
    img = Image.new("RGB", (595, 842), color="white")
    img.save(str(img_path))

    bboxes = [
        {"x0": 50, "y0": 700, "x1": 300, "y1": 750, "label": "[1]", "color_idx": 0},
    ]
    result = draw_bboxes_on_page(str(img_path), bboxes, pdf_w=595, pdf_h=842)

    assert isinstance(result, Image.Image)
    assert result.size == (595, 842)
    # Check that at least one pixel changed from white (bbox was drawn)
    pixels = list(result.getdata())
    assert any(p != (255, 255, 255) for p in pixels)
