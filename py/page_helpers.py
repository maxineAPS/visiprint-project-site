"""Helpers for the walkthrough's Python cells.

They load the figure's images, hand the result back to the page and note the search
steps for the log. The image processing itself is the release's code, taken unchanged
from visiprint_pipeline.py.
"""
import ast
import io
import linecache
import sys
import time
import traceback

import cv2
import numpy as np
from PIL import Image

ASSETS = "/assets"
shown = None


def load(name, mode="RGB", size=None):
    """Open one of this figure's images. size=512 resizes to 512 x 512, the way the
    pipeline resizes screenshots to 1024 x 1024."""
    img = Image.open(f"{ASSETS}/{name}").convert(mode)
    return img.resize((size, size), Image.BICUBIC) if size else img


def show(img):
    """Draw an image (PIL image or array) in the figure."""
    global shown
    shown = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img).astype(np.uint8))


def fixed_threshold(target, mask, threshold):
    """A p_range that makes the search's single step use this Canny high threshold:
    the percentile of the in-mask gradient magnitudes whose value is `threshold`.
    The magnitudes are computed exactly as get_adaptive_canny_edges computes them."""
    gray = cv2.cvtColor(np.array(target), cv2.COLOR_RGB2GRAY)
    inside = np.array(mask) > 128
    gray[~inside] = np.median(gray[inside])
    blurred = cv2.GaussianBlur(gray, (5, 5), 0.8)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    mags = np.sort(np.sqrt(gx * gx + gy * gy)[inside])
    p = 100 * float(np.interp(threshold, mags, np.arange(mags.size))) / (mags.size - 1)
    return (p, p)


class _NoteSteps(ast.NodeTransformer):
    """After every `dens = ...` line, add a call that notes the search step for the log.
    Nothing else changes, and line numbers stay those of visiprint_pipeline.py."""

    def generic_visit(self, node):
        super().generic_visit(node)
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if not (isinstance(stmts, list) and stmts and isinstance(stmts[0], ast.stmt)):
                continue
            new = []
            for st in stmts:
                new.append(st)
                if isinstance(st, ast.Assign) and any(getattr(t, "id", None) == "dens" for t in st.targets):
                    new.append(ast.copy_location(ast.parse("_note_step(locals())").body[0], st))
            setattr(node, field, new)
        return node


def _compile(src, filename):
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    tree = ast.fix_missing_locations(_NoteSteps().visit(ast.parse(src, filename)))
    return compile(tree, filename, "exec")


def run_cell(fn_src, first_line, call_src):
    """Define the (possibly edited) release function, run the reader's call, and
    return what the figure needs: the image, printed output, errors, search steps."""
    global shown
    shown = None
    steps = []

    def note_step(v):
        steps.append([float(v.get("p", 0)), float(v.get("t_high", 0)), float(v.get("dens", 0)),
                      float(v.get("lower", 0)), float(v.get("upper", 1)), int(v.get("max_iters", 0))])

    ns = {"np": np, "cv2": cv2, "Image": Image, "_note_step": note_step,
          "load": load, "show": show, "fixed_threshold": fixed_threshold}
    out, error = io.StringIO(), None
    saved = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = out
    t0 = time.perf_counter()
    try:
        # Padding keeps the line numbers of visiprint_pipeline.py in tracebacks.
        exec(_compile("\n" * (first_line - 1) + fn_src, "visiprint_pipeline.py"), ns)
        exec(_compile(call_src, "your cell"), ns)
    except SyntaxError as e:
        error = "".join(traceback.format_exception_only(type(e), e))
    except BaseException as e:  # show every error to the reader
        error = "".join(traceback.format_exception(type(e), e, e.__traceback__.tb_next))
    finally:
        sys.stdout, sys.stderr = saved
    ms = (time.perf_counter() - t0) * 1000
    image = None
    if shown is not None:
        rgba = shown.convert("RGBA")
        image = [rgba.width, rgba.height, rgba.tobytes()]
    return {"image": image, "stdout": out.getvalue(), "error": error, "steps": steps, "ms": ms}
