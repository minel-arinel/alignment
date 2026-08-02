

import re

import SimpleITK as sitk
import numpy as np

import os
from pathlib import Path

try:
    import cv2
except:
    print("cv2 not available")


def load_image(image_path):
    """
    Load a 2D/3D image from TIFF, PNG, or other common formats.

    Falls back to Pillow when ``tifffile`` cannot decode compressed TIFFs
    (e.g. LZW without ``imagecodecs`` installed).
    """
    path = Path(image_path)
    suffix = path.suffix.lower()

    if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".gif"):
        from PIL import Image

        return np.asarray(Image.open(path))

    try:
        from tifffile import imread

        return imread(str(path))
    except ValueError as exc:
        msg = str(exc).lower()
        if "imagecodecs" in msg or "compression" in msg:
            from PIL import Image

            return np.asarray(Image.open(path))
        raise


def embed_image(image, default_size=1024):
    """
    puts images inside a black cube - standardizes size and such (helpful sometimes)
    :param image:
    :param default_size:
    :return:
    """
    if image.ndim == 3:
        print("please input 2D image")
        return

    while max(image.shape) > default_size:
        default_size *= 2
        print(f"increasing default size to {default_size}")

    new_image = np.zeros((default_size, default_size))
    midpt = default_size // 2

    image = np.clip(image, a_min=0, a_max=2**16)

    offset_y = 0
    ydim = image.shape[0]
    if ydim % 2 != 0:  # if its odd kick it one pixel
        offset_y += 1
    offset_x = 0
    xdim = image.shape[1]
    if xdim % 2 != 0:  # if its odd kick it one pixel
        offset_x += 1

    new_image[
        midpt - ydim // 2 : midpt + ydim // 2 + offset_y,
        midpt - xdim // 2 : midpt + xdim // 2 + offset_x,
    ] = image

    return new_image


def rotate_image_2d(image, angle_degrees_clockwise, cval=0.0):
    """
    Rotate a 2D image in-place on the same grid (same output shape as input).

    Positive ``angle_degrees_clockwise`` rotates clockwise as displayed by ``imshow``
    (origin='upper').
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"rotate_image_2d expects a 2D array, got shape {arr.shape}")
    sitk_img = sitk.GetImageFromArray(arr)
    size = sitk_img.GetSize()  # (columns, rows)
    center_index = [(size[0] - 1) / 2.0, (size[1] - 1) / 2.0]
    center = sitk_img.TransformContinuousIndexToPhysicalPoint(center_index)
    transform = sitk.Euler2DTransform()
    transform.SetCenter(center)
    transform.SetAngle(-np.deg2rad(float(angle_degrees_clockwise)))
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(cval))
    resampler.SetTransform(transform)
    out = resampler.Execute(sitk_img)
    return sitk.GetArrayFromImage(out)


def experiment_std_to_plot_img(experiment_std_scaled, cw_degrees=0, embed_size=1024):
    """
    Build the embedded experiment canvas used for registration.

    Applies optional fine clockwise rotation on the native STD, then ``rot90`` (k=1)
    and :func:`embed_image` — same pipeline as ``brain_alignment_from_tiff_masks.ipynb``.
    """
    work = np.asarray(experiment_std_scaled, dtype=np.float64)
    if work.ndim != 2:
        raise ValueError(
            f"experiment_std_to_plot_img expects a 2D array, got shape {work.shape}"
        )
    if cw_degrees:
        work = rotate_image_2d(work, cw_degrees, cval=0.0)
    work = np.rot90(work, k=1)
    return embed_image(work, embed_size)


def rotate_points_clockwise_native(
    points_xy,
    native_shape_hw,
    angle_degrees_clockwise,
):
    """
    Rotate ``(x, y)`` column/row points clockwise on a native ``(H, W)`` image.

    Uses the same y-down / ``imshow`` convention as :func:`rotate_image_2d`.
    Pure NumPy (no SimpleITK) so notebook COM cells always pick up the rotation.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points_xy must be shape (N, 2)")
    h, w = int(native_shape_hw[0]), int(native_shape_hw[1])
    angle = float(angle_degrees_clockwise)
    if abs(angle) < 1e-9:
        return pts.copy()

    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    theta = np.deg2rad(angle)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    x_new = cos_t * dx + sin_t * dy + cx
    y_new = -sin_t * dx + cos_t * dy + cy
    return np.column_stack([x_new, y_new])


def _com_mesmerize_via_experiment_std_pipeline(
    points_xy,
    experiment_std,
    cw_degrees=0,
    embed_size=1024,
):
    """
    Map each COM by running :func:`experiment_std_to_plot_img` on a native impulse.

    Guarantees identical CW / rot90 / embed behaviour to ``plot_img`` (no separate
    point-rotation formula that can drift from the image resampler).
    """
    ref = np.asarray(experiment_std, dtype=np.float64)
    if ref.ndim != 2:
        raise ValueError(f"experiment_std must be 2D, got shape {ref.shape}")
    h, w = ref.shape[:2]
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    cw = float(cw_degrees)
    out = np.empty((len(pts), 2), dtype=np.float64)
    for i, (x, y) in enumerate(pts):
        imp = np.zeros((h, w), dtype=np.float64)
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if 0 <= yi < h and 0 <= xi < w:
            imp[yi, xi] = 1.0
        canvas = experiment_std_to_plot_img(imp, cw_degrees=cw, embed_size=embed_size)
        iy, ix = np.unravel_index(int(np.argmax(canvas)), canvas.shape)
        out[i] = (float(ix), float(iy))
    return out


def com_mesmerize_to_plot_img_xy(
    points_xy,
    native_shape_hw,
    cw_degrees=0,
    embed_size=1024,
    experiment_std=None,
):
    """
    Map CaImAn / mesmerize ROI centers into embedded ``plot_img`` coordinates.

    ``points_xy`` are ``(px, py)`` column/row on the native STD (same as
    ``brain_alignment_from_tiff`` ``com_to_target_xy`` input).

    Pass ``experiment_std`` (e.g. ``experiment_std_scaled``) so each COM is pushed
    through :func:`experiment_std_to_plot_img` — the **same** native CW → rot90 →
    embed path as ``plot_img``. Without it, an analytic rot90+embed path is used
    (CW via :func:`transform_points_rotate_clockwise`).

    When ``cw_degrees=0`` and native shape is ``(640, 1024)``, the result matches
    ``com_to_target_xy`` with ``SHIFT_X = (1024 - 640) // 2``.
    """
    if experiment_std is not None:
        ref = np.asarray(experiment_std, dtype=np.float64)
        h, w = int(native_shape_hw[0]), int(native_shape_hw[1])
        if ref.shape[:2] != (h, w):
            raise ValueError(
                f"experiment_std shape {ref.shape[:2]} != native_shape_hw {h, w}"
            )
        return _com_mesmerize_via_experiment_std_pipeline(
            points_xy,
            experiment_std,
            cw_degrees=cw_degrees,
            embed_size=embed_size,
        )

    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim == 1:
        if pts.size != 2:
            raise ValueError(f"points_xy must have length 2, got shape {pts.shape}")
        pts = pts.reshape(1, 2)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points_xy must be shape (N, 2) or a length-2 vector")

    h, w = int(native_shape_hw[0]), int(native_shape_hw[1])
    cw = float(cw_degrees)
    if abs(cw) > 1e-9:
        pts = transform_points_rotate_clockwise(pts, (h, w), cw)

    rotated = np.column_stack([pts[:, 1], w - 1 - pts[:, 0]])
    embedded, _ = embed_points_xy(rotated, (w, h), embed_size)
    return embedded


def com_to_target_plot_img_xy(
    points_xy,
    native_shape_hw,
    cw_degrees=0,
    embed_size=1024,
    experiment_std=None,
):
    """Alias for :func:`com_mesmerize_to_plot_img_xy`."""
    return com_mesmerize_to_plot_img_xy(
        points_xy,
        native_shape_hw,
        cw_degrees=cw_degrees,
        embed_size=embed_size,
        experiment_std=experiment_std,
    )


def save_embedded_plot_img(plot_img, path, embed_size=1024, vmin=None, vmax=None):
    """
    Write a 1024² (or ``embed_size``²) embedded plot image to PNG.

    Scales ``plot_img`` to 8-bit grayscale using ``vmin`` / ``vmax`` (defaults to
    array min/max). Returns the resolved output path as a :class:`pathlib.Path`.
    """
    from PIL import Image

    arr = np.asarray(plot_img, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"save_embedded_plot_img expects a 2D array, got shape {arr.shape}")
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if (h, w) != (int(embed_size), int(embed_size)):
        raise ValueError(
            f"Expected embedded plot image {embed_size}×{embed_size}, got {arr.shape[:2]}"
        )

    lo = float(np.min(arr) if vmin is None else vmin)
    hi = float(np.max(arr) if vmax is None else vmax)
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    img_u8 = (scaled * 255.0).astype(np.uint8)

    out_path = Path(path)
    if out_path.suffix.lower() != ".png":
        out_path = out_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_u8, mode="L").save(out_path)
    return out_path


def embed_content_bounds(native_shape, default_size=1024):
    """
    Row/col slice bounds where ``embed_image`` places ``native_shape`` in a square canvas.

    Returns (canvas_size, y0, y1, x0, x1) with Python half-open slices ``canvas[y0:y1, x0:x1]``.
    """
    ydim, xdim = int(native_shape[0]), int(native_shape[1])
    canvas_size = int(default_size)
    while max(ydim, xdim) > canvas_size:
        canvas_size *= 2
    midpt = canvas_size // 2
    offset_y = 1 if ydim % 2 else 0
    offset_x = 1 if xdim % 2 else 0
    y0 = midpt - ydim // 2
    y1 = midpt + ydim // 2 + offset_y
    x0 = midpt - xdim // 2
    x1 = midpt + xdim // 2 + offset_x
    return canvas_size, y0, y1, x0, x1


def registration_mask_from_exclusions(
    shape,
    exclude_top=0,
    exclude_bottom=0,
    exclude_left=0,
    exclude_right=0,
):
    """
    Binary mask for Elastix / ITK registration (1 = use in metric, 0 = ignore).

    Row 0 is the top of the array as shown by ``imshow`` (increasing row = down).
    """
    h, w = shape[:2]
    mask = np.ones((h, w), dtype=np.uint8)
    if exclude_top:
        mask[: int(exclude_top), :] = 0
    if exclude_bottom:
        mask[h - int(exclude_bottom) :, :] = 0
    if exclude_left:
        mask[:, : int(exclude_left)] = 0
    if exclude_right:
        mask[:, w - int(exclude_right) :] = 0
    return mask


def binarize_registration_mask(arr, positive_threshold=0):
    """uint8 mask with 1 where ``arr > positive_threshold``."""
    return (np.asarray(arr) > float(positive_threshold)).astype(np.uint8)


def load_mask_array(mask_path):
    """Load a 2D mask from a TIFF or PNG (uses first plane if 3D)."""
    arr = load_image(str(mask_path))
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Mask {mask_path} must be 2D or Z-stack, got shape {arr.shape}")
    return arr


def registration_mask_to_canvas(mask, target_shape, embed_size=1024):
    """
    Binarize and place ``mask`` on ``target_shape`` (H, W).

    If the mask already matches ``target_shape``, return it. Otherwise ``embed_image`` is
    used (same centering as intensity images).
    """
    h, w = int(target_shape[0]), int(target_shape[1])
    m = binarize_registration_mask(mask)
    if m.shape == (h, w):
        return m
    if max(m.shape) <= embed_size:
        emb = embed_image(m.astype(np.float64), embed_size)
        if emb.shape != (h, w):
            raise ValueError(
                f"Embedded mask shape {emb.shape} != target {target_shape[:2]} "
                f"(embed_size={embed_size})"
            )
        return binarize_registration_mask(emb)
    raise ValueError(
        f"Mask shape {m.shape} does not match target {target_shape[:2]} "
        f"and is too large to embed (embed_size={embed_size})"
    )


def transform_points_rotate_clockwise(points_xy, shape_hw, angle_degrees_clockwise):
    """
    Map native ``(x, y)`` points through the same CW rotation as :func:`rotate_image_2d`.

    ``points_xy`` are column/row indices (x = col, y = row, origin upper-left).

    Elastix/SimpleITK resampling maps **output** indices to **input** indices, so the
    forward map ``input → output`` uses the transform **inverse** (ITK
    ``ResampleImageFilter``).
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points_xy must be shape (N, 2)")
    h, w = int(shape_hw[0]), int(shape_hw[1])
    angle = float(angle_degrees_clockwise)
    if abs(angle) < 1e-9:
        return pts.copy()

    sitk_img = sitk.GetImageFromArray(np.zeros((h, w), np.float32))
    center_index = [(w - 1) / 2.0, (h - 1) / 2.0]
    center_phys = sitk_img.TransformContinuousIndexToPhysicalPoint(center_index)
    transform = sitk.Euler2DTransform()
    transform.SetCenter(center_phys)
    transform.SetAngle(-np.deg2rad(angle))
    forward = transform.GetInverse()

    out = []
    for x, y in pts:
        phys = sitk_img.TransformContinuousIndexToPhysicalPoint((float(x), float(y)))
        phys2 = forward.TransformPoint(phys)
        idx = sitk_img.TransformPhysicalPointToContinuousIndex(phys2)
        out.append((float(idx[0]), float(idx[1])))
    return np.asarray(out, dtype=np.float64)


def embed_points_xy(points_xy, native_shape_hw, embed_size=1024):
    """Map ``(x, y)`` from a native array into embedded canvas coordinates."""
    pts = np.asarray(points_xy, dtype=np.float64)
    h, w = int(native_shape_hw[0]), int(native_shape_hw[1])
    _canvas, y0, _y1, x0, _x1 = embed_content_bounds((h, w), embed_size)
    embedded = pts.copy()
    embedded[:, 0] = pts[:, 0] + float(x0)
    embedded[:, 1] = pts[:, 1] + float(y0)
    return embedded, int(_canvas)


def mask_bottom_corners_after_rot90(mask_rot90):
    """
    Bottom-left / bottom-right on a rot90 mask: min-x column max-y, max-x column max-y.

    Returns ``((x_left, y_left), (x_right, y_right))`` as column/row indices.
    """
    m = binarize_registration_mask(mask_rot90)
    cols = np.where(m.any(axis=0))[0]
    if cols.size == 0:
        raise ValueError("Empty mask")
    x_left = int(cols[0])
    x_right = int(cols[-1])
    y_left = int(np.max(np.where(m[:, x_left])[0]))
    y_right = int(np.max(np.where(m[:, x_right])[0]))
    return (x_left, y_left), (x_right, y_right)


def _fill_rows_between_left_right_extrema(mask):
    """Per row, fill between leftmost and rightmost foreground pixel (closes V-shaped gaps)."""
    out = np.asarray(mask, dtype=np.uint8).copy()
    h, w = out.shape[:2]
    for r in range(h):
        xs = np.where(out[r])[0]
        if xs.size >= 2:
            out[r, int(xs[0]) : int(xs[-1]) + 1] = 1
    return out


def build_closed_top_region_mask_from_native(
    mask_path,
    plot_img_shape,
    experiment_std_scaled=None,
    cw_degrees=0,
    embed_size=1024,
    positive_threshold=0,
):
    """
    Build the top piecewise region mask on the embedded experiment canvas.

    Pipeline (matches hand-drawn mask aligned to native STD before notebook transforms):

    1. Binarize native mask, ``rot90`` (k=1, same as experiment).
    2. Bottom-left = max row on min-x column; bottom-right = max row on max-x column.
    3. Fine clockwise rotation on the mask; rotate both corner points the same way.
    4. ``embed_image`` to 1024²; shift corners into embed coordinates.
    5. Draw horizontals from ``(0, y_left)`` to ``(x_left, y_left)`` and
       ``(x_right, y_right)`` to the right edge.
    6. Row-wise fill between left/right extrema to close and fill the top region.

    Returns ``(mask_uint8, info_dict)``.
    """
    target_h, target_w = int(plot_img_shape[0]), int(plot_img_shape[1])
    raw = load_mask_array(mask_path)
    m = binarize_registration_mask(raw, positive_threshold=positive_threshold)

    if experiment_std_scaled is not None:
        native_shape = np.asarray(experiment_std_scaled).shape[:2]
        if m.shape != native_shape:
            raise ValueError(
                f"Mask shape {m.shape} != experiment native {native_shape}. "
                "Draw the mask on the native STD or match shapes."
            )

    work = np.rot90(m, k=1)
    (x_l, y_l), (x_r, y_r) = mask_bottom_corners_after_rot90(work)

    if cw_degrees:
        work = binarize_registration_mask(
            rotate_image_2d(work.astype(np.float64), cw_degrees, cval=0.0)
        )
        corners = transform_points_rotate_clockwise(
            np.array([[x_l, y_l], [x_r, y_r]], dtype=np.float64),
            work.shape,
            cw_degrees,
        )
        x_l, y_l = corners[0]
        x_r, y_r = corners[1]

    embedded = binarize_registration_mask(embed_image(work.astype(np.float64), embed_size))
    if embedded.shape != (target_h, target_w):
        raise ValueError(
            f"Embedded mask {embedded.shape} != plot_img {plot_img_shape[:2]}"
        )

    corners_emb, _canvas = embed_points_xy(
        np.array([[x_l, y_l], [x_r, y_r]], dtype=np.float64),
        work.shape,
        embed_size=embed_size,
    )
    x_l_e, y_l_e = corners_emb[0]
    x_r_e, y_r_e = corners_emb[1]

    xl = int(np.clip(np.round(x_l_e), 0, target_w - 1))
    xr = int(np.clip(np.round(x_r_e), 0, target_w - 1))
    yl = int(np.clip(np.round(y_l_e), 0, target_h - 1))
    yr = int(np.clip(np.round(y_r_e), 0, target_h - 1))

    out = embedded.copy()
    out[yl, : xl + 1] = 1
    out[yr, xr:target_w] = 1
    out = _fill_rows_between_left_right_extrema(out)

    info = {
        "corners_rot90_native": mask_bottom_corners_after_rot90(np.rot90(m, k=1)),
        "corners_after_cw": (float(x_l), float(y_l), float(x_r), float(y_r)),
        "corners_embedded": ((xl, yl), (xr, yr)),
        "rot90_shape": tuple(int(s) for s in work.shape),
    }
    return out.astype(np.uint8), info


def prepare_experiment_mask_for_plot_img(
    mask_path,
    plot_img,
    experiment_std_scaled=None,
    cw_degrees=0,
    embed_size=1024,
    positive_threshold=0,
):
    """
    Load an experiment metric mask on the same 1024² canvas as ``plot_img``.

    Accepts masks drawn on ``plot_img`` directly, or on the native STD before the
    notebook pipeline (``experiment_std_scaled`` + optional CW rotation + rot90 + embed).
    """
    target_shape = plot_img.shape[:2]
    raw = load_mask_array(mask_path)
    m = binarize_registration_mask(raw, positive_threshold=positive_threshold)
    if m.shape == target_shape:
        return m

    if experiment_std_scaled is None:
        raise ValueError(
            f"Mask {mask_path} shape {m.shape} != plot_img {target_shape}. "
            "Pass experiment_std_scaled so native masks can be rotated/embedded like the STD."
        )
    native_shape = np.asarray(experiment_std_scaled).shape[:2]
    if m.shape != native_shape:
        return registration_mask_to_canvas(m, target_shape, embed_size=embed_size)

    work = m.astype(np.float64)
    if cw_degrees:
        work = rotate_image_2d(work, cw_degrees, cval=0.0)
    work = np.rot90(work, k=1)
    work = embed_image(work, embed_size)
    out = binarize_registration_mask(work, positive_threshold=positive_threshold)
    if out.shape != target_shape:
        raise ValueError(
            f"Prepared experiment mask shape {out.shape} != plot_img {target_shape}"
        )
    return out


def prepare_atlas_mask_for_embedded_slice(
    mask_path,
    embedded_shape,
    native_shape=None,
    embed_size=1024,
    positive_threshold=0,
):
    """Load a moving (atlas) metric mask on the embedded atlas canvas."""
    raw = load_mask_array(mask_path)
    m = binarize_registration_mask(raw, positive_threshold=positive_threshold)
    if m.shape == embedded_shape[:2]:
        return m
    if native_shape is not None and m.shape == tuple(native_shape)[:2]:
        emb = embed_image(m.astype(np.float64), embed_size)
        out = binarize_registration_mask(emb, positive_threshold=positive_threshold)
        if out.shape != embedded_shape[:2]:
            raise ValueError(
                f"Embedded atlas mask {out.shape} != atlas slice {embedded_shape[:2]}"
            )
        return out
    return registration_mask_to_canvas(m, embedded_shape, embed_size=embed_size)


def mask_embedded_row_bounds(mask):
    """Tight [row_start, row_end) span of positive mask rows (for piecewise QC / inverse map)."""
    m = np.asarray(mask, dtype=np.uint8)
    rows = np.where(m.any(axis=1))[0]
    if rows.size == 0:
        raise ValueError("Mask has no positive rows")
    return int(rows[0]), int(rows[-1]) + 1


def validate_piecewise_region_masks(masks_by_name, warn_overlap=True):
    """
    Check uint8 region masks on a shared canvas: non-empty, unique names, optional overlap warn.
    """
    names = list(masks_by_name.keys())
    if len(names) != len(set(names)):
        raise ValueError("Duplicate region names in masks_by_name")
    combined = None
    for name in names:
        m = np.asarray(masks_by_name[name], dtype=np.uint8)
        if m.ndim != 2:
            raise ValueError(f"{name!r}: mask must be 2D, got {m.shape}")
        if int(m.sum()) == 0:
            raise ValueError(f"{name!r}: mask is empty")
        if combined is None:
            combined = np.zeros_like(m, dtype=np.int32)
        elif m.shape != combined.shape:
            raise ValueError(f"{name!r}: shape {m.shape} != {combined.shape}")
        overlap = combined & m.astype(bool)
        if warn_overlap and overlap.any():
            print(
                f"[piecewise] warning: {name!r} overlaps prior regions "
                f"({int(overlap.sum())} px)"
            )
        combined += m.astype(np.int32)
    gap = int((combined == 0).sum())
    if gap:
        print(
            f"[piecewise] {gap} px not covered by any region mask "
            "(OK if you do not need alignment there)."
        )


def build_atlas_metric_regions_from_experiment_sections(
    experiment_sections,
    mapzebrain_by_section,
    col_span="content",
    embed_size=1024,
):
    """Alias for :func:`build_atlas_metric_regions_from_experiment_bands`."""
    return build_atlas_metric_regions_from_experiment_bands(
        experiment_sections,
        mapzebrain_by_section,
        col_span=col_span,
        embed_size=embed_size,
    )


def atlas_metric_roi_for_experiment_band(
    experiment_band,
    native_shape,
    embed_size=1024,
    col_span="content",
):
    """
    Moving-metric ROI on the embedded atlas canvas from an experiment row band.

    Uses the same ``row_start`` / ``row_end`` as ``EXPERIMENT_BANDS`` (embedded 1024²,
    row 0 = top). ``col_span`` is ``content`` (default: atlas embed x-bounds only) or
    ``full`` (columns 0–embed_size).
    """
    canvas_size, _y0, _y1, _x0, _x1 = embed_content_bounds(native_shape, embed_size)
    h = w = int(canvas_size)
    rs, re = int(experiment_band["row_start"]), int(experiment_band["row_end"])
    if not (0 <= rs < re <= h):
        raise ValueError(
            f"Experiment band rows [{rs}, {re}) invalid for embed canvas height {h}"
        )
    span = str(col_span).lower().strip()
    if span == "full":
        cs, ce = 0, w
    elif span == "content":
        cs, ce = int(_x0), int(_x1)
    else:
        raise ValueError('col_span must be "content" or "full"')
    return {
        "row_start": rs,
        "row_end": re,
        "col_start": cs,
        "col_end": ce,
    }


def build_atlas_metric_regions_from_experiment_bands(
    experiment_bands,
    mapzebrain_by_band,
    col_span="content",
    embed_size=1024,
):
    """
    One atlas metric ROI per band: same embedded rows as experiment, atlas x-span from
    ``col_span``. Keys in ``mapzebrain_by_band`` must match ``experiment_bands[*]['name']``.
    """
    regions = []
    for band in experiment_bands:
        name = str(band["name"])
        if name not in mapzebrain_by_band:
            raise KeyError(f"{name!r} missing from mapzebrain_by_band")
        native_shape = mapzebrain_by_band[name]["native_shape"]
        regions.append(
            atlas_metric_roi_for_experiment_band(
                band, native_shape, embed_size=embed_size, col_span=col_span
            )
        )
    return regions


def moving_metric_mask_from_roi(embedded_shape, native_shape, roi, coord_frame="embedded"):
    """
    Build a uint8 moving-image metric mask from a band ROI dict.

    ``roi`` must include row_start, row_end, col_start, col_end.
    ``coord_frame`` is ``embedded`` (1024 canvas) or ``native`` (pre-embed slice).
    """
    roi = dict(roi)
    frame = str(coord_frame).lower().strip()
    if frame == "embedded":
        return registration_mask_embedded_rectangle(embedded_shape, **roi)
    if frame == "native":
        return registration_mask_native_rectangle(
            embedded_shape, native_shape, **roi
        )
    raise ValueError('coord_frame must be "embedded" or "native"')


def registration_mask_embedded_rectangle(
    shape,
    row_start,
    row_end,
    col_start,
    col_end,
):
    """
    Binary mask on the embedded 1024² canvas: 1 inside the half-open rectangle, 0 outside.

    Coordinates match ``imshow`` on images after ``embed_image`` (row 0 = top).
    """
    h, w = int(shape[0]), int(shape[1])
    rs, re = int(row_start), int(row_end)
    cs, ce = int(col_start), int(col_end)
    if not (0 <= rs < re <= h and 0 <= cs < ce <= w):
        raise ValueError(
            f"Invalid embedded ROI [{rs}, {re}) x [{cs}, {ce}) for shape ({h}, {w})"
        )
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[rs:re, cs:ce] = 1
    return mask


def registration_mask_native_rectangle(
    embedded_shape,
    native_shape,
    row_start,
    row_end,
    col_start,
    col_end,
    embed_size=1024,
):
    """
    Metric mask on the embed canvas for a rectangle in **native** atlas pixels.

    Native rows/cols are relative to the raw slice before ``embed_image``; the ROI is
    translated into embedded coordinates via :func:`embed_content_bounds`.
    """
    canvas_size, y0, y1, x0, x1 = embed_content_bounds(native_shape, embed_size)
    h, w = embedded_shape[:2]
    if (h, w) != (canvas_size, canvas_size):
        raise ValueError(
            f"embedded_shape {embedded_shape[:2]} != embed canvas ({canvas_size}, {canvas_size})"
        )
    ny, nx = int(native_shape[0]), int(native_shape[1])
    rs, re = int(row_start), int(row_end)
    cs, ce = int(col_start), int(col_end)
    if not (0 <= rs < re <= ny and 0 <= cs < ce <= nx):
        raise ValueError(
            f"Invalid native ROI [{rs}, {re}) x [{cs}, {ce}) for native shape ({ny}, {nx})"
        )
    return registration_mask_embedded_rectangle(
        embedded_shape,
        y0 + rs,
        y0 + re,
        x0 + cs,
        x0 + ce,
    )


def registration_mask_combine(*masks, min_sum=1):
    """Element-wise product of uint8 masks (1 = included in metric)."""
    if not masks:
        raise ValueError("At least one mask required")
    out = np.asarray(masks[0], dtype=np.uint8).copy()
    for m in masks[1:]:
        out = (out.astype(np.uint8) * np.asarray(m, dtype=np.uint8)).astype(np.uint8)
    if int(out.sum()) < int(min_sum):
        raise ValueError("Combined mask has no valid voxels")
    return out


def apply_moving_blackout(image, mask):
    """
    Zero image intensities outside a uint8 mask (1 = keep).

    Apply before registration so voxels outside the ROI are not passed to Elastix.
    Works for fixed (experiment) or moving (atlas) images; metric masks are optional.
    """
    img = np.asarray(image, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    if img.shape != m.shape:
        raise ValueError(f"image shape {img.shape} != mask shape {m.shape}")
    return img * m


apply_registration_blackout = apply_moving_blackout


def _mask_to_sitk(mask, reference_shape):
    if mask is None:
        return None
    if isinstance(mask, sitk.Image):
        return mask
    arr = np.asarray(mask)
    if arr.shape != reference_shape:
        raise ValueError(
            f"Mask shape {arr.shape} does not match image shape {reference_shape}"
        )
    return sitk.GetImageFromArray(arr.astype(np.uint8))


def _attach_mask_to_image(mask, reference_image):
    """uint8 mask with same geometry as a SimpleITK reference image (for Elastix)."""
    if mask is None:
        return None
    ref = reference_image if isinstance(reference_image, sitk.Image) else None
    if ref is None:
        raise TypeError("reference_image must be a sitk.Image")
    shape = (ref.GetHeight(), ref.GetWidth())
    mask_img = _mask_to_sitk(mask, shape)
    if mask_img is None:
        return None
    mask_img = sitk.Cast(mask_img, sitk.sitkUInt8)
    mask_img.CopyInformation(ref)
    return mask_img


def _configure_elastix_parameter_map_for_masks(pmap):
    """Use a sampler that respects sparse valid regions inside masks."""
    pmap["ImageSampler"] = ["RandomSparseMask"]
    for key in (
        "ErodeMask",
        "ErodeFixedMask",
        "ErodeMovingMask",
        "ErodeFixedMask2",
        "ErodeMovingMask2",
    ):
        if key in pmap:
            del pmap[key]
    return pmap


def _configure_elastix_parameter_map_for_numpy_images(pmap):
    """Arrays from GetImageFromArray have identity direction cosines."""
    pmap["UseDirectionCosines"] = ["false"]
    pmap["ResultImageFormat"] = ["mha"]
    return pmap


def _find_transformix_executable():
    """Sibling of :func:`_find_elastix_executable` in ``tools/elastix/bin``."""
    import os
    import shutil

    elastix = _find_elastix_executable()
    if elastix:
        sibling = Path(elastix).with_name("transformix")
        if sibling.is_file() and os.access(sibling, os.X_OK):
            return str(sibling.resolve())
    which = shutil.which("transformix")
    if which:
        return os.path.abspath(which)
    bundled = _bundled_elastix_bin_dir() / "transformix"
    if bundled.is_file() and os.access(bundled, os.X_OK):
        return str(bundled.resolve())
    return None


def _elastix_result_image_path(out_dir: Path):
    """
    Latest ``result.N.*`` written by elastix (extension may be mha, nii, etc.).
    """
    import re

    best_idx = -1
    best_path = None
    for path in out_dir.iterdir():
        m = re.fullmatch(r"result\.(\d+)\.[^.]+", path.name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx >= best_idx:
            best_idx = idx
            best_path = path
    return best_path


def _copy_elastix_transform_pmaps(out_dir: Path):
    """Copy ``TransformParameters.N.txt`` to ``transform_pmap_N.txt`` for transformix."""
    import shutil

    tp_files = sorted(
        out_dir.glob("TransformParameters.*.txt"),
        key=lambda p: int(p.stem.split(".")[-1]),
    )
    for n, src in enumerate(tp_files):
        shutil.copy2(src, out_dir / f"transform_pmap_{n}.txt")
    return tp_files


def _elastix_log_excerpt(out_dir: Path, max_lines: int = 40) -> str:
    log_path = out_dir / "elastix.log"
    if not log_path.is_file():
        return ""
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-max_lines:])


def _warp_moving_with_saved_transforms(moving_image_path: Path, out_dir: Path) -> np.ndarray:
    """Apply ``transform_pmap_*.txt`` in ``out_dir`` via SimpleITK transformix."""
    moving_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(moving_image_path)))
    return transform_image_from_saved(moving_arr, str(out_dir))


def _configure_elastix_parameter_maps_for_masks(parameter_map_vector):
    for i in range(parameter_map_vector.size()):
        parameter_map_vector[i] = _configure_elastix_parameter_map_for_masks(
            parameter_map_vector[i]
        )


def _alignment_repo_root() -> Path:
    """Repository root (parent of ``registration/``)."""
    return Path(__file__).resolve().parents[1]


def _bundled_elastix_bin_dir() -> Path:
    """``tools/elastix/bin`` from :func:`scripts/install_elastix.sh`."""
    return _alignment_repo_root() / "tools" / "elastix" / "bin"


def _find_elastix_executable():
    """Locate a standalone elastix binary (not bundled inside SimpleITK)."""
    import os
    import shutil

    candidates = []
    for env_key in ("ELASTIX", "ELASTIX_EXECUTABLE"):
        env_val = os.environ.get(env_key)
        if env_val:
            candidates.append(env_val)
    which = shutil.which("elastix")
    if which:
        candidates.append(which)
    bundled = _bundled_elastix_bin_dir() / "elastix"
    candidates.append(str(bundled))
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(os.path.join(conda_prefix, "bin", "elastix"))
    candidates.extend(
        (
            "/opt/homebrew/bin/elastix",
            "/usr/local/bin/elastix",
        )
    )
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return os.path.abspath(path)
    return None


def elastix_executable_info() -> dict:
    """
    Report how mask-aware registration will run.

    Returns keys: ``path`` (str or None), ``masks_via_cli`` (bool), ``install_hint`` (str).
    """
    path = _find_elastix_executable()
    if path:
        return {
            "path": path,
            "masks_via_cli": True,
            "install_hint": "",
        }
    script = _alignment_repo_root() / "scripts" / "install_elastix.sh"
    return {
        "path": None,
        "masks_via_cli": False,
        "install_hint": (
            f"Masks require the elastix CLI. Install with:\n"
            f"  bash {script}\n"
            f"Or set ELASTIX_EXECUTABLE to your elastix binary."
        ),
    }


def require_elastix_for_masks():
    """Raise ``RuntimeError`` if the elastix CLI is missing (needed for -fMask/-mMask)."""
    info = elastix_executable_info()
    if info["path"]:
        return info["path"]
    raise RuntimeError(info["install_hint"])


def _register_image2_elastix_cli(
    reference_image,
    align_image,
    fixed_mask_image,
    moving_mask_image,
    savepath,
    iterations,
    scalePenalty,
):
    """
    Run affine + B-spline via the elastix binary so -fMask/-mMask are honored.

    SimpleITK's ElastixImageFilter often omits mask flags on the command line.
    """
    import shutil
    import subprocess
    from pathlib import Path

    out = Path(savepath)
    out.mkdir(parents=True, exist_ok=True)

    fixed_path = out / "fixed_image.mha"
    moving_path = out / "moving_image.mha"
    sitk.WriteImage(reference_image, str(fixed_path))
    sitk.WriteImage(align_image, str(moving_path))

    fmask_path = None
    mmask_path = None
    if fixed_mask_image is not None:
        fmask_path = out / "metric_fixed_mask.mha"
        sitk.WriteImage(fixed_mask_image, str(fmask_path))
    if moving_mask_image is not None:
        mmask_path = out / "metric_moving_mask.mha"
        sitk.WriteImage(moving_mask_image, str(mmask_path))

    param_dir = out / "elastix_params"
    param_dir.mkdir(exist_ok=True)

    use_masks = fixed_mask_image is not None or moving_mask_image is not None
    pmap1 = sitk.GetDefaultParameterMap("affine")
    pmap1["MaximumNumberOfIterations"] = [str(iterations[0])]
    pmap1 = _configure_elastix_parameter_map_for_numpy_images(pmap1)
    if use_masks:
        pmap1 = _configure_elastix_parameter_map_for_masks(pmap1)

    pmap2 = sitk.GetDefaultParameterMap("bspline")
    pmap2["MaximumNumberOfIterations"] = [str(iterations[1])]
    pmap2["Metric0Weight"] = ["0.1"]
    pmap2["Metric1Weight"] = [str(scalePenalty)]
    pmap2 = _configure_elastix_parameter_map_for_numpy_images(pmap2)
    if use_masks:
        pmap2 = _configure_elastix_parameter_map_for_masks(pmap2)

    affine_txt = param_dir / "affine.txt"
    bspline_txt = param_dir / "bspline.txt"
    sitk.WriteParameterFile(pmap1, str(affine_txt))
    sitk.WriteParameterFile(pmap2, str(bspline_txt))

    elastix = _find_elastix_executable()
    if elastix is None:
        raise RuntimeError(
            "register_image2 with masks requires the elastix binary on PATH; "
            "SimpleITK does not pass -fMask/-mMask reliably."
        )

    cmd = [
        elastix,
        "-f",
        str(fixed_path),
        "-m",
        str(moving_path),
        "-out",
        str(out),
        "-p",
        str(affine_txt),
        "-p",
        str(bspline_txt),
    ]
    if fmask_path is not None:
        cmd.extend(["-fMask", str(fmask_path)])
    if mmask_path is not None:
        cmd.extend(["-mMask", str(mmask_path)])
    print(f"register_image2 (elastix CLI): {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"elastix failed (exit {proc.returncode}) for {out}\n"
            f"{proc.stdout}\n{proc.stderr}\n"
            f"{_elastix_log_excerpt(out)}"
        )

    tp_files = _copy_elastix_transform_pmaps(out)
    if not tp_files:
        raise FileNotFoundError(
            f"Elastix did not write TransformParameters.*.txt in {out}\n"
            f"{_elastix_log_excerpt(out)}"
        )

    result_path = _elastix_result_image_path(out)
    if result_path is not None:
        return sitk.GetArrayFromImage(sitk.ReadImage(str(result_path)))

    # Elastix 5.x may write result.N.nii or omit the volume; warp from transforms.
    return _warp_moving_with_saved_transforms(moving_path, out)


def trim_image(image, fixMax=False, ind=0):
    """
    uses opencv to get a point list and delete data outside roi

    :param image: as array
    :return: trimmed image array
    """
    if image.ndim == 3:
        image_slice = image[ind].copy()
    else:
        image_slice = image.copy()

    if fixMax:
        image_slice = image_slice / 2**12

    list_of_points = []

    def roi_grabber(event, x, y, flags, params):
        if event == 1:  # left click
            list_of_points.append((x, y))
        if event == 2:  # right click
            cv2.destroyAllWindows()

    cv2.namedWindow(f"roi_finding_window")
    cv2.setMouseCallback(f"roi_finding_window", roi_grabber)
    cv2.imshow(f"roi_finding_window", np.array(image_slice, "uint8"))
    try:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        cv2.destroyAllWindows()

    image_mask = np.zeros(image_slice.shape, dtype="int32")
    image_mask = cv2.fillPoly(image_mask, np.int32([list_of_points]), 1, 255)
    if image.ndim == 3:
        images = [np.ma.masked_where(image_mask != 1, i).filled(0) for i in image]
        return np.array(images)
    else:
        return np.ma.masked_where(image_mask != 1, image).filled(0)


def estimate_transform_itk(moving, fixed, tx):
    from SimpleITK import GetImageFromArray

    moving_ = GetImageFromArray(moving.astype("float32"))
    fixed_ = GetImageFromArray(fixed.astype("float32"))
    return tx.Execute(moving_, fixed_)


def calculate_match_value(image_reference, image_target):

    def_size = 1024
    while max(max(image_reference.shape, image_target.shape)) >= def_size:
        def_size *= 2
    image_target = embed_image(image_target, def_size)
    image_reference = embed_image(image_reference, def_size)

    reference_image = sitk.GetImageFromArray(image_reference)
    align_image = sitk.GetImageFromArray(image_target)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(reference_image)
    elastixImageFilter.SetMovingImage(align_image)

    param_map = sitk.GetDefaultParameterMap("rigid")
    param_map["MaximumNumberOfIterations"] = ["512"]
    elastixImageFilter.SetParameterMap(param_map)

    pmap = sitk.GetDefaultParameterMap("bspline")
    pmap["MaximumNumberOfIterations"] = ["128"]
    pmap["Metric0Weight"] = ["0.1"]
    pmap["Metric1Weight"] = ["20"]
    elastixImageFilter.AddParameterMap(pmap)

    elastixImageFilter.LogToConsoleOn()
    elastixImageFilter.Execute()
    res = elastixImageFilter.GetResultImage()

    r = sitk.ImageRegistrationMethod()
    r.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    r.SetOptimizerAsLBFGSB(maximumNumberOfCorrections=3, numberOfIterations=250)
    r.SetMetricSamplingStrategy(r.RANDOM)
    r.SetMetricSamplingPercentage(0.5)
    tx = sitk.TranslationTransform(2)

    r.SetInitialTransform(tx)
    r.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2])
    r.SetSmoothingSigmasPerLevel(smoothingSigmas=[3, 1])

    res_img = sitk.GetArrayFromImage(res)

    tx = estimate_transform_itk(image_reference, res_img, r)
    return r.GetMetricValue()


def register_image(
    image_reference, image_target, savepath=None, embed=False, scalePenalty=10
):
    """

    :param image_reference:
    :param image_target:
    :param savepath: directory
    :param embed: make embedding image optional
    :param scalePenalty: 10 - default, lower is squishier, higher is more rigid
    :return:
    """
    if embed:
        def_size = 1024
        while max(max(image_reference.shape, image_target.shape)) >= def_size:
            def_size *= 2
        image_target = embed_image(image_target, def_size)
        image_reference = embed_image(image_reference, def_size)

    reference_image = sitk.GetImageFromArray(image_reference)
    align_image = sitk.GetImageFromArray(image_target)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(reference_image)
    elastixImageFilter.SetMovingImage(align_image)

    pmap = sitk.GetDefaultParameterMap("rigid")
    pmap["MaximumNumberOfIterations"] = ["4096"]
    elastixImageFilter.SetParameterMap(pmap)

    pmap = sitk.GetDefaultParameterMap("bspline")
    pmap["MaximumNumberOfIterations"] = ["4096"]
    pmap["Metric0Weight"] = ["0.1"]
    pmap["Metric1Weight"] = [str(scalePenalty)]
    elastixImageFilter.AddParameterMap(pmap)

    elastixImageFilter.LogToConsoleOn()
    elastixImageFilter.Execute()
    res = elastixImageFilter.GetResultImage()

    if savepath:
        from pathlib import Path

        pmaps = elastixImageFilter.GetTransformParameterMap()

        for n, pmap in enumerate(pmaps):
            sitk.WriteParameterFile(
                pmap, Path(savepath).joinpath(f"transform_pmap_{n}.txt").as_posix()
            )

    return sitk.GetArrayFromImage(res)


def register_image2(
    image_reference,
    image_target,
    savepath=None,
    embed=False,
    scalePenalty=10,
    iterations=(512, 512),
    fixed_mask=None,
    moving_mask=None,
    use_elastix_cli=False,
):
    """
    Affine + B-spline registration via SimpleITK's Elastix wrapper.

    Prefer zeroing intensities outside each ROI with :func:`apply_registration_blackout`
    and leaving ``fixed_mask`` / ``moving_mask`` as ``None`` (default path).

    ``fixed_mask`` / ``moving_mask``: optional uint8 masks (1 = include in metric). Only
    used when non-``None``; with ``use_elastix_cli=True`` masks are sent via the
    standalone elastix binary (-fMask / -mMask).

    :return:
    """
    if embed:
        def_size = 1024
        while max(max(image_reference.shape, image_target.shape)) >= def_size:
            def_size *= 2
        image_target = embed_image(image_target, def_size)
        image_reference = embed_image(image_reference, def_size)

    reference_image = sitk.GetImageFromArray(np.asarray(image_reference, dtype=np.float32))
    align_image = sitk.GetImageFromArray(np.asarray(image_target, dtype=np.float32))

    fixed_mask_image = _attach_mask_to_image(fixed_mask, reference_image)
    moving_mask_image = _attach_mask_to_image(moving_mask, align_image)
    use_masks = fixed_mask_image is not None or moving_mask_image is not None

    if use_masks and savepath is None:
        raise ValueError(
            "register_image2 with fixed_mask or moving_mask requires savepath"
        )

    if use_masks and use_elastix_cli:
        elastix_bin = _find_elastix_executable()
        if elastix_bin is None:
            require_elastix_for_masks()
        print(
            f"register_image2: using elastix CLI for masks ({elastix_bin})",
            flush=True,
        )
        return _register_image2_elastix_cli(
            reference_image,
            align_image,
            fixed_mask_image,
            moving_mask_image,
            savepath,
            iterations,
            scalePenalty,
        )

    from pathlib import Path

    if savepath:
        Path(savepath).mkdir(parents=True, exist_ok=True)

    elastixImageFilter = sitk.ElastixImageFilter()
    if savepath:
        elastixImageFilter.SetOutputDirectory(str(Path(savepath)))
    elastixImageFilter.SetFixedImage(reference_image)
    elastixImageFilter.SetMovingImage(align_image)
    if fixed_mask_image is not None:
        elastixImageFilter.SetFixedMask(fixed_mask_image)
    if moving_mask_image is not None:
        elastixImageFilter.SetMovingMask(moving_mask_image)

    parameterMapVector = sitk.VectorOfParameterMap()
    pmap1 = sitk.GetDefaultParameterMap("affine")
    pmap1["MaximumNumberOfIterations"] = [str(iterations[0])]
    pmap1 = _configure_elastix_parameter_map_for_numpy_images(pmap1)
    parameterMapVector.append(pmap1)

    pmap2 = sitk.GetDefaultParameterMap("bspline")
    pmap2["MaximumNumberOfIterations"] = [str(iterations[1])]
    pmap2["Metric0Weight"] = ["0.1"]
    pmap2["Metric1Weight"] = [str(scalePenalty)]
    pmap2 = _configure_elastix_parameter_map_for_numpy_images(pmap2)
    parameterMapVector.append(pmap2)

    if use_masks:
        _configure_elastix_parameter_maps_for_masks(parameterMapVector)

    elastixImageFilter.LogToConsoleOn()
    elastixImageFilter.SetParameterMap(parameterMapVector)
    if savepath:
        if fixed_mask_image is not None:
            sitk.WriteImage(
                fixed_mask_image,
                str(Path(savepath) / "metric_fixed_mask.mha"),
            )
        if moving_mask_image is not None:
            sitk.WriteImage(
                moving_mask_image,
                str(Path(savepath) / "metric_moving_mask.mha"),
            )
    elastixImageFilter.Execute()
    res = elastixImageFilter.GetResultImage()

    if savepath:
        from pathlib import Path

        pmaps = elastixImageFilter.GetTransformParameterMap()

        for n, pmap in enumerate(pmaps):
            sitk.WriteParameterFile(
                pmap, Path(savepath).joinpath(f"transform_pmap_{n}.txt").as_posix()
            )

    return sitk.GetArrayFromImage(res)


def register_image_similarity_affine_bspline(
    image_reference,
    image_target,
    savepath=None,
    embed=False,
    scalePenalty=50,
    iterations_similarity=4096,
    iterations_affine=8000,
    iterations_bspline=500,
    smoothing_sigma_pixels=2.0,
):
    """
    Elastix pipeline: similarity → affine → B-spline.

    Use when ``register_image2`` (affine → B-spline only) leaves large scale / FOV mismatch,
    e.g. sparse functional planes vs a dense atlas: similarity explicitly estimates uniform scale.

    Parameters
    ----------
    smoothing_sigma_pixels : float
        Gaussian smoothing sigma (in pixels; image spacing is 1 from ``GetImageFromArray``).
        Set to 0 to disable. Mild smoothing often stabilizes mutual information for sparse moving images.
    """
    if embed:
        def_size = 1024
        while max(max(image_reference.shape, image_target.shape)) >= def_size:
            def_size *= 2
        image_target = embed_image(image_target, def_size)
        image_reference = embed_image(image_reference, def_size)

    ref_arr = np.asarray(image_reference, dtype=np.float32)
    mov_arr = np.asarray(image_target, dtype=np.float32)

    reference_image = sitk.GetImageFromArray(ref_arr)
    align_image = sitk.GetImageFromArray(mov_arr)

    if smoothing_sigma_pixels and float(smoothing_sigma_pixels) > 0:
        sigma = float(smoothing_sigma_pixels)
        reference_image = sitk.SmoothingRecursiveGaussian(reference_image, sigma)
        align_image = sitk.SmoothingRecursiveGaussian(align_image, sigma)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(reference_image)
    elastixImageFilter.SetMovingImage(align_image)

    parameterMapVector = sitk.VectorOfParameterMap()

    try:
        pmap_sim = sitk.GetDefaultParameterMap("similarity")
    except Exception:
        pmap_sim = None
    if pmap_sim is not None:
        pmap_sim["MaximumNumberOfIterations"] = [str(int(iterations_similarity))]
        parameterMapVector.append(pmap_sim)

    pmap_affine = sitk.GetDefaultParameterMap("affine")
    pmap_affine["MaximumNumberOfIterations"] = [str(int(iterations_affine))]
    parameterMapVector.append(pmap_affine)

    pmap_bspline = sitk.GetDefaultParameterMap("bspline")
    pmap_bspline["MaximumNumberOfIterations"] = [str(int(iterations_bspline))]
    pmap_bspline["Metric0Weight"] = ["0.1"]
    pmap_bspline["Metric1Weight"] = [str(scalePenalty)]
    parameterMapVector.append(pmap_bspline)

    elastixImageFilter.LogToConsoleOn()
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()
    res = elastixImageFilter.GetResultImage()

    if savepath:
        from pathlib import Path

        if not os.path.exists(savepath):
            os.mkdir(savepath)

        pmaps = elastixImageFilter.GetTransformParameterMap()

        for n, pmap in enumerate(pmaps):
            sitk.WriteParameterFile(
                pmap, Path(savepath).joinpath(f"transform_pmap_{n}.txt").as_posix()
            )

    return sitk.GetArrayFromImage(res)


def calculate_match_value2(image_reference, image_target):
    def_size = 1024
    while max(max(image_reference.shape, image_target.shape)) >= def_size:
        def_size *= 2
    image_target = embed_image(image_target, def_size)
    image_reference = embed_image(image_reference, def_size)

    res = register_image2(image_target, image_reference, iterations=(50, 100))
    r = sitk.ImageRegistrationMethod()
    r.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    r.SetOptimizerAsLBFGSB(maximumNumberOfCorrections=3, numberOfIterations=100)
    r.SetMetricSamplingStrategy(r.RANDOM)
    r.SetMetricSamplingPercentage(0.5)
    tx = sitk.TranslationTransform(2)
    r.SetInitialTransform(tx)
    r.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2])
    r.SetSmoothingSigmasPerLevel(smoothingSigmas=[3, 1])
    # res_img = sitk.GetArrayFromImage(res)
    tx = estimate_transform_itk(image_reference, res, r)
    return r.GetMetricValue()


def transform_points(
    folderpath: str, points: list, cleanup: bool = True, floating=False
) -> list:
    """
    transforms a list of points

    :param cleanup:
    :param folderpath:
    :param points:
    :return:
    """
    from pathlib import Path

    # write pts to file
    point_path = Path(folderpath).joinpath("point_set.txt")
    if os.path.exists(point_path):
        os.remove(point_path)

    filestream = open(point_path, "a")
    filestream.write("point")
    filestream.write("\n")
    filestream.write(f"{len(points)}")
    filestream.write("\n")

    for pt in points:
        filestream.write(f"{pt[0]} {pt[1]}")
        filestream.write("\n")

    filestream.flush()
    filestream.close()

    # load our saved parameter maps and build filter
    pmap_files = []
    with os.scandir(folderpath) as entries:
        for entry in entries:
            # Ignore macOS AppleDouble sidecar files (._*) and only load real pmaps.
            if "transform_pmap" in entry.name and not entry.name.startswith("._"):
                pmap_files.append(entry.path)
    pmap_files.sort()

    if len(pmap_files) == 0:
        raise FileNotFoundError(
            f"No transform_pmap files found in {folderpath}"
        )

    pmap0 = sitk.ReadParameterFile(pmap_files[0])

    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(pmap0)

    for pmap_file in pmap_files[1:]:
        pmap = sitk.ReadParameterFile(pmap_file)
        transformixImageFilter.AddTransformParameterMap(pmap)

    transformixImageFilter.SetFixedPointSetFileName(point_path.as_posix())
    transformixImageFilter.SetOutputDirectory(folderpath.as_posix())
    transformixImageFilter.Execute()

    output_pts_path = Path(folderpath).joinpath("outputpoints.txt")

    with open(output_pts_path) as file:
        contents = file.read()
    lines = contents.split("\n")
    coords = []
    if not floating:
        for line in lines:
            if line != "":
                x = int(line.split(";")[3].split("[ ")[1].split(" ]")[0].split(" ")[0])
                y = int(line.split(";")[3].split("[ ")[1].split(" ]")[0].split(" ")[1])
                coord = (x, y)
                coords.append(coord)
    else:
        for line in lines:
            if line != "":
                x = float(
                    line.split(";")[3].split("[ ")[1].split(" ]")[0].split(" ")[0]
                )
                y = float(
                    line.split(";")[3].split("[ ")[1].split(" ]")[0].split(" ")[1]
                )
                coord = (x, y)
                coords.append(coord)
    return coords


def piecewise_region_at_xy(x, y, bands):
    """
    Return the band name whose embedded row interval [row_start, row_end) contains y.

    ``bands`` entries need ``name``, ``row_start``, and ``row_end`` (experiment / plot_img frame).
    """
    py = float(y)
    for band in bands:
        if int(band["row_start"]) <= py < int(band["row_end"]):
            return str(band["name"])
    return None


def piecewise_inverse_transform_points(
    points,
    results,
    bands=None,
    floating=False,
):
    """
    Inverse-map experiment-frame (x, y) points to atlas coordinates using per-band Elastix folders.

    ``results`` maps band name -> dict with at least ``save_path`` (from piecewise registration).
  Optional ``bands`` defaults to inferring row ranges from ``results`` values.

    Returns a list parallel to ``points``; entries are (x_atlas, y_atlas) or None when y falls
    outside all bands or the band has no transform.
    """
    if not results:
        raise ValueError("results is empty — run piecewise registration first.")

    if bands is None:
        bands = [
            {
                "name": name,
                "row_start": int(res.get("row_start", 0)),
                "row_end": int(res.get("row_end", 10**9)),
            }
            for name, res in results.items()
        ]

    out = [None] * len(points)
    by_band = {}
    for i, pt in enumerate(points):
        x, y = float(pt[0]), float(pt[1])
        name = piecewise_region_at_xy(x, y, bands)
        if name is None or name not in results:
            continue
        by_band.setdefault(name, []).append((i, (x, y)))

    from pathlib import Path

    for name, idx_pts in by_band.items():
        save_path = Path(results[name]["save_path"])
        pts = [p for _, p in idx_pts]
        mapped = transform_points(save_path, pts, floating=floating)
        for (i, _), coord in zip(idx_pts, mapped):
            out[i] = coord
    return out


def piecewise_warped_row_montage(results, shape=None, fill=0.0):
    """
    QC-only montage in experiment coordinates: for each embedded row y, show the warp from the
    band that owns y.

    Each Elastix run still resamples the **entire** moving atlas; this does not clip the atlas
    before transforming. It only picks which band's full-field warp to display at each row.
    """
    if not results:
        raise ValueError("results is empty")
    first = next(iter(results.values()))
    warped0 = np.asarray(first["warped"])
    if shape is None:
        shape = warped0.shape[:2]
    out = np.full(shape, float(fill), dtype=np.float64)
    for res in results.values():
        rs, re = int(res["row_start"]), int(res["row_end"])
        w = np.asarray(res["warped"], dtype=np.float64)
        out[rs:re, :] = w[rs:re, :]
    return out


def piecewise_composite_warped(results, shape=None, fill=0.0):
    """Alias for :func:`piecewise_warped_row_montage` (QC montage, not input clipping)."""
    return piecewise_warped_row_montage(results, shape=shape, fill=fill)


def return_conv_pt(_y, _x, xform_path, size1=1024, size2=1024):
    test_image = np.zeros([size1, size2])

    circ_img = cv2.circle(test_image, (_x, _y), 15, 255, -1)
    xform_img = transform_image_from_saved(
        circ_img,
        xform_path,
    )

    xval = np.nanmean(np.where(xform_img == 255)[0], axis=0)
    yval = np.nanmean(np.where(xform_img == 255)[1], axis=0)
    return xval, yval


def embed_pt(pt, ydim, xdim, refdim):
    y = pt[0]
    x = pt[1]
    return y - ydim // 2 + refdim // 2, x - xdim // 2 + refdim // 2


def transform_image_from_saved(image, savepath):
    align_image = sitk.GetImageFromArray(image)

    pmap_files = []
    with os.scandir(savepath) as entries:
        for entry in entries:
            # Ignore macOS AppleDouble sidecar files (._*) and only load real pmaps.
            if "transform_pmap" in entry.name and not entry.name.startswith("._"):
                pmap_files.append(entry.path)
    pmap_files.sort()

    if len(pmap_files) == 0:
        raise FileNotFoundError(
            f"No transform_pmap files found in {savepath}"
        )

    pmap0 = sitk.ReadParameterFile(pmap_files[0])

    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(pmap0)

    for pmap_file in pmap_files[1:]:
        pmap = sitk.ReadParameterFile(pmap_file)
        transformixImageFilter.AddTransformParameterMap(pmap)

    transformixImageFilter.SetMovingImage(align_image)
    transformixImageFilter.Execute()
    res = transformixImageFilter.GetResultImage()

    return sitk.GetArrayFromImage(res)


def find_best_z_match(
    stack_reference, image_target, rigorous=False, l=None, r=None, check_distance=3
):
    """

    :param stack_reference: 3d image stack to align image target to
    :param image_target: target image array (2d)
    :param rigorous: if you have an abundance of time this can be true
    :param l:
    :param r:
    :return: Z index of reference stack and results dict
    """

    results_dictionary = {}

    if not l:
        l = 0
    if not r:
        r = len(stack_reference) - 1

    if not rigorous:
        while r - l > 1:

            if l not in results_dictionary.keys():
                try:
                    results_dictionary[l] = abs(
                        calculate_match_value(stack_reference[l], image_target)
                    )
                except:
                    results_dictionary[l] = 0

            if r not in results_dictionary.keys():
                try:
                    results_dictionary[r] = abs(
                        calculate_match_value(stack_reference[r], image_target)
                    )
                except:
                    results_dictionary[r] = 0

            midpt = ((r - l) // 2) + l

            while midpt in results_dictionary.keys():
                midpt += 1

                if midpt >= r:
                    break

            try:
                results_dictionary[midpt] = abs(
                    calculate_match_value(stack_reference[midpt], image_target)
                )
            except:
                results_dictionary[midpt] = 0

            if results_dictionary[r] > results_dictionary[l]:
                if results_dictionary[midpt] >= results_dictionary[l]:
                    l = midpt
                else:
                    break
            elif results_dictionary[l] >= results_dictionary[r]:
                if results_dictionary[midpt] >= results_dictionary[r]:
                    r = midpt
                else:
                    break

            print(l, r)

        maxval = max(results_dictionary.values())
        maxkey = {v: k for k, v in results_dictionary.items()}[maxval]
        for ind in np.arange(maxkey - check_distance, maxkey + check_distance):
            if ind not in results_dictionary.keys():
                results_dictionary[ind] = abs(
                    calculate_match_value(stack_reference[ind], image_target)
                )
        maxval = max(results_dictionary.values())
        maxkey = {v: k for k, v in results_dictionary.items()}[maxval]
        return maxkey, results_dictionary
    else:
        for i in np.arange(l, r):
            results_dictionary[i] = abs(
                calculate_match_value(stack_reference[i], image_target)
            )
        maxval = max(results_dictionary.values())
        maxkey = {v: k for k, v in results_dictionary.items()}[maxval]
        return maxkey, results_dictionary


def find_best_z_match2(
    stack_reference, image_target, rigorous=False, l=None, r=None, check_distance=3
):
    """
    :param stack_reference: 3d image stack to align image target to
    :param image_target: target image array (2d)
    :param rigorous: if you have an abundance of time this can be true
    :param l:
    :param r:
    :return: Z index of reference stack and results dict
    """

    results_dictionary = {}

    if not l:
        l = 0
    if not r:
        r = len(stack_reference) - 1

    if not rigorous:
        while r - l > 1:

            if l not in results_dictionary.keys():
                try:
                    results_dictionary[l] = abs(
                        calculate_match_value2(stack_reference[l], image_target)
                    )
                except:
                    results_dictionary[l] = 0

            if r not in results_dictionary.keys():
                try:
                    results_dictionary[r] = abs(
                        calculate_match_value2(stack_reference[r], image_target)
                    )
                except:
                    results_dictionary[r] = 0

            midpt = ((r - l) // 2) + l

            while midpt in results_dictionary.keys():
                midpt += 1

                if midpt >= r:
                    break

            try:
                results_dictionary[midpt] = abs(
                    calculate_match_value2(stack_reference[midpt], image_target)
                )
            except:
                results_dictionary[midpt] = 0

            if results_dictionary[r] > results_dictionary[l]:
                if results_dictionary[midpt] >= results_dictionary[l]:
                    l = midpt
                else:
                    break
            elif results_dictionary[l] >= results_dictionary[r]:
                if results_dictionary[midpt] >= results_dictionary[r]:
                    r = midpt
                else:
                    break

            print(l, r)

        maxval = max(results_dictionary.values())
        maxkey = {v: k for k, v in results_dictionary.items()}[maxval]
        for ind in np.arange(maxkey - check_distance, maxkey + check_distance):
            if ind not in results_dictionary.keys():
                results_dictionary[ind] = abs(
                    calculate_match_value2(stack_reference[ind], image_target)
                )
        maxval = max(results_dictionary.values())
        maxkey = {v: k for k, v in results_dictionary.items()}[maxval]
        return maxkey, results_dictionary
    else:
        for i in np.arange(l, r):
            results_dictionary[i] = abs(
                calculate_match_value2(stack_reference[i], image_target)
            )
        maxval = max(results_dictionary.values())
        maxkey = {v: k for k, v in results_dictionary.items()}[maxval]
        return maxkey, results_dictionary


# --- 3D functional stack ↔ mapZebrain atlas (sparse z, native in-plane geometry) ---

FUNC_SPACING_UM = (0.783202792352194, 0.783202792352194, 20.0)  # (x, y, z) microns
ATLAS_SPACING_UM = (1.0, 1.0, 1.0)


def discover_std_plane_tiffs(mcorr_dir, plane_indices=None):
    """
    Find motion-corrected STD TIFFs under ``mcorr_tiff_outputs``.

    Returns a list of ``(plane_index, path)`` sorted by increasing plane index.
    If ``plane_indices`` is set, only those planes are kept (raises if any are missing).
    """
    mcorr_dir = Path(mcorr_dir)
    if not mcorr_dir.is_dir():
        raise FileNotFoundError(f"Missing mcorr output folder: {mcorr_dir}")
    plane_re = re.compile(r"plane(\d+)", re.IGNORECASE)
    by_plane = {}
    for path in sorted(mcorr_dir.glob("STD*")):
        if not path.is_file():
            continue
        m = plane_re.search(path.name)
        if m is None:
            continue
        pl = int(m.group(1))
        if pl in by_plane:
            raise ValueError(
                f"Multiple STD* files match plane{pl}: {by_plane[pl]!s} and {path!s}"
            )
        by_plane[pl] = path
    if not by_plane:
        names = sorted(p.name for p in mcorr_dir.glob("STD*") if p.is_file())
        raise FileNotFoundError(
            f"No STD* files with plane index in {mcorr_dir}. Found: {names}"
        )
    if plane_indices is None:
        return sorted(by_plane.items())
    out = []
    for pl in plane_indices:
        pl = int(pl)
        if pl not in by_plane:
            raise FileNotFoundError(
                f"No STD* file for plane {pl} in {mcorr_dir}. "
                f"Available planes: {sorted(by_plane)}"
            )
        out.append((pl, by_plane[pl]))
    return out


def native_xy_to_rot90_xy(col, row, native_shape_hw, rot90_k=1):
    """
    Map ``(col, row)`` on the raw STD (x, y) to indices after ``np.rot90(slice, k=rot90_k)``.

    Common cases (``imshow``, origin upper):
    - ``k=1`` (CCW): legacy ``brain_alignment_fish07`` ``plot_img`` — native anterior-left → bottom.
    - ``k=-1`` / ``k=3`` (CW): native anterior-left → **top**, matching mapZebrain dorsal/anterior-up.
    """
    h, w = int(native_shape_hw[0]), int(native_shape_hw[1])
    col = float(col)
    row = float(row)
    k = int(rot90_k) % 4
    if k == 0:
        return col, row
    if k == 1:
        return row, float(w - 1 - col)
    if k == 2:
        return float(w - 1 - col), float(h - 1 - row)
    # k == 3 (-1 mod 4): 90° clockwise
    return float(h - 1 - row), col


def native_xy_to_rot90_k1_xy(col, row, native_shape_hw):
    """Backward-compatible alias for :func:`native_xy_to_rot90_xy` with ``k=1``."""
    return native_xy_to_rot90_xy(col, row, native_shape_hw, rot90_k=1)


def load_std_functional_volume(
    plane_tiff_items,
    scale_max=True,
    intensity_scale=2**10,
    rot90_k=1,
):
    """
    Stack native STD planes into ``(n_z, y, x)`` float32 (no 1024² embed).

    ``rot90_k``: if non-zero, apply ``np.rot90(slice, k=rot90_k)`` per plane before stacking
    (default ``1`` = legacy fish07 ``plot_img``). Slices are forced C-contiguous for SimpleITK.

    Returns ``func_np``, ``plane_order``, and ``native_plane_shape`` ``(H, W)`` before rotation.
    """
    slices = []
    plane_order = []
    native_plane_shape = None
    for pl, path in plane_tiff_items:
        arr = np.asarray(load_image(path), dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D STD from {path}, got shape {arr.shape}")
        if native_plane_shape is None:
            native_plane_shape = tuple(int(x) for x in arr.shape)
        elif tuple(arr.shape) != native_plane_shape:
            raise ValueError(
                f"Inconsistent STD plane shapes: {native_plane_shape} vs {arr.shape} ({path})"
            )
        if scale_max:
            mx = float(np.nanmax(arr))
            if mx > 0:
                arr = arr / mx
            if intensity_scale:
                arr = arr * float(intensity_scale)
        if rot90_k != 0:
            arr = np.ascontiguousarray(np.rot90(arr, k=int(rot90_k)))
        slices.append(arr)
        plane_order.append(int(pl))
    if not slices:
        raise ValueError("plane_tiff_items is empty")
    return np.stack(slices, axis=0), plane_order, native_plane_shape


def sitk_volume_from_numpy_zyx(arr_zyx, spacing_xyz):
    """``tifffile``/NumPy ``(z, y, x)`` → SimpleITK image with ``spacing_xyz``."""
    img = sitk.GetImageFromArray(np.asarray(arr_zyx, dtype=np.float32))
    img.SetSpacing(tuple(float(s) for s in spacing_xyz))
    return img


def sitk_image_from_numpy_yx(arr_yx, spacing_xy):
    """NumPy ``(y, x)`` → 2D SimpleITK image with ``spacing_xy`` = ``(x, y)`` µm."""
    img = sitk.GetImageFromArray(np.asarray(arr_yx, dtype=np.float32))
    img.SetSpacing(tuple(float(s) for s in spacing_xy))
    return img


def _slice_registration_transform_to_affine2d(transform):
    """
    Approximate the registration output (Affine or Composite) as one 2D affine
    by sampling three physical points (func → atlas).
    """
    p0 = np.array(transform.TransformPoint((0.0, 0.0)), dtype=float)
    px = np.array(transform.TransformPoint((1.0, 0.0)), dtype=float) - p0
    py = np.array(transform.TransformPoint((0.0, 1.0)), dtype=float) - p0
    affine = sitk.AffineTransform(2)
    affine.SetMatrix(
        (
            float(px[0]),
            float(py[0]),
            float(px[1]),
            float(py[1]),
        )
    )
    affine.SetTranslation((float(p0[0]), float(p0[1])))
    return affine


def affine_2d_func_to_atlas_as_3d(
    transform_2d,
    func_spacing=FUNC_SPACING_UM,
    atlas_spacing=ATLAS_SPACING_UM,
    func_stack_z_index=0,
    atlas_z_index=0,
):
    """
    Lift a 2D affine (functional physical → atlas physical, from slice registration)
    into a 3D affine for a single functional z-slice at ``func_stack_z_index``.
    """
    transform_2d = _slice_registration_transform_to_affine2d(transform_2d)
    m2 = np.array(transform_2d.GetMatrix(), dtype=float).reshape(2, 2)
    t2 = np.array(transform_2d.GetTranslation(), dtype=float)
    m3 = np.eye(3, dtype=float)
    m3[0:2, 0:2] = m2
    t3 = sitk.AffineTransform(3)
    t3.SetMatrix(tuple(float(x) for x in m3.reshape(-1)))
    func_z_phys = float(func_stack_z_index) * float(func_spacing[2])
    atlas_z_phys = float(atlas_z_index) * float(atlas_spacing[2])
    t3.SetTranslation(
        (
            float(t2[0]),
            float(t2[1]),
            float(atlas_z_phys - func_z_phys),
        )
    )
    return t3


def atlas_z_at_atlas_y(
    atlas_y,
    atlas_z_top,
    atlas_z_bottom,
    atlas_row_y_top=None,
    atlas_row_y_bottom=None,
    n_atlas_rows=None,
    extrapolate=True,
):
    """
    Map atlas **y** index (row in ``imshow`` on a z-slice) → atlas **z** along the oblique cut.

    ``atlas_row_y_top`` / ``atlas_row_y_bottom`` are the dorsal/ventral tissue bounds
    read from the reference planes at ``atlas_z_top`` and ``atlas_z_bottom`` (§2a).
    The experiment plane row does not affect depth.

    ``extrapolate``: when ``True`` (default) the linear y→z fit continues past the
    ``[y_top, y_bottom]`` band so the reslice plane stays a single flat (tilted)
    plane through the whole volume — z keeps dipping/rising beyond the selected
    region (and may fall outside ``[0, nz)``, which samples as background). When
    ``False``, values outside the band are clamped to ``atlas_z_top`` /
    ``atlas_z_bottom`` (legacy plateau behaviour).
    """
    if n_atlas_rows is not None:
        n_atlas_rows = int(n_atlas_rows)
        y_top = 0 if atlas_row_y_top is None else int(atlas_row_y_top)
        y_bot = (n_atlas_rows - 1) if atlas_row_y_bottom is None else int(atlas_row_y_bottom)
    else:
        if atlas_row_y_top is None or atlas_row_y_bottom is None:
            raise ValueError("Provide atlas_row_y_top and atlas_row_y_bottom, or n_atlas_rows")
        y_top = int(atlas_row_y_top)
        y_bot = int(atlas_row_y_bottom)
    if y_bot <= y_top:
        z0, z1 = float(atlas_z_top), float(atlas_z_bottom)
        out = np.asarray(atlas_y, dtype=np.float64)
        return np.full_like(out, z0) if np.ndim(out) else z0
    y = np.asarray(atlas_y, dtype=np.float64)
    t = (y - float(y_top)) / float(y_bot - y_top)
    if not extrapolate:
        t = np.clip(t, 0.0, 1.0)
    return float(atlas_z_top) + t * (float(atlas_z_bottom) - float(atlas_z_top))


def atlas_z_at_func_row(row, n_rows, atlas_z_top, atlas_z_bottom, func_row_top=None, func_row_bottom=None):
    """Deprecated: use :func:`atlas_z_at_atlas_y` (depth from atlas y, not experiment row)."""
    return atlas_z_at_atlas_y(
        float(row),
        atlas_z_top,
        atlas_z_bottom,
        atlas_row_y_top=func_row_top,
        atlas_row_y_bottom=func_row_bottom,
        n_atlas_rows=n_rows,
    )


def _transform_to_affine3d_matrix(transform):
    """Approximate registration output as 3×3 + offset (functional physical → atlas physical)."""
    p0 = np.array(transform.TransformPoint((0.0, 0.0, 0.0)), dtype=float)
    px = np.array(transform.TransformPoint((1.0, 0.0, 0.0)), dtype=float) - p0
    py = np.array(transform.TransformPoint((0.0, 1.0, 0.0)), dtype=float) - p0
    pz = np.array(transform.TransformPoint((0.0, 0.0, 1.0)), dtype=float) - p0
    return np.column_stack([px, py, pz]), p0


def _trilinear_sample_zyx_batch(volume_zyx, cz, cy, cx):
    """Trilinear sample ``volume_zyx`` at continuous indices (z, y, x)."""
    vol = np.asarray(volume_zyx, dtype=np.float32)
    nz, ny, nx = vol.shape
    cz = np.asarray(cz, dtype=np.float64)
    cy = np.asarray(cy, dtype=np.float64)
    cx = np.asarray(cx, dtype=np.float64)

    z0 = np.clip(np.floor(cz).astype(np.int64), 0, nz - 1)
    y0 = np.clip(np.floor(cy).astype(np.int64), 0, ny - 1)
    x0 = np.clip(np.floor(cx).astype(np.int64), 0, nx - 1)
    z1 = np.clip(z0 + 1, 0, nz - 1)
    y1 = np.clip(y0 + 1, 0, ny - 1)
    x1 = np.clip(x0 + 1, 0, nx - 1)

    zd = np.clip(cz - z0, 0.0, 1.0)
    yd = np.clip(cy - y0, 0.0, 1.0)
    xd = np.clip(cx - x0, 0.0, 1.0)

    c000 = vol[z0, y0, x0]
    c001 = vol[z0, y0, x1]
    c010 = vol[z0, y1, x0]
    c011 = vol[z0, y1, x1]
    c100 = vol[z1, y0, x0]
    c101 = vol[z1, y0, x1]
    c110 = vol[z1, y1, x0]
    c111 = vol[z1, y1, x1]

    c00 = c000 * (1.0 - xd) + c001 * xd
    c01 = c010 * (1.0 - xd) + c011 * xd
    c10 = c100 * (1.0 - xd) + c101 * xd
    c11 = c110 * (1.0 - xd) + c111 * xd
    c0 = c00 * (1.0 - yd) + c01 * yd
    c1 = c10 * (1.0 - yd) + c11 * yd
    return (c0 * (1.0 - zd) + c1 * zd).astype(np.float32)


def reslice_atlas_oblique_onto_func_plane(
    atlas_np_zyx,
    func_img,
    transform_func_to_atlas,
    atlas_z_top,
    atlas_z_bottom,
    atlas_row_y_top=None,
    atlas_row_y_bottom=None,
    func_plane_index=0,
    func_spacing=FUNC_SPACING_UM,
    atlas_spacing=ATLAS_SPACING_UM,
    extrapolate=True,
):
    """
    Build the atlas on the functional grid with **prescribed oblique depth**.

    In-plane (x, y) from the registration transform. Atlas **z** from **atlas y**
    (``atlas_row_y_top`` → ``atlas_z_top``, ``atlas_row_y_bottom`` → ``atlas_z_bottom``),
    not from experiment-plane row. Unlike ``sitk.Resample`` (~one atlas z for all rows).

    ``extrapolate`` (default ``True``) keeps the y→z fit linear past the selected
    band so the cut stays one flat tilted plane through the volume (see
    :func:`atlas_z_at_atlas_y`).
    """
    atlas = np.ascontiguousarray(np.asarray(atlas_np_zyx, dtype=np.float32))
    a_mat, a_off = _transform_to_affine3d_matrix(transform_func_to_atlas)
    sx, sy, sz = (float(func_spacing[0]), float(func_spacing[1]), float(func_spacing[2]))
    ax_s, ay_s, az_s = (
        float(atlas_spacing[0]),
        float(atlas_spacing[1]),
        float(atlas_spacing[2]),
    )

    func_arr = sitk.GetArrayFromImage(func_img)
    h, w = int(func_arr.shape[-2]), int(func_arr.shape[-1])
    n_atlas_y = int(atlas.shape[1])

    rows = np.arange(h, dtype=np.float64)
    cols = np.arange(w, dtype=np.float64)
    cc, rr = np.meshgrid(cols, rows)
    pf = np.stack(
        [cc * sx, rr * sy, np.full_like(cc, float(func_plane_index) * sz)],
        axis=-1,
    )
    ap = pf @ a_mat.T + a_off

    cy = ap[..., 1] / ay_s
    cx = ap[..., 0] / ax_s
    cz = atlas_z_at_atlas_y(
        cy,
        atlas_z_top,
        atlas_z_bottom,
        atlas_row_y_top=atlas_row_y_top,
        atlas_row_y_bottom=atlas_row_y_bottom,
        n_atlas_rows=n_atlas_y,
        extrapolate=extrapolate,
    )

    try:
        from scipy.ndimage import map_coordinates

        sampled = map_coordinates(
            atlas,
            np.stack([cz, cy, cx]),
            order=1,
            mode="constant",
            cval=0.0,
        )
    except ImportError:
        sampled = _trilinear_sample_zyx_batch(atlas, cz, cy, cx)
    return np.asarray(sampled, dtype=np.float32)


def _crop_atlas_slab_image(atlas_np_zyx, z_lo, z_hi, atlas_spacing=ATLAS_SPACING_UM):
    """Atlas sub-volume with z origin set so physical coords match the full stack."""
    z_lo = int(z_lo)
    z_hi = int(z_hi)
    slab = np.ascontiguousarray(
        np.asarray(atlas_np_zyx[z_lo : z_hi + 1], dtype=np.float32)
    )
    img = sitk_volume_from_numpy_zyx(slab, atlas_spacing)
    origin = list(img.GetOrigin())
    origin[2] = float(z_lo) * float(atlas_spacing[2])
    img.SetOrigin(tuple(origin))
    return img, (z_lo, z_hi)


def register_functional_oblique_plane_to_mapzebrain(
    atlas_np_zyx,
    plane_tiff_items,
    atlas_z_top,
    atlas_z_bottom,
    atlas_row_y_top=None,
    atlas_row_y_bottom=None,
    rot90_k=1,
    func_spacing=FUNC_SPACING_UM,
    atlas_spacing=ATLAS_SPACING_UM,
    initial_transform=None,
    save_transform_path=None,
    number_of_histogram_bins=50,
    metric_sampling_percentage=0.25,
    learning_rate=1.0,
    number_of_iterations=300,
    min_step=1e-4,
    gradient_magnitude_tolerance=1e-6,
    relaxation_factor=0.5,
    shrink_factors=(4, 2, 1),
    smoothing_sigmas=(2, 1, 0),
    normalize=True,
):
    """
    **Oblique slice-to-volume:** one tilted STD plane vs an atlas z-slab.

    Optimizer (``RegularStepGradientDescent``) stop conditions — first match wins:

    - ``gradient_magnitude_tolerance``: stop when gradient norm is below this
    - ``min_step``: stop when the step length shrinks below this
    - ``number_of_iterations``: hard cap (your "Maximum number of iterations exceeded" message)

    The experiment plane cuts through atlas z = ``atlas_z_top`` (image top) to
    ``atlas_z_bottom`` (image bottom). Registration uses a **3D affine** (fixed =
    1-slice functional stack, moving = cropped atlas slab) so in-plane pose and
    out-of-plane tilt are estimated together. ROI mapping uses the returned
    transform against the **full** atlas volume.

    ``plane_tiff_items`` must contain exactly one ``(plane_number, path)`` pair.
    """
    if len(plane_tiff_items) != 1:
        raise ValueError(
            f"Oblique plane registration expects one plane, got {len(plane_tiff_items)}."
        )
    plane_number, _path = plane_tiff_items[0]
    func_np_zyx, plane_order, native_plane_shape = load_std_functional_volume(
        plane_tiff_items,
        rot90_k=rot90_k,
    )
    if func_np_zyx.shape[0] != 1:
        raise ValueError(f"Expected one z-slice, got shape {func_np_zyx.shape}")

    z_lo = int(min(atlas_z_top, atlas_z_bottom))
    z_hi = int(max(atlas_z_top, atlas_z_bottom))
    nz_atlas = int(atlas_np_zyx.shape[0])
    if z_lo < 0 or z_hi >= nz_atlas:
        raise ValueError(
            f"atlas z range [{z_lo}, {z_hi}] out of bounds for atlas nz={nz_atlas}"
        )

    func_np_zyx = np.ascontiguousarray(func_np_zyx, dtype=np.float32)
    func = sitk_volume_from_numpy_zyx(func_np_zyx, func_spacing)
    atlas_slab_img, (z_lo, z_hi) = _crop_atlas_slab_image(
        atlas_np_zyx, z_lo, z_hi, atlas_spacing
    )
    atlas_full = sitk_volume_from_numpy_zyx(
        np.ascontiguousarray(np.asarray(atlas_np_zyx, dtype=np.float32)),
        atlas_spacing,
    )

    if normalize:
        func_reg = sitk.Normalize(func)
        atlas_reg = sitk.Normalize(atlas_slab_img)
    else:
        func_reg = func
        atlas_reg = atlas_slab_img

    if initial_transform is None:
        initial_transform = sitk.CenteredTransformInitializer(
            func_reg,
            atlas_reg,
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

    shrink_levels = [1]
    smooth_levels = [0.0]

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=int(number_of_histogram_bins))
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(float(metric_sampling_percentage))
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=float(learning_rate),
        minStep=float(min_step),
        numberOfIterations=int(number_of_iterations),
        relaxationFactor=float(relaxation_factor),
        gradientMagnitudeTolerance=float(gradient_magnitude_tolerance),
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel(shrink_levels)
    reg.SetSmoothingSigmasPerLevel(smooth_levels)
    reg.SetInitialTransform(initial_transform, inPlace=False)

    transform = reg.Execute(func_reg, atlas_reg)
    atlas_on_func_affine = sitk.GetArrayFromImage(
        sitk.Resample(
            atlas_full,
            func,
            transform,
            sitk.sitkLinear,
            0.0,
            atlas_full.GetPixelID(),
        )
    )
    atlas_oblique_2d = reslice_atlas_oblique_onto_func_plane(
        atlas_np_zyx,
        func,
        transform,
        atlas_z_top,
        atlas_z_bottom,
        atlas_row_y_top=atlas_row_y_top,
        atlas_row_y_bottom=atlas_row_y_bottom,
        func_plane_index=0,
        func_spacing=func_spacing,
        atlas_spacing=atlas_spacing,
    )
    atlas_on_func = atlas_oblique_2d[np.newaxis, ...]

    h_func = int(func_np_zyx.shape[1])
    _z_affine_rows = []
    for _r in (0, h_func // 2, h_func - 1):
        _fp = func.TransformContinuousIndexToPhysicalPoint((0.0, float(_r), 0.0))
        _ci = atlas_full.TransformPhysicalPointToContinuousIndex(
            transform.TransformPoint(_fp)
        )
        _z_affine_rows.append(float(_ci[2]))

    if save_transform_path is not None:
        sitk.WriteTransform(transform, str(save_transform_path))

    info = {
        "mode": "oblique_slice_to_volume",
        "acquisition_plane": int(plane_number),
        "atlas_z_top": int(atlas_z_top),
        "atlas_z_bottom": int(atlas_z_bottom),
        "atlas_row_y_top": None if atlas_row_y_top is None else int(atlas_row_y_top),
        "atlas_row_y_bottom": None if atlas_row_y_bottom is None else int(atlas_row_y_bottom),
        "atlas_z_slab": (z_lo, z_hi),
        "atlas_resliced_method": "oblique_row_z",
        "atlas_z_axis_aligned_at_rows_top_mid_bot": tuple(_z_affine_rows),
        "stop": reg.GetOptimizerStopConditionDescription(),
        "metric": reg.GetMetricValue(),
        "func_spacing": tuple(func_spacing),
        "atlas_spacing": tuple(atlas_spacing),
        "func_shape_zyx": tuple(int(x) for x in func_np_zyx.shape),
        "rot90_k": int(rot90_k),
        "shrink_factors_per_level": shrink_levels,
        "smoothing_sigmas_per_level": smooth_levels,
    }
    return (
        transform,
        atlas_on_func,
        func_np_zyx,
        plane_order,
        native_plane_shape,
        info,
    )


def register_func_plane_array_oblique_to_mapzebrain(
    atlas_np_zyx,
    func_plane_2d,
    atlas_z_top,
    atlas_z_bottom,
    atlas_row_y_top=None,
    atlas_row_y_bottom=None,
    func_spacing=FUNC_SPACING_UM,
    atlas_spacing=ATLAS_SPACING_UM,
    initial_transform=None,
    save_transform_path=None,
    number_of_histogram_bins=50,
    metric_sampling_percentage=0.25,
    learning_rate=1.0,
    number_of_iterations=300,
    min_step=1e-4,
    gradient_magnitude_tolerance=1e-6,
    relaxation_factor=0.5,
    normalize=True,
):
    """
    **Oblique slice-to-volume from an in-memory plane** (no STD TIFF reload).

    Identical registration math to
    :func:`register_functional_oblique_plane_to_mapzebrain`, but the functional
    plane is supplied directly as a 2D array (``func_plane_2d``, already rotated /
    scaled exactly as it will be displayed). Use this when the STD projection comes
    from a mesmerize CaImAn memmap (``np.nanstd`` over time) rather than a per-plane
    ``STD*`` TIFF.

    The plane cuts atlas z = ``atlas_z_top`` (image top) → ``atlas_z_bottom``
    (image bottom). A 3D affine (fixed = 1-slice functional stack, moving = cropped
    atlas z-slab) estimates in-plane pose + out-of-plane tilt together; the returned
    transform maps against the **full** atlas volume.

    Returns ``transform, atlas_on_func, info`` where ``atlas_on_func`` is the oblique
    resliced atlas on the functional grid with a leading singleton z axis ``(1, y, x)``.
    """
    func_plane_2d = np.asarray(func_plane_2d, dtype=np.float32)
    if func_plane_2d.ndim != 2:
        raise ValueError(
            f"func_plane_2d must be a 2D array, got shape {func_plane_2d.shape}"
        )
    func_np_zyx = np.ascontiguousarray(func_plane_2d[np.newaxis, ...], dtype=np.float32)

    z_lo = int(min(atlas_z_top, atlas_z_bottom))
    z_hi = int(max(atlas_z_top, atlas_z_bottom))
    nz_atlas = int(atlas_np_zyx.shape[0])
    if z_lo < 0 or z_hi >= nz_atlas:
        raise ValueError(
            f"atlas z range [{z_lo}, {z_hi}] out of bounds for atlas nz={nz_atlas}"
        )

    func = sitk_volume_from_numpy_zyx(func_np_zyx, func_spacing)
    atlas_slab_img, (z_lo, z_hi) = _crop_atlas_slab_image(
        atlas_np_zyx, z_lo, z_hi, atlas_spacing
    )
    atlas_full = sitk_volume_from_numpy_zyx(
        np.ascontiguousarray(np.asarray(atlas_np_zyx, dtype=np.float32)),
        atlas_spacing,
    )

    if normalize:
        func_reg = sitk.Normalize(func)
        atlas_reg = sitk.Normalize(atlas_slab_img)
    else:
        func_reg = func
        atlas_reg = atlas_slab_img

    if initial_transform is None:
        initial_transform = sitk.CenteredTransformInitializer(
            func_reg,
            atlas_reg,
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

    shrink_levels = [1]
    smooth_levels = [0.0]

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=int(number_of_histogram_bins))
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(float(metric_sampling_percentage))
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=float(learning_rate),
        minStep=float(min_step),
        numberOfIterations=int(number_of_iterations),
        relaxationFactor=float(relaxation_factor),
        gradientMagnitudeTolerance=float(gradient_magnitude_tolerance),
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel(shrink_levels)
    reg.SetSmoothingSigmasPerLevel(smooth_levels)
    reg.SetInitialTransform(initial_transform, inPlace=False)

    transform = reg.Execute(func_reg, atlas_reg)
    atlas_oblique_2d = reslice_atlas_oblique_onto_func_plane(
        atlas_np_zyx,
        func,
        transform,
        atlas_z_top,
        atlas_z_bottom,
        atlas_row_y_top=atlas_row_y_top,
        atlas_row_y_bottom=atlas_row_y_bottom,
        func_plane_index=0,
        func_spacing=func_spacing,
        atlas_spacing=atlas_spacing,
    )
    atlas_on_func = atlas_oblique_2d[np.newaxis, ...]

    h_func = int(func_np_zyx.shape[1])
    _z_affine_rows = []
    for _r in (0, h_func // 2, h_func - 1):
        _fp = func.TransformContinuousIndexToPhysicalPoint((0.0, float(_r), 0.0))
        _ci = atlas_full.TransformPhysicalPointToContinuousIndex(
            transform.TransformPoint(_fp)
        )
        _z_affine_rows.append(float(_ci[2]))

    if save_transform_path is not None:
        sitk.WriteTransform(transform, str(save_transform_path))

    info = {
        "mode": "oblique_slice_to_volume_array",
        "atlas_z_top": int(atlas_z_top),
        "atlas_z_bottom": int(atlas_z_bottom),
        "atlas_row_y_top": None if atlas_row_y_top is None else int(atlas_row_y_top),
        "atlas_row_y_bottom": None if atlas_row_y_bottom is None else int(atlas_row_y_bottom),
        "atlas_z_slab": (z_lo, z_hi),
        "atlas_resliced_method": "oblique_row_z",
        "atlas_z_axis_aligned_at_rows_top_mid_bot": tuple(_z_affine_rows),
        "stop": reg.GetOptimizerStopConditionDescription(),
        "metric": reg.GetMetricValue(),
        "func_spacing": tuple(func_spacing),
        "atlas_spacing": tuple(atlas_spacing),
        "func_shape_zyx": tuple(int(x) for x in func_np_zyx.shape),
    }
    return transform, atlas_on_func, info


def register_functional_slice_to_mapzebrain(
    atlas_np_zyx,
    plane_tiff_items,
    atlas_z,
    rot90_k=1,
    func_spacing=FUNC_SPACING_UM,
    atlas_spacing=ATLAS_SPACING_UM,
    initial_transform=None,
    save_transform_path=None,
    number_of_histogram_bins=50,
    metric_sampling_percentage=0.25,
    learning_rate=1.0,
    number_of_iterations=300,
    shrink_factors=(4, 2, 1),
    smoothing_sigmas=(2, 1, 0),
    normalize=True,
):
    """
    **Slice-to-volume (single plane):** 2D affine of one STD plane to one atlas z-slice.

    Fixed = functional plane (after ``rot90_k``), moving = ``atlas_np_zyx[atlas_z]``.
    Returns a **3D** transform (in-plane warp + fixed atlas depth) compatible with
    :func:`func_index_to_atlas_continuous_index`.

    ``plane_tiff_items`` must contain exactly one ``(plane_number, path)`` pair.
    """
    if len(plane_tiff_items) != 1:
        raise ValueError(
            f"Slice registration expects one plane, got {len(plane_tiff_items)}. "
            "Use register_functional_volume_to_mapzebrain for multi-plane stacks."
        )
    plane_number, _path = plane_tiff_items[0]
    func_np_zyx, plane_order, native_plane_shape = load_std_functional_volume(
        plane_tiff_items,
        rot90_k=rot90_k,
    )
    if func_np_zyx.shape[0] != 1:
        raise ValueError(f"Expected one z-slice, got shape {func_np_zyx.shape}")

    atlas_z = int(atlas_z)
    nz_atlas = int(atlas_np_zyx.shape[0])
    if not (0 <= atlas_z < nz_atlas):
        raise ValueError(f"atlas_z={atlas_z} out of range for atlas with {nz_atlas} slices")

    func_2d = np.ascontiguousarray(func_np_zyx[0], dtype=np.float32)
    atlas_2d = np.ascontiguousarray(
        np.asarray(atlas_np_zyx[atlas_z], dtype=np.float32),
        dtype=np.float32,
    )
    func_img_2d = sitk_image_from_numpy_yx(func_2d, func_spacing[:2])
    atlas_img_2d = sitk_image_from_numpy_yx(atlas_2d, atlas_spacing[:2])

    if normalize:
        func_reg = sitk.Normalize(func_img_2d)
        atlas_reg = sitk.Normalize(atlas_img_2d)
    else:
        func_reg = func_img_2d
        atlas_reg = atlas_img_2d

    if initial_transform is None:
        initial_transform = sitk.CenteredTransformInitializer(
            func_reg,
            atlas_reg,
            sitk.AffineTransform(2),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
    elif not isinstance(initial_transform, sitk.Transform):
        initial_transform = sitk.AffineTransform(2)

    shrink_levels = [int(s) for s in shrink_factors]
    smooth_levels = [float(s) for s in smoothing_sigmas[: len(shrink_levels)]]
    while len(smooth_levels) < len(shrink_levels):
        smooth_levels.append(0.0)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=int(number_of_histogram_bins))
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(float(metric_sampling_percentage))
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=float(learning_rate),
        minStep=1e-4,
        numberOfIterations=int(number_of_iterations),
        gradientMagnitudeTolerance=1e-6,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel(shrink_levels)
    reg.SetSmoothingSigmasPerLevel(smooth_levels)
    reg.SetInitialTransform(initial_transform, inPlace=False)

    transform_2d = reg.Execute(func_reg, atlas_reg)
    transform_3d = affine_2d_func_to_atlas_as_3d(
        transform_2d,
        func_spacing=func_spacing,
        atlas_spacing=atlas_spacing,
        func_stack_z_index=0,
        atlas_z_index=atlas_z,
    )

    func_img = sitk_volume_from_numpy_zyx(func_np_zyx, func_spacing)
    atlas_img = sitk_volume_from_numpy_zyx(
        np.ascontiguousarray(np.asarray(atlas_np_zyx, dtype=np.float32)),
        atlas_spacing,
    )
    atlas_on_func_img = sitk.Resample(
        atlas_img,
        func_img,
        transform_3d,
        sitk.sitkLinear,
        0.0,
        atlas_img.GetPixelID(),
    )
    atlas_on_func = sitk.GetArrayFromImage(atlas_on_func_img)

    if save_transform_path is not None:
        sitk.WriteTransform(transform_3d, str(save_transform_path))

    info = {
        "mode": "slice_to_volume",
        "acquisition_plane": int(plane_number),
        "atlas_z": atlas_z,
        "stop": reg.GetOptimizerStopConditionDescription(),
        "metric": reg.GetMetricValue(),
        "func_spacing": tuple(func_spacing),
        "atlas_spacing": tuple(atlas_spacing),
        "func_shape_zyx": tuple(int(x) for x in func_np_zyx.shape),
        "rot90_k": int(rot90_k),
        "shrink_factors_per_level": shrink_levels,
        "smoothing_sigmas_per_level": smooth_levels,
    }
    return (
        transform_3d,
        atlas_on_func,
        func_np_zyx,
        plane_order,
        native_plane_shape,
        info,
    )


def register_functional_volume_to_mapzebrain(
    atlas_np_zyx,
    func_np_zyx=None,
    func_spacing=FUNC_SPACING_UM,
    atlas_spacing=ATLAS_SPACING_UM,
    plane_tiff_items=None,
    rot90_k=1,
    atlas_z=None,
    initial_transform=None,
    save_transform_path=None,
    number_of_histogram_bins=50,
    metric_sampling_percentage=0.25,
    learning_rate=1.0,
    number_of_iterations=300,
    shrink_factors=(4, 2, 1),
    smoothing_sigmas=(2, 1, 0),
    normalize=True,
):
    """
    One 3D affine: fixed = functional stack, moving = mapZebrain atlas (Mattes MI).

    Prefer passing ``plane_tiff_items`` + ``rot90_k`` so the functional volume is built
    inside this function (guarantees registration uses the same rot90 as ``load_std_functional_volume``).

    For one **tilted** plane spanning a z-range in the atlas, use
    :func:`register_functional_oblique_plane_to_mapzebrain` (not this function).

    Deprecated: ``atlas_z`` alone routes to :func:`register_functional_slice_to_mapzebrain`
    (single atlas slice, no tilt).

    Returns
    -------
    transform, atlas_on_func, func_np_zyx, plane_order, native_plane_shape, info
    """
    plane_order = None
    native_plane_shape = None
    if plane_tiff_items is not None:
        func_np_zyx, plane_order, native_plane_shape = load_std_functional_volume(
            plane_tiff_items,
            rot90_k=rot90_k,
        )
    elif func_np_zyx is None:
        raise ValueError("Provide func_np_zyx or plane_tiff_items")

    func_np_zyx = np.ascontiguousarray(np.asarray(func_np_zyx, dtype=np.float32))
    if func_np_zyx.shape[0] == 1 and atlas_z is not None:
        if plane_tiff_items is None:
            raise ValueError(
                "Single-plane registration requires plane_tiff_items so rot90 and TIFF reload match."
            )
        return register_functional_slice_to_mapzebrain(
            atlas_np_zyx,
            plane_tiff_items,
            atlas_z=int(atlas_z),
            rot90_k=rot90_k,
            func_spacing=func_spacing,
            atlas_spacing=atlas_spacing,
            initial_transform=initial_transform,
            save_transform_path=save_transform_path,
            number_of_histogram_bins=number_of_histogram_bins,
            metric_sampling_percentage=metric_sampling_percentage,
            learning_rate=learning_rate,
            number_of_iterations=number_of_iterations,
            shrink_factors=shrink_factors,
            smoothing_sigmas=smoothing_sigmas,
            normalize=normalize,
        )
    func = sitk_volume_from_numpy_zyx(func_np_zyx, func_spacing)
    atlas = sitk_volume_from_numpy_zyx(
        np.ascontiguousarray(np.asarray(atlas_np_zyx, dtype=np.float32)),
        atlas_spacing,
    )
    if normalize:
        func_reg = sitk.Normalize(func)
        atlas_reg = sitk.Normalize(atlas)
    else:
        func_reg = func
        atlas_reg = atlas

    if initial_transform is None:
        initial_transform = sitk.CenteredTransformInitializer(
            func_reg,
            atlas_reg,
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

    nz_func = int(func_np_zyx.shape[0])
    # ITK Gaussian smoothing needs >= 4 samples along z; sparse stacks (e.g. 11 planes) are fine.
    if nz_func < 4:
        shrink_levels = [1]
        smooth_levels = [0.0]
    else:
        shrink_levels = [int(s) for s in shrink_factors]
        while shrink_levels and nz_func // shrink_levels[0] < 1:
            shrink_levels = shrink_levels[1:]
        if not shrink_levels:
            shrink_levels = [1]
        smooth_levels = [
            0.0 if nz_func < 4 else float(s)
            for s in smoothing_sigmas[: len(shrink_levels)]
        ]
        while len(smooth_levels) < len(shrink_levels):
            smooth_levels.append(0.0)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=int(number_of_histogram_bins))
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(float(metric_sampling_percentage))
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=float(learning_rate),
        minStep=1e-4,
        numberOfIterations=int(number_of_iterations),
        gradientMagnitudeTolerance=1e-6,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel(shrink_levels)
    reg.SetSmoothingSigmasPerLevel(smooth_levels)
    reg.SetInitialTransform(initial_transform, inPlace=False)

    transform = reg.Execute(func_reg, atlas_reg)
    atlas_on_func_img = sitk.Resample(
        atlas,
        func,
        transform,
        sitk.sitkLinear,
        0.0,
        atlas.GetPixelID(),
    )
    atlas_on_func = sitk.GetArrayFromImage(atlas_on_func_img)

    if save_transform_path is not None:
        sitk.WriteTransform(transform, str(save_transform_path))

    info = {
        "stop": reg.GetOptimizerStopConditionDescription(),
        "metric": reg.GetMetricValue(),
        "func_spacing": tuple(func_spacing),
        "atlas_spacing": tuple(atlas_spacing),
        "func_shape_zyx": tuple(int(x) for x in func_np_zyx.shape),
        "atlas_shape_zyx": tuple(int(x) for x in atlas_np_zyx.shape),
        "shrink_factors_per_level": shrink_levels,
        "smoothing_sigmas_per_level": smooth_levels,
        "rot90_k": int(rot90_k) if plane_tiff_items is not None else None,
        "mode": "volume",
    }
    return transform, atlas_on_func, func_np_zyx, plane_order, native_plane_shape, info


def func_index_to_atlas_continuous_index(col, row, plane_index, func_img, transform, atlas_img):
    """
    ROI COM in functional **native** indices ``(col, row, plane_index)`` → atlas ``(x, y, z)`` continuous index.

    ``plane_index`` is the z index in the stacked functional volume (0 … n_planes-1), not necessarily
    the acquisition plane number unless the stack was built in that order.
    """
    fp = func_img.TransformContinuousIndexToPhysicalPoint(
        (float(col), float(row), float(plane_index))
    )
    ap = transform.TransformPoint(fp)
    return atlas_img.TransformPhysicalPointToContinuousIndex(ap)


def func_index_to_atlas_oblique_continuous_index(
    col,
    row,
    plane_index,
    func_img,
    transform,
    atlas_img,
    atlas_z_top,
    atlas_z_bottom,
    atlas_row_y_top=None,
    atlas_row_y_bottom=None,
    extrapolate=True,
):
    """In-plane from ``transform``; atlas z from **atlas y** (oblique band on reference planes).

    ``extrapolate`` (default ``True``) keeps the y→z map linear past the selected band
    (see :func:`atlas_z_at_atlas_y`)."""
    fp = func_img.TransformContinuousIndexToPhysicalPoint(
        (float(col), float(row), float(plane_index))
    )
    ap = transform.TransformPoint(fp)
    ax, ay, az_from_t = atlas_img.TransformPhysicalPointToContinuousIndex(ap)
    n_atlas_y = int(sitk.GetArrayFromImage(atlas_img).shape[-2])
    az = atlas_z_at_atlas_y(
        ay,
        atlas_z_top,
        atlas_z_bottom,
        atlas_row_y_top=atlas_row_y_top,
        atlas_row_y_bottom=atlas_row_y_bottom,
        n_atlas_rows=n_atlas_y,
        extrapolate=extrapolate,
    )
    return ax, ay, az


def func_plane_number_to_stack_index(plane_number, plane_order):
    """Map acquisition plane number → z index in the stacked volume."""
    plane_order = [int(p) for p in plane_order]
    return plane_order.index(int(plane_number))


def region_names_at_atlas_index(ax, ay, az, region_mask_paths, threshold=0):
    """
    Return region TIFF stem names whose mask is positive at ``(x, y, z)`` voxel indices.

    ``region_mask_paths``: iterable of paths to 3D region masks (z, y, x), same atlas frame as T_AVG_HuCD.
    """
    ax_i, ay_i, az_i = int(round(ax)), int(round(ay)), int(round(az))
    hits = []
    for path in region_mask_paths:
        path = Path(path)
        stack = np.asarray(load_image(path))
        if stack.ndim != 3:
            raise ValueError(f"Expected 3D region mask {path}, got {stack.shape}")
        nz, ny, nx = stack.shape
        if not (0 <= az_i < nz and 0 <= ay_i < ny and 0 <= ax_i < nx):
            continue
        if stack[az_i, ay_i, ax_i] > threshold:
            hits.append(path.stem)
    return hits


def list_mapzebrain_region_mask_paths(regions_dir):
    """Sorted paths to ``*.tif`` / ``*.tiff`` under a mapZebrain regions folder."""
    regions_dir = Path(regions_dir)
    if not regions_dir.is_dir():
        raise FileNotFoundError(regions_dir)
    paths = sorted(
        p
        for p in regions_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff")
    )
    return paths
