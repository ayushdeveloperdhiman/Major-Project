"""Shared visual system for every figure and dashboard chart in the project.

Design rules (validated with the data-viz skill's palette validator; see README):
* Colour follows the ENTITY, never its rank.
* Sentiment is *polarity* -> diverging pair blue(positive) <-> red(negative) with a gray midpoint.
  Validated: red/blue CVD dE 21.6 light / 19.2 dark, normal-vision dE 32.3 / 29.0.
* The three behavioural segments use categorical slots orange / violet / aqua.
  Validated ``--pairs all``: CVD dE 9.2 light / 9.4 dark, normal-vision 27.6 / 24.6.
  Aqua is 2.74:1 on the light surface (WARN) -> it is always paired with visible direct labels or a table.
* Sentiment colours and segment colours are NEVER drawn in the same chart: the validator FAILs that union
  (blue<->violet CVD dE 4.5; red<->orange normal-vision dE 7.1).
* Magnitude = one-hue (blue) sequential ramp, light -> dark.  Status colours are reserved for KPI status.
* Thin marks (<= 24 px), 4 px rounded data-end (square at the baseline), 2 px lines, hairline solid grid,
  selective labels, a legend whenever there are >= 2 series, text in text-tokens (never series colour).
* No dual axes.  Every figure has a CSV twin in outputs/tables (the table view).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from . import config as C  # noqa: E402

# ------------------------------------------------------------------ tokens #
THEMES = {
    "light": dict(surface="#fcfcfb", page="#f9f9f7", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", border="rgba(11,11,11,0.10)", deemph="#c3c2b7",
                  on_fill="#ffffff", good_text="#006300"),
    "dark": dict(surface="#1a1a19", page="#0d0d0d", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", border="rgba(255,255,255,0.10)", deemph="#4a4a46",
                 on_fill="#ffffff", good_text="#0ca30c"),
}
SENTIMENT = {  # diverging pair + neutral midpoint
    "light": {"Negative": "#e34948", "Neutral": "#c3c2b7", "Positive": "#2a78d6"},
    "dark": {"Negative": "#e66767", "Neutral": "#898781", "Positive": "#3987e5"},
}
SEGMENT_NAMES = ["At-Risk", "Satisfied Non-Repeaters", "Loyal Repeaters"]
SEGMENT = {
    "light": dict(zip(SEGMENT_NAMES, ["#eb6834", "#6250d6", "#1baf7a"])),
    "dark": dict(zip(SEGMENT_NAMES, ["#d95926", "#9085e9", "#199e70"])),
}
ACCENT = {"light": "#2a78d6", "dark": "#3987e5"}          # categorical slot 1 (single neutral series)
SERIES = {  # categorical slots 1..8 (fixed order, never cycled)
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]   # steps 100..700
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
STATUS_ICON = {"good": "▲", "warning": "◆", "serious": "●", "critical": "▼"}

FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
_available = {f.name for f in fm.fontManager.ttflist}
FONT = next((f for f in FONT_STACK if f in _available), "DejaVu Sans")


def seq_cmap(theme: str = "light") -> LinearSegmentedColormap:
    ramp = BLUE_RAMP if theme == "light" else BLUE_RAMP[::-1]      # anchor flips in dark
    return LinearSegmentedColormap.from_list(f"cvi_blue_{theme}", ramp)


def div_cmap(theme: str = "light") -> LinearSegmentedColormap:
    mid = "#f0efec" if theme == "light" else "#383835"
    neg, pos = SENTIMENT[theme]["Negative"], SENTIMENT[theme]["Positive"]
    return LinearSegmentedColormap.from_list(f"cvi_div_{theme}", [neg, mid, pos])


def apply_style(theme: str = "light") -> dict:
    t = THEMES[theme]
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "font.family": FONT, "font.size": 8.5, "text.color": t["ink"],
        "axes.edgecolor": t["axis"], "axes.labelcolor": t["ink2"], "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": t["muted"], "ytick.color": t["muted"], "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 4, "ytick.major.pad": 4,
        "axes.grid": False, "grid.color": t["grid"], "grid.linewidth": 0.8, "grid.linestyle": "-",
        "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": t["ink2"],
        "axes.titlesize": 9.5, "axes.titleweight": 500, "axes.titlelocation": "left", "axes.titlepad": 8,
        "figure.dpi": 110, "savefig.dpi": 200,
    })
    return t


# ------------------------------------------------------------- primitives #
def new_fig(w: float = 6.6, h: float = 3.6, theme: str = "light", **kw):
    t = apply_style(theme)
    fig, ax = plt.subplots(figsize=(w, h), **kw)
    return fig, ax, t


def new_grid(nrows: int = 1, ncols: int = 1, w: float = 6.6, h: float = 3.6, theme: str = "light", **kw):
    """Create a figure/axes grid AFTER applying the style (rcParams are read at artist-creation time)."""
    t = apply_style(theme)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), **kw)
    return fig, axes, t


def style_axes(ax, theme: str = "light", grid: str = "y", spines_left: bool = True) -> None:
    t = THEMES[theme]
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.spines["left"].set_color(t["axis"])
    ax.spines["left"].set_visible(spines_left)
    ax.tick_params(axis="both", colors=t["muted"], length=0, labelsize=8)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontfamily(FONT)
    if grid:
        ax.grid(True, axis=grid, color=t["grid"], linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
    ax.set_facecolor(t["surface"])


def heading(fig, title: str, sub: str | None = None, theme: str = "light", x: float = 0.012, y: float | None = None) -> None:
    """Left-aligned title (primary ink) + subtitle (secondary ink) - never in series colour.
    Offsets are ABSOLUTE (inches) so short and tall figures lay out identically."""
    t = THEMES[theme]
    H = fig.get_figheight()
    top = 1 - 0.10 / H
    fig.text(x, top, title, ha="left", va="top", fontsize=11, fontweight=500, color=t["ink"])
    if sub:
        fig.text(x, top - 0.27 / H, sub, ha="left", va="top", fontsize=8.2, color=t["ink2"])


def top(fig, inches: float = 0.82) -> float:
    """Figure-fraction for subplots_adjust(top=...) that leaves `inches` for the heading block."""
    return 1 - inches / fig.get_figheight()


def legend_keys(ax, items: list[tuple[str, str]], theme: str = "light", loc: str = "upper right", ncol: int = 1,
                kind: str = "rect", bbox=None) -> None:
    """Legend with colour keys: items=[(label, colour)].  rect for bars/areas, line for lines."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor="none", label=l) if kind == "rect"
               else Line2D([0], [0], color=c, lw=2, marker="o", markersize=5, label=l) for l, c in items]
    leg = ax.legend(handles=handles, loc=loc, ncol=ncol, frameon=False, handlelength=1.0, handleheight=1.0,
                    labelcolor=THEMES[theme]["ink2"], fontsize=8, bbox_to_anchor=bbox, borderaxespad=0.2,
                    columnspacing=1.2, handletextpad=0.5)
    ax.add_artist(leg)


def _px_per_unit(ax, axis: str) -> float:
    fig = ax.figure
    bbox = ax.get_position()
    if axis == "x":
        lo, hi = ax.get_xlim()
        return (bbox.width * fig.get_figwidth() * fig.dpi) / max(abs(hi - lo), 1e-12)
    lo, hi = ax.get_ylim()
    return (bbox.height * fig.get_figheight() * fig.dpi) / max(abs(hi - lo), 1e-12)


def rounded_bar(ax, x0: float, y0: float, w: float, h: float, color: str, orient: str = "h", radius_px: float = 4.0, z: int = 3):
    """Bar with a 4px rounded DATA end and a square baseline end (baseline overshoot is clipped by the axes)."""
    px, py = _px_per_unit(ax, "x"), _px_per_unit(ax, "y")
    rx, ry = radius_px / px, radius_px / py
    ext = 3 * radius_px            # overshoot past the baseline so its corners are clipped away
    if orient == "h":              # grows along +x from x0
        a = x0 - ext / px if w >= 0 else x0
        ww = (w + ext / px) if w >= 0 else (w - ext / px)
        if w < 0:
            a, ww = x0 + w - ext / px, abs(w) + 2 * ext / px
        patch = FancyBboxPatch((a, y0 - h / 2), ww, h, boxstyle=f"round,pad=0,rounding_size={rx}",
                               mutation_aspect=ry / rx, fc=color, ec="none", lw=0, zorder=z)
    else:                          # grows along +y from y0
        b = y0 - ext / py
        hh = h + ext / py
        patch = FancyBboxPatch((x0 - w / 2, b), w, hh, boxstyle=f"round,pad=0,rounding_size={rx}",
                               mutation_aspect=ry / rx, fc=color, ec="none", lw=0, zorder=z)
    ax.add_patch(patch)
    return patch


def hbars(ax, labels, values, colors, theme="light", thick_px: float = 16, fmt="{:.1f}%", err=None, label_pad_px: float = 6,
          ref: float | None = None, ref_label: str | None = None, xmax: float | None = None, sort: bool = False):
    """Ranked horizontal bars, categories top->bottom, value at the tip, optional CI whiskers and a reference line."""
    t = THEMES[theme]
    n = len(labels)
    order = np.argsort(values)[::-1] if sort else np.arange(n)
    labels = [labels[i] for i in order]
    values = np.asarray(values, float)[order]
    colors = [colors[i] if isinstance(colors, (list, tuple, np.ndarray)) else colors for i in order]
    if err is not None:
        err = np.asarray(err, float)[:, order] if np.ndim(err) == 2 else np.asarray(err, float)[order]
    ypos = np.arange(n)[::-1]
    hi = (values + (err[1] if err is not None and np.ndim(err) == 2 else 0)).max() if n else 1
    ax.set_xlim(0, xmax or hi * 1.22)
    ax.set_ylim(-1.15 if ref_label else -0.7, n - 0.3)
    style_axes(ax, theme, grid="x", spines_left=False)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, color=t["ink2"], fontsize=8.2)
    py = _px_per_unit(ax, "y")
    h = thick_px / py
    for y, v, c in zip(ypos, values, colors):
        rounded_bar(ax, 0, y, v, h, c, "h")
    if err is not None and np.ndim(err) == 2:
        for y, v, lo_, hi_ in zip(ypos, values, err[0], err[1]):
            ax.plot([lo_, hi_], [y, y], color=t["ink2"], lw=0.9, zorder=4, solid_capstyle="butt")
            for e in (lo_, hi_):
                ax.plot([e, e], [y - 0.13, y + 0.13], color=t["ink2"], lw=0.9, zorder=4)
    px = _px_per_unit(ax, "x")
    for y, v, hi_ in zip(ypos, values, (err[1] if err is not None and np.ndim(err) == 2 else values)):
        ax.text(max(v, hi_) + label_pad_px / px, y, fmt.format(v), va="center", ha="left", fontsize=8, color=t["ink"], zorder=5)
    if ref is not None:
        ax.axvline(ref, color=t["muted"], lw=1.0, zorder=2)
        if ref_label:
            ax.text(ref, -1.05, ref_label, ha="center", va="bottom", fontsize=7.4, color=t["muted"])
    return ypos


def vbars(ax, labels, values, colors, theme="light", thick_px: float = 26, fmt="{:.0f}", err=None, ymax=None, label=True, label_pad_px=5):
    t = THEMES[theme]
    n = len(labels)
    xs = np.arange(n)
    values = np.asarray(values, float)
    colors = colors if isinstance(colors, (list, tuple)) else [colors] * n
    hi = (err[1].max() if err is not None else values.max())
    ax.set_ylim(0, ymax or hi * 1.18)
    ax.set_xlim(-0.6, n - 0.4)
    style_axes(ax, theme, grid="y")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, color=t["ink2"], fontsize=8.2)
    px = _px_per_unit(ax, "x")
    w = thick_px / px
    for x, v, c in zip(xs, values, colors):
        rounded_bar(ax, x, 0, w, v, c, "v")
    if err is not None:
        for x, lo_, hi_ in zip(xs, err[0], err[1]):
            ax.plot([x, x], [lo_, hi_], color=t["ink2"], lw=0.9, zorder=4)
            for e in (lo_, hi_):
                ax.plot([x - 0.06, x + 0.06], [e, e], color=t["ink2"], lw=0.9, zorder=4)
    if label:
        py = _px_per_unit(ax, "y")
        tops = err[1] if err is not None else values
        for x, v, top in zip(xs, values, tops):
            ax.text(x, top + label_pad_px / py, fmt.format(v), ha="center", va="bottom", fontsize=8, color=t["ink"], zorder=5)
    return xs


def stacked_gap_bars_h(ax, labels, parts: dict[str, np.ndarray], colors: dict[str, str], theme="light", thick_px=22, gap_px=2,
                       label_min_share: float = 0.075, fmt="{:.0f}%"):
    """100% stacked horizontal bars, 2px surface gaps; in-segment labels only where the text fits."""
    t = THEMES[theme]
    n = len(labels)
    ypos = np.arange(n)[::-1]
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, n - 0.3)
    style_axes(ax, theme, grid="x", spines_left=False)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, color=t["ink2"], fontsize=8.2)
    py, px = _px_per_unit(ax, "y"), _px_per_unit(ax, "x")
    h = thick_px / py
    left = np.zeros(n)
    for name, vals in parts.items():
        vals = np.asarray(vals, float)
        for i, (y, v) in enumerate(zip(ypos, vals)):
            if v <= 0:
                continue
            gap = gap_px / px
            ax.add_patch(FancyBboxPatch((left[i], y - h / 2), max(v - gap, 0), h, boxstyle="square,pad=0", fc=colors[name], ec="none", lw=0, zorder=3))
            if v / 100 >= label_min_share:
                txt_color = "#0b0b0b" if colors[name] in ("#c3c2b7", "#eda100") else t["on_fill"]
                ax.text(left[i] + v / 2 - gap / 2, y, fmt.format(v), ha="center", va="center", fontsize=7.6, color=txt_color, zorder=5)
        left = left + vals
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    return ypos


def save(fig, name: str, theme: str = "light") -> str:
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=200, facecolor=THEMES[theme]["surface"])
    plt.close(fig)
    return str(path)
