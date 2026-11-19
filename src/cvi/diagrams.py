"""Tiny diagram toolkit: boxes are sized from MEASURED text, coordinates are in inches (1 unit = 1 inch,
so circles stay circles and nothing overflows).  Used for the pipeline and star-schema diagrams."""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from . import viz

T = viz.THEMES["light"]


def canvas(w: float, h: float):
    t = viz.apply_style("light")
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


@dataclass
class Box:
    title: str
    lines: list[str] = field(default_factory=list)
    fs: float = 6.6
    title_fs: float = 7.9
    accent: str | None = None
    fill: str | None = None
    edge: str | None = None
    lw: float = 0.9
    pad: float = 0.09
    line_gap: float = 0.035
    min_w: float = 0.0
    bold_edge: bool = False

    def _text_w(self, fig, ax, s: str, fs: float, weight=None) -> float:
        t = ax.text(0, 0, s, fontsize=fs, fontweight=weight or "normal")
        r = fig.canvas.get_renderer()
        w = t.get_window_extent(r).width / fig.dpi
        t.remove()
        return w

    def size(self, fig, ax) -> tuple[float, float]:
        w = max([self._text_w(fig, ax, self.title, self.title_fs, 500)] + [self._text_w(fig, ax, ln, self.fs) for ln in self.lines] + [0]) + 2 * self.pad
        w = max(w, self.min_w)
        lh = self.fs / 72 * 1.18
        th = self.title_fs / 72 * 1.18
        h = self.pad * 2 + th + (self.line_gap * 1.5 + len(self.lines) * (lh + self.line_gap) if self.lines else 0)
        return w, h

    def draw(self, fig, ax, x: float, y_top: float, w: float | None = None, h: float | None = None) -> dict:
        bw, bh = self.size(fig, ax)
        w = max(w or bw, bw)
        h = max(h or bh, bh)
        y0 = y_top - h
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=self.fill or T["surface"],
                                    ec=self.edge or T["axis"], lw=self.lw * (1.6 if self.bold_edge else 1), zorder=2))
        if self.accent:
            ax.add_patch(FancyBboxPatch((x, y_top - 0.035), w, 0.035, boxstyle="square,pad=0", fc=self.accent, ec="none", zorder=3))
        ax.text(x + self.pad, y_top - self.pad, self.title, ha="left", va="top", fontsize=self.title_fs, fontweight=500, color=T["ink"], zorder=4)
        lh = self.fs / 72 * 1.18
        cy = y_top - self.pad - self.title_fs / 72 * 1.18 - self.line_gap * 1.5
        for ln in self.lines:
            ax.text(x + self.pad, cy, ln, ha="left", va="top", fontsize=self.fs, color=T["ink2"], zorder=4)
            cy -= lh + self.line_gap
        return {"l": x, "r": x + w, "t": y_top, "b": y0, "cx": x + w / 2, "cy": y0 + h / 2, "w": w, "h": h}


def arrow(ax, p0, p1, color=None, lw=1.2, head=True):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>" if head else "-", mutation_scale=9, lw=lw, color=color or T["muted"], zorder=3, shrinkA=0, shrinkB=0))


def polyline(ax, pts, color=None, lw=1.2, head=True):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color or T["muted"], lw=lw, zorder=3, solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], color, lw, head)


def badge(ax, x, y, text, r=0.1):
    ax.add_patch(Circle((x, y), r, fc=T["surface"], ec=T["ink2"], lw=1.0, zorder=6))
    ax.text(x, y, text, ha="center", va="center", fontsize=6.8, color=T["ink"], zorder=7)


def line(ax, p0, p1, color=None, lw=0.9):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color or T["axis"], lw=lw, zorder=1, solid_capstyle="butt")
