"""그림·그래프 제시형 문항의 figure spec → 이미지(SVG/PNG) 렌더러.

설계:
- 생성기는 그림을 그리지 않고 JSON 직렬화 가능한 `figure` spec 만 첨부한다
  (예: {"kind": "polyline", "points": [...]}). 렌더링은 이 모듈이 담당.
  → 생성/검증(테스트)은 파일 IO 없이 빠르게, 이미지는 필요할 때만.
- 엔진이 파라메트릭이므로 그림도 코드로 정확히 그릴 수 있다 (정답과 그림이
  같은 파라미터에서 나와 불일치가 원천적으로 불가능).
- 그림 안에는 한글을 쓰지 않는다 (수학 라벨만 — 폰트 의존성 제거).

사용:
    from csat_forge.figure_render import render_figure
    path = render_figure(rec["completion"]["figure"], "out/fig1.svg")

CLI (엔진 그림 문항 샘플 렌더):
    python -m csat_forge.figure_render --out-dir figures
"""
from __future__ import annotations
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _axes_style(ax, xlabel="x", ylabel="y"):
    """수능 그림 스타일: 원점 교차 축 + 화살표 라벨, 테두리 제거."""
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel(xlabel, loc="right", fontsize=11, style="italic")
    ax.set_ylabel(ylabel, loc="top", rotation=0, fontsize=11, style="italic")
    ax.text(-0.15, -0.4, "O", fontsize=11, ha="right", style="italic")


def _render_parabola_region(spec, ax):
    kf, a = spec["k"], spec["a"]
    xs = np.linspace(-0.3*a, a*1.3, 300)
    ys = kf*xs*(a - xs)
    ax.plot(xs, ys, color="black", lw=1.4)
    xs_in = np.linspace(0, a, 200)
    ax.fill_between(xs_in, kf*xs_in*(a - xs_in), 0,
                    facecolor="none", edgecolor="black", hatch="///", lw=0)
    ax.set_xticks([a])
    ax.set_yticks([])
    _axes_style(ax)


def _render_polyline(spec, ax):
    pts = spec["points"]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color="black", lw=1.4)
    # 꼭짓점에서 축으로 점선 안내선
    for px, py in pts:
        if px and py:
            ax.plot([px, px], [0, py], ls=":", lw=0.8, color="gray")
            ax.plot([0, px], [py, py], ls=":", lw=0.8, color="gray")
    ax.set_xticks(sorted({p[0] for p in pts if p[0]}))
    ax.set_yticks(sorted({p[1] for p in pts if p[1]}))
    _axes_style(ax, spec.get("xlabel", "x"), spec.get("ylabel", "y"))


_RENDERERS = {
    "parabola_region": _render_parabola_region,
    "polyline": _render_polyline,
}


def render_figure(spec: dict, out_path: str) -> str:
    """figure spec → 이미지 파일 (확장자로 SVG/PNG 결정). 경로 반환."""
    kind = spec.get("kind")
    if kind not in _RENDERERS:
        raise ValueError(f"지원하지 않는 figure kind: {kind}")
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    _RENDERERS[kind](spec, ax)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def main():
    import argparse
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from csat_forge import problem_engine as pe
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join("finetune", "data", "figures"))
    ap.add_argument("--fmt", choices=["svg", "png"], default="svg")
    args = ap.parse_args()
    for name in ("fig_parabola_area", "fig_velocity_graph"):
        rec = pe.GENERATORS[name]()
        spec = rec["completion"]["figure"]
        path = os.path.join(args.out_dir, f"{name}.{args.fmt}")
        render_figure(spec, path)
        print(f"{name}: {rec['completion']['question'][:60]}… → {path}")


if __name__ == "__main__":
    main()
