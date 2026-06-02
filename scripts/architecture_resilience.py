"""Resilience / failure-mode architecture diagram (SVG) for the live trading system.
Frames each real-time pipeline stage as FAILURE MODE -> RESILIENCE MECHANISM, plus the
concurrency/threading model. matplotlib only.  Run: python scripts/architecture_resilience.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "architecture_resilience"
SLATE, RED, GREEN = "#34495E", "#B23A3A", "#2E8B7A"

ROWS = [
    ("Tick ingest\n(DatabentoLiveFeed)",
     "Feed stalls / bursts;\nlocal queue overflow",
     "Bounded async queue decouples ingest from\nprocessing (slow-client safe); drop-on-full\ncounter; reconnecting websocket consumer"),
    ("Cold start /\nmid-session restart",
     "Indicators cold; pre-start\nsession bars missing; stale state",
     "warm_start seeds 15 trading days; session_backfill\nfetches the gap via Historical API (T+30 / T+60);\nstate restore gated to <5-min freshness"),
    ("Engine state\nafter restart",
     "Phantom positions; warm-replay\ndouble-counts the same bars",
     "Warm-start positions force-cleared post-seed;\nend_date_exclusive blocks same-day double-count;\ncoordinator desync-fix reconciles vs broker"),
    ("Risk / coordination",
     "Self-hedging across strats;\ntrading into red-folder news",
     "No-hedge filter blocks + rolls back opposing\nentries; NewsBlackout watcher force-flattens\n2:30 before every T1 (FOMC/CPI/NFP)"),
    ("Executor -> broker\n(network I/O)",
     "Broker slow / unreachable\nblocks the engine thread",
     "Dedicated poster thread; engine does non-blocking\nenqueue (microseconds); 5s POST timeout;\nfailed order NOT silently retried (no duplicates)"),
    ("Python process\ndies / goes silent",
     "Open positions left\nunmanaged at the broker",
     "Heartbeat dead-man's-switch: NT8 addon\nauto-flattens all tagged positions after\n30s of missed heartbeats (failsafe)"),
    ("Order integrity\n/ fills",
     "Close hits wrong or already-\nclosed position; double-close reverses",
     "Tag-scoped per-strategy routing; broker-side\nside/qty validation ABORTS a close that doesn't\nmatch the tracked position (no accidental flip)"),
    ("MT5 has no\nresting bracket",
     "Intrabar SL/TP only seen on\nbar close -> overshoot risk",
     "TickPositionMonitor checks every tick and fires\nan immediate protective close the moment a\nlevel is crossed (matches NT8 resting-fill timing)"),
]


def box(ax, cx, cy, w, h, text, fc, fs=8, tc="white", bold=True):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                 boxstyle="round,pad=0.1,rounding_size=0.5", fc=fc, ec="white", lw=1.2, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, x0, y0, x1, y1, color="#888"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                 lw=1.3, color=color, zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(17, 13.5))
    ax.set_xlim(0, 104); ax.set_ylim(-4, 102); ax.axis("off")
    sx, fx, mx = 13, 45, 81
    sw, fw, mw = 21, 27, 39
    # headers
    for cx, w, lbl, c in [(sx, sw, "PIPELINE STAGE", SLATE), (fx, fw, "FAILURE MODE", RED),
                          (mx, mw, "RESILIENCE / RECOVERY MECHANISM", GREEN)]:
        box(ax, cx, 96.5, w, 5, lbl, c, fs=10)
    n = len(ROWS); top = 87; gap = 10.0; bh = 8.4
    for i, (stage, fail, mit) in enumerate(ROWS):
        y = top - i * gap
        box(ax, sx, y, sw, bh, stage, SLATE, fs=8)
        box(ax, fx, y, fw, bh, fail, RED, fs=7.4, bold=False)
        box(ax, mx, y, mw, bh, mit, GREEN, fs=7.2, bold=False)
        arrow(ax, sx + sw/2, y, fx - fw/2, y, "#999")
        arrow(ax, fx + fw/2, y, mx - mw/2, y, GREEN)
    # concurrency footer
    box(ax, 50, 1.5, 98, 7.5,
        "CONCURRENCY MODEL  —  daemon threads decouple the engine from all I/O:\n"
        "[websocket tick consumer]   [HTTP order poster]   [10s heartbeat]   "
        "[news-blackout watcher]   [session-backfill scheduler]\n"
        "the strategy / engine thread never blocks on broker or network I/O",
        "#2C3E50", fs=9)
    ax.set_title("Live Trading System — Resilience & Failure-Mode Architecture",
                 fontsize=16, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".svg"))
    fig.savefig(OUT.with_suffix(".png"), dpi=110)
    print(f"Saved -> {OUT.with_suffix('.svg')}  (+ .png preview)")


if __name__ == "__main__":
    main()
