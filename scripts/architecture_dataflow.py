"""Render a professional dataflow / architecture diagram of the NQ trading system to SVG.
Layers: external feeds -> data stores -> ingestion -> bar building -> strategy engines ->
risk/coordination -> execution -> brokers, plus a research/backtest lane. matplotlib only.
Run:  python scripts/architecture_dataflow.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "architecture_dataflow"

C = {  # palette
    "ext": "#3C6E9C", "store": "#6B7A8F", "ingest": "#2E8B7A", "bars": "#2E8B7A",
    "eng": "#D08226", "risk": "#7E5BA6", "exec": "#C0392B", "broker": "#2C3E50",
    "research": "#6B8E23",
}
NODES = {}   # name -> dict(cx,cy,w,h,fc,label,fs)
EDGES = []   # (a,b,label,style,color)


def N(name, cx, cy, w, h, fc, label, fs=9):
    NODES[name] = dict(cx=cx, cy=cy, w=w, h=h, fc=fc, label=label, fs=fs)


def E(a, b, label="", style="solid", color="#555"):
    EDGES.append((a, b, label, style, color))


# ---------------- layout (0-100 canvas) ----------------
# External feeds (top)
N("dbl", 16, 94, 24, 7, C["ext"], "Databento Live\n(MBP-1 NQ ticks)")
N("theta", 58, 94, 24, 7, C["ext"], "ThetaData\n(QQQ / NDX EOD greeks)")
# Data stores
N("barpq", 16, 81, 26, 7, C["store"], "Bar parquets  (D:/)\n1-min + 5-min volumetric/delta")
N("gampq", 58, 81, 26, 7, C["store"], "Gamma-levels parquet\n(MenthorQ-style daily)")
# Ingestion
N("feed", 13, 67, 20, 7, C["ingest"], "DatabentoLiveFeed\nasync tick queue (slow-client)")
N("warm", 37, 67, 20, 7, C["ingest"], "warm_start /\nsession_backfill")
N("gamref", 62, 67, 22, 7, C["ingest"], "gamma_refresh\n(greeks -> NQ levels)")
# Bar building
N("bars", 30, 53.5, 36, 7, C["bars"], "MultiBarBuilder\n5-min + 20-min bars  (+ per-level volume / delta)")
# Strategy engines
N("od", 7.5, 40, 13, 7.5, C["eng"], "OD\novernight drift\n19:00 long")
N("rv", 23, 40, 13, 7.5, C["eng"], "RV\nrough-vol +\norderflow")
N("b2", 38.5, 40, 13, 7.5, C["eng"], "B2\novernight-range\ngamma")
N("fb", 54, 40, 13, 7.5, C["eng"], "FB\nORB\nbreakout")
# Risk / coordination
N("coord", 24, 27, 26, 7.5, C["risk"], "Coordinator\nno-hedge filter + state persist\n+ martingale state")
N("news", 55, 28, 16, 6.5, C["risk"], "NewsBlackout\nwatcher (flatten\npre-T1 events)")
N("state", 73, 27, 15, 7, C["store"], "State store\npositions / marti /\ncoordinator JSON")
# Execution
N("nt8x", 20, 14.5, 20, 7, C["exec"], "NT8 Executor\nasync poster -> HTTP :8081\nheartbeat failsafe")
N("mt5x", 47, 14.5, 20, 7, C["exec"], "MT5 Executor\ndynamic lot sizing\n($-risk per trade)")
N("tick", 73, 14.5, 15, 7, C["risk"], "TickPositionMonitor\nintrabar protective\nclose (MT5)")
# Brokers
N("nt8", 20, 3.5, 22, 6.5, C["broker"], "NinjaTrader 8\nNQMultiStratReceiver addon -> NQ futures")
N("mt5", 47, 3.5, 20, 6.5, C["broker"], "MetaTrader 5\nNAS100 CFD (prop)")
# Research lane (right)
N("bt", 91, 70, 16, 8, C["research"], "Backtest engine\n+ per-strat sweeps")
N("mc", 91, 54, 16, 8, C["research"], "Monte-Carlo\nprop-firm sims\n(pass-rate / EV / DD)")
N("cfg", 91, 38, 16, 7, C["research"], "Locked configs\n(per strat)")

# ---------------- edges ----------------
E("dbl", "feed", "ticks")
E("theta", "gamref", "greeks")
E("gamref", "gampq", "")
E("barpq", "warm", "history")
E("feed", "bars", "ticks")
E("warm", "bars", "seed / gap-fill")
E("bars", "od"); E("bars", "rv"); E("bars", "b2"); E("bars", "fb")
E("gampq", "b2", "gamma sign", "dashed", "#7E5BA6")
for s in ("od", "rv", "b2", "fb"):
    E(s, "coord", "" if s != "od" else "signals")
E("news", "coord", "flatten gate", "dashed", "#7E5BA6")
E("coord", "nt8x", "approved")
E("coord", "mt5x", "approved")
E("coord", "state", "persist", "dashed", "#6B7A8F")
E("tick", "mt5x", "intrabar close", "dashed", "#7E5BA6")
E("nt8x", "nt8", "tagged orders")
E("mt5x", "mt5", "orders")
# research lane
E("barpq", "bt", "5yr history")
E("bt", "mc", "trade logs")
E("mc", "cfg", "best configs")
E("cfg", "fb", "params", "dashed", "#6B8E23")
E("cfg", "rv", "", "dashed", "#6B8E23")


def edge_pt(n, tx, ty):
    dx, dy = tx - n["cx"], ty - n["cy"]
    if dx == 0 and dy == 0:
        return n["cx"], n["cy"]
    sx = (n["w"] / 2) / abs(dx) if dx else 1e9
    sy = (n["h"] / 2) / abs(dy) if dy else 1e9
    t = min(sx, sy)
    return n["cx"] + dx * t, n["cy"] + dy * t


def main():
    fig, ax = plt.subplots(figsize=(17, 12))
    ax.set_xlim(0, 102); ax.set_ylim(-1, 100); ax.axis("off")

    # layer band labels
    bands = [(94, "EXTERNAL FEEDS"), (81, "DATA STORES"), (67, "INGESTION"),
             (53.5, "BAR BUILDING"), (40, "STRATEGY ENGINES"), (27.5, "RISK / COORDINATION"),
             (14.5, "EXECUTION"), (3.5, "BROKERS")]
    for y, lbl in bands:
        ax.text(-0.5, y, lbl, rotation=90, va="center", ha="center",
                fontsize=8, color="#999", fontweight="bold")
    ax.text(91, 80, "RESEARCH / BACKTEST", ha="center", fontsize=8, color="#999", fontweight="bold")

    # edges first (under boxes)
    for a, b, label, style, color in EDGES:
        na, nb = NODES[a], NODES[b]
        x0, y0 = edge_pt(na, nb["cx"], nb["cy"])
        x1, y1 = edge_pt(nb, na["cx"], na["cy"])
        ls = (0, (5, 3)) if style == "dashed" else "-"
        arr = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                              lw=1.3, color=color, linestyle=ls, shrinkA=0, shrinkB=0,
                              connectionstyle="arc3,rad=0.0", zorder=1)
        ax.add_patch(arr)
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, fontsize=6.5, color="#444",
                    ha="center", va="center", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

    # boxes
    for n in NODES.values():
        box = FancyBboxPatch((n["cx"] - n["w"] / 2, n["cy"] - n["h"] / 2), n["w"], n["h"],
                             boxstyle="round,pad=0.15,rounding_size=0.6", fc=n["fc"],
                             ec="white", lw=1.4, zorder=2)
        ax.add_patch(box)
        ax.text(n["cx"], n["cy"], n["label"], ha="center", va="center",
                fontsize=n["fs"], color="white", fontweight="bold", zorder=3)

    ax.set_title("NQ Order-Flow Trading System — Data Flow & Architecture",
                 fontsize=16, fontweight="bold", pad=16)
    # legend
    leg = [("Live tick/bar dataflow", "#555", "-"), ("Feature / gate / config (control)", "#7E5BA6", "dashed")]
    for i, (lbl, col, st) in enumerate(leg):
        y = 98 - i * 2.4
        ax.plot([2, 8], [y, y], color=col, lw=1.4,
                ls=(0, (5, 3)) if st == "dashed" else "-")
        ax.text(8.6, y, lbl, fontsize=8, va="center", color="#333")

    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".svg"))
    fig.savefig(OUT.with_suffix(".png"), dpi=110)   # png for quick visual check
    print(f"Saved -> {OUT.with_suffix('.svg')}  (+ .png preview)")


if __name__ == "__main__":
    main()
