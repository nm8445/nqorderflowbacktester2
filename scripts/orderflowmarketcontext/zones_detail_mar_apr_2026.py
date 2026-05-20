"""
Enumerate all support/resistance zones formed March-April 2026, with
formation run detail (bar times + wick deltas), invalidation events, and
"respect" events (touches where the bar's close is on the favorable side
of the zone boundary — i.e. support touch that closes above zone_high,
resistance touch that closes below zone_low).

Outputs (under scripts/orderflowmarketcontext/):
  zones_mar_apr_2026.csv             flat row per zone
  zones_mar_apr_2026_events.csv      flat row per (zone, event) — formation bar,
                                      touches, respects, invalidation
  zones_mar_apr_2026.md              human-readable digest
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ET = "America/New_York"
DATA = Path("D:/trading_pythonbacktest_data")
OUT = Path(__file__).resolve().parent
DATE_FROM = "2026-03-01"
DATE_TO = "2026-05-01"  # exclusive


def fix_tz(s):
    s = pd.to_datetime(s)
    return s.dt.tz_localize("UTC").dt.tz_convert(ET) if s.dt.tz is None else s.dt.tz_convert(ET)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="Zone-cache suffix, e.g. '_pivot'")
    args = ap.parse_args()
    suf = args.suffix

    zones = pd.read_parquet(DATA / f"wick_zones_5min{suf}.parquet")
    zones["formed_at"] = fix_tz(zones["formed_at"])
    zones["run_start_time"] = fix_tz(zones["run_start_time"])
    zones["last_touch_time"] = fix_tz(zones["last_touch_time"])
    zones["invalidation_time"] = fix_tz(zones["invalidation_time"])

    mask = (zones["formed_at"] >= DATE_FROM) & (zones["formed_at"] < DATE_TO)
    zf = zones[mask].sort_values("formed_at").reset_index(drop=True)
    print(f"Zones formed {DATE_FROM} to {DATE_TO}: {len(zf)}")
    print(f"  support: {(zf['zone_type']=='support').sum()}   resistance: {(zf['zone_type']=='resistance').sum()}")

    touches = pd.read_parquet(DATA / f"wick_zone_touches_5min{suf}.parquet")
    touches["touch_time"] = fix_tz(touches["touch_time"])

    idx = pd.read_parquet(DATA / "wick_delta_index_5min.parquet")
    idx["bar_open_time"] = fix_tz(idx["bar_open_time"])
    idx = idx.set_index("bar_open_time")

    # Build event rows
    events = []
    zone_summary = []
    for _, z in zf.iterrows():
        zid = int(z["zone_id"])
        ztype = z["zone_type"]

        # ── Formation bars: all bars from run_start_time to formed_at ──
        run_bars = idx.loc[(idx.index >= z["run_start_time"]) & (idx.index <= z["formed_at"])]
        n_formation = len(run_bars)
        for ts, r in run_bars.iterrows():
            events.append({
                "zone_id": zid, "zone_type": ztype,
                "event_type": "formation_bar",
                "event_time": ts,
                "bar_open": r["open"], "bar_close": r["close"],
                "bar_high": r["high"], "bar_low": r["low"],
                "bot_wick_delta": r["bot_wick_delta"],
                "top_wick_delta": r["top_wick_delta"],
                "bar_total_delta": r["bar_total_delta"],
                "is_bullish": bool(r["is_bullish"]),
                "is_bearish": bool(r["is_bearish"]),
                "zone_low": z["zone_low"], "zone_high": z["zone_high"],
            })

        # ── Touches and respects ──
        tc = touches[touches["zone_id"] == zid].sort_values("touch_time")
        n_touch = len(tc); n_respect = 0
        for _, t in tc.iterrows():
            # Respect rule: support touch close > zone_high; resistance touch close < zone_low
            respected = (
                (ztype == "support"    and t["touch_close"] > z["zone_high"]) or
                (ztype == "resistance" and t["touch_close"] < z["zone_low"])
            )
            ev = "respect" if respected else "touch"
            if respected: n_respect += 1
            events.append({
                "zone_id": zid, "zone_type": ztype,
                "event_type": ev,
                "event_time": t["touch_time"],
                "bar_low": t["touch_low"], "bar_high": t["touch_high"],
                "bar_close": t["touch_close"],
                "zone_low": z["zone_low"], "zone_high": z["zone_high"],
                "is_rth": bool(t.get("is_rth", False)),
            })

        # ── Invalidation ──
        if pd.notna(z["invalidation_time"]):
            events.append({
                "zone_id": zid, "zone_type": ztype,
                "event_type": f"invalidation_{z['invalidation_reason']}",
                "event_time": z["invalidation_time"],
                "zone_low": z["zone_low"], "zone_high": z["zone_high"],
            })

        # Formation run wick-delta summary for the per-zone CSV
        if ztype == "support":
            formation_wick_deltas = ", ".join(
                f"{v:+.0f}" for v in run_bars["bot_wick_delta"].tolist())
        else:
            formation_wick_deltas = ", ".join(
                f"{v:+.0f}" for v in run_bars["top_wick_delta"].tolist())

        zone_summary.append({
            "zone_id": zid, "zone_type": ztype,
            "formed_at": z["formed_at"],
            "run_start_time": z["run_start_time"],
            "run_bars": int(z["run_bars"]),
            "zone_low": z["zone_low"], "zone_high": z["zone_high"],
            "zone_height_pts": z["zone_high"] - z["zone_low"],
            "formation_wick_deltas": formation_wick_deltas,
            "n_touches": n_touch,
            "n_respects": n_respect,
            "last_touch_time": z["last_touch_time"],
            "invalidation_time": z["invalidation_time"],
            "invalidation_reason": z["invalidation_reason"],
        })

    zsum_df = pd.DataFrame(zone_summary)
    ev_df = pd.DataFrame(events).sort_values(["zone_id", "event_time"]).reset_index(drop=True)

    zsum_df.to_csv(OUT / f"zones_mar_apr_2026{suf}.csv", index=False)
    ev_df.to_csv(OUT / f"zones_mar_apr_2026_events{suf}.csv", index=False)

    # Build Markdown digest — compact
    md = []
    md.append(f"# Zones formed 2026-03-01 to 2026-04-30\n")
    md.append(f"Total: **{len(zsum_df)}** zones  "
              f"({(zsum_df['zone_type']=='support').sum()} support, "
              f"{(zsum_df['zone_type']=='resistance').sum()} resistance)\n\n")

    # Lifecycle summary
    inv_counts = zsum_df["invalidation_reason"].fillna("still_active").value_counts()
    md.append("**Lifecycle summary:**\n\n")
    for r, c in inv_counts.items():
        md.append(f"- {r}: {c}\n")
    md.append("\n**Touch/respect distribution:**\n\n")
    md.append(f"- total touches across these zones: {zsum_df['n_touches'].sum()}\n")
    md.append(f"- total respects: {zsum_df['n_respects'].sum()}\n")
    md.append(f"- zones with 0 touches: {(zsum_df['n_touches']==0).sum()}\n")
    md.append(f"- zones with at least 1 respect: {(zsum_df['n_respects']>0).sum()}\n\n")

    md.append("---\n\n")
    md.append("## Per-zone detail\n\n")
    for _, z in zsum_df.iterrows():
        md.append(f"### Zone #{z['zone_id']}  —  {z['zone_type']}  [{z['zone_low']:.2f}, {z['zone_high']:.2f}]  "
                  f"(height {z['zone_height_pts']:.2f} pts)\n\n")
        md.append(f"- **Formed:** {z['formed_at']} (run of {z['run_bars']} bars, started {z['run_start_time']})\n")
        md.append(f"- **Formation wick deltas:** {z['formation_wick_deltas']}\n")

        # Formation bars
        md.append("- **Formation bars:**\n")
        fbars = ev_df[(ev_df["zone_id"] == z["zone_id"]) & (ev_df["event_type"] == "formation_bar")]
        for _, b in fbars.iterrows():
            bull = "bullish" if b["is_bullish"] else ("bearish" if b["is_bearish"] else "doji")
            md.append(f"    - {b['event_time']}  {bull:<7}  "
                      f"OHLC=({b['bar_open']:.2f},{b['bar_high']:.2f},{b['bar_low']:.2f},{b['bar_close']:.2f})  "
                      f"bot_wick_delta={b['bot_wick_delta']:+.0f}  "
                      f"top_wick_delta={b['top_wick_delta']:+.0f}  "
                      f"total_delta={b['bar_total_delta']:+.0f}\n")

        # Touches / respects
        touches_and_respects = ev_df[(ev_df["zone_id"] == z["zone_id"]) &
                                        (ev_df["event_type"].isin(["touch", "respect"]))]
        if not touches_and_respects.empty:
            md.append(f"- **Touches/respects** ({len(touches_and_respects)} total):\n")
            for _, t in touches_and_respects.iterrows():
                tag = "✓ RESPECT" if t["event_type"] == "respect" else "touch"
                rth = "RTH" if t.get("is_rth") else "ETH"
                md.append(f"    - {t['event_time']}  [{rth}] {tag}  "
                          f"bar_low={t['bar_low']:.2f}  bar_high={t['bar_high']:.2f}  "
                          f"bar_close={t['bar_close']:.2f}\n")
        else:
            md.append("- **Touches/respects:** none\n")

        if pd.notna(z["invalidation_time"]):
            md.append(f"- **Invalidated:** {z['invalidation_time']}  ({z['invalidation_reason']})\n")
        else:
            md.append(f"- **Invalidated:** still active at end of data\n")
        md.append("\n")

    (OUT / f"zones_mar_apr_2026{suf}.md").write_text("".join(md), encoding="utf-8")

    # Terminal output: brief summary
    print()
    print(f"Saved:")
    print(f"  {OUT/f'zones_mar_apr_2026{suf}.csv'}   ({len(zsum_df)} rows)")
    print(f"  {OUT/f'zones_mar_apr_2026_events{suf}.csv'}   ({len(ev_df)} rows)")
    print(f"  {OUT/f'zones_mar_apr_2026{suf}.md'}")
    print()
    print(f"Invalidation breakdown:")
    print(inv_counts.to_string())
    print()
    print(f"Touches total: {zsum_df['n_touches'].sum()}")
    print(f"Respects total: {zsum_df['n_respects'].sum()}")
    print(f"Zones with >= 1 respect: {(zsum_df['n_respects']>0).sum()} / {len(zsum_df)}")
    print()
    print("First 10 zones (summary):")
    cols = ["zone_id","zone_type","formed_at","zone_low","zone_high",
            "run_bars","formation_wick_deltas","n_touches","n_respects",
            "invalidation_time","invalidation_reason"]
    print(zsum_df[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
