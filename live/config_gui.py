"""
Startup config GUI — choose trading mode before live trader starts.

Combined strategy: base VWAP (ADX 15-30) + trending band (ADX 30+).

Modes:
  - Fullport:  Base SL 0.50/TP 1.90, Trend SL 1.00/TP 1.00, dynamic sizing (~$1k risk)
  - Funded:    Base SL 0.50/TP 1.90, Trend SL 1.00/TP 1.00, fixed 3 MNQ
  - Custom:    User sets all params
"""
import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from typing import Optional
import math


MNQ_POINT_VALUE = 2.0  # $2 per point per MNQ contract


@dataclass
class TradingConfig:
    mode: str               # "fullport", "funded", "custom"
    sl_mult: float          # Base SL multiplier
    tp_mult: float          # Base TP multiplier
    trend_sl_mult: float    # Trending band SL multiplier
    trend_tp_mult: float    # Trending band TP multiplier
    instrument: str         # NT8 instrument name e.g. "MNQ 06-26"
    account: str            # NT8 account name (primary/first account)
    contracts: Optional[int]  # fixed contract count (None = dynamic)
    target_risk: float      # $ risk per trade for dynamic sizing
    paper: bool             # paper mode (no orders to NT8)
    accounts: Optional[list] = None  # list of accounts for rotation (Lucid multi-acct); None = single-account mode using `account`


PRESETS = {
    "fullport": {
        "sl_mult": 0.50,
        "tp_mult": 1.90,
        "trend_sl_mult": 1.00,
        "trend_tp_mult": 1.00,
        "contracts": None,  # dynamic
        "target_risk": 1000.0,
    },
    "funded": {
        "sl_mult": 0.50,
        "tp_mult": 1.90,
        "trend_sl_mult": 1.00,
        "trend_tp_mult": 1.00,
        "contracts": 3,
        "target_risk": 0,
    },
}


def calculate_contracts(atr: float, sl_mult: float, target_risk: float) -> int:
    """Calculate MNQ contract count for target dollar risk."""
    sl_points = atr * sl_mult
    if sl_points <= 0:
        return 1
    contracts = int(target_risk / (sl_points * MNQ_POINT_VALUE))
    return max(1, contracts)


def show_config_gui() -> Optional[TradingConfig]:
    """Show config window. Returns TradingConfig or None if cancelled."""
    result = [None]

    root = tk.Tk()
    root.title("VWAP Strategy — Live Config")
    root.geometry("460x620")
    root.resizable(False, False)

    # Center on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() - 460) // 2
    y = (root.winfo_screenheight() - 620) // 2
    root.geometry(f"+{x}+{y}")

    # Style
    style = ttk.Style()
    style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
    style.configure("Sub.TLabel", font=("Segoe UI", 9))

    frame = ttk.Frame(root, padding=20)
    frame.pack(fill="both", expand=True)

    # Header
    ttk.Label(frame, text="Combined VWAP Strategy", style="Header.TLabel").pack(anchor="w")
    ttk.Label(frame, text="Base (ADX 15-30) + Trending Band (ADX 30+)", style="Sub.TLabel").pack(anchor="w")
    ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=(8, 12))

    # Mode selection
    mode_var = tk.StringVar(value="fullport")

    mode_frame = ttk.LabelFrame(frame, text="Trading Mode", padding=8)
    mode_frame.pack(fill="x", pady=(0, 8))

    ttk.Radiobutton(mode_frame, text="Fullport Challenge — Base 0.50/1.90 + Trend 1.00/1.00, ~$1k risk",
                     variable=mode_var, value="fullport").pack(anchor="w", pady=2)
    ttk.Radiobutton(mode_frame, text="Funded Payout — Base 0.50/1.90 + Trend 1.00/1.00, 3 MNQ",
                     variable=mode_var, value="funded").pack(anchor="w", pady=2)
    ttk.Radiobutton(mode_frame, text="Custom",
                     variable=mode_var, value="custom").pack(anchor="w", pady=2)

    # Custom settings
    custom_frame = ttk.LabelFrame(frame, text="Settings (editable in Custom mode)", padding=8)
    custom_frame.pack(fill="x", pady=(0, 8))

    # Base TP mult
    tp_frame = ttk.Frame(custom_frame)
    tp_frame.pack(fill="x", pady=2)
    ttk.Label(tp_frame, text="Base TP:", width=16).pack(side="left")
    tp_var = tk.StringVar(value="1.90")
    tp_entry = ttk.Entry(tp_frame, textvariable=tp_var, width=8)
    tp_entry.pack(side="left")
    ttk.Label(tp_frame, text="(SL always 0.50x, ADX 15-30)", style="Sub.TLabel").pack(side="left", padx=8)

    # Trend SL mult
    tsl_frame = ttk.Frame(custom_frame)
    tsl_frame.pack(fill="x", pady=2)
    ttk.Label(tsl_frame, text="Trend SL:", width=16).pack(side="left")
    tsl_var = tk.StringVar(value="1.00")
    tsl_entry = ttk.Entry(tsl_frame, textvariable=tsl_var, width=8)
    tsl_entry.pack(side="left")
    ttk.Label(tsl_frame, text="(ADX 30+, STD1 band)", style="Sub.TLabel").pack(side="left", padx=8)

    # Trend TP mult
    ttp_frame = ttk.Frame(custom_frame)
    ttp_frame.pack(fill="x", pady=2)
    ttk.Label(ttp_frame, text="Trend TP:", width=16).pack(side="left")
    ttp_var = tk.StringVar(value="1.00")
    ttp_entry = ttk.Entry(ttp_frame, textvariable=ttp_var, width=8)
    ttp_entry.pack(side="left")

    # Contracts
    con_frame = ttk.Frame(custom_frame)
    con_frame.pack(fill="x", pady=2)
    ttk.Label(con_frame, text="Contracts:", width=16).pack(side="left")
    con_var = tk.StringVar(value="dynamic")
    con_entry = ttk.Entry(con_frame, textvariable=con_var, width=8)
    con_entry.pack(side="left")
    ttk.Label(con_frame, text="('dynamic' or number)", style="Sub.TLabel").pack(side="left", padx=8)

    # Target risk
    risk_frame = ttk.Frame(custom_frame)
    risk_frame.pack(fill="x", pady=2)
    ttk.Label(risk_frame, text="Target Risk $:", width=16).pack(side="left")
    risk_var = tk.StringVar(value="1000")
    risk_entry = ttk.Entry(risk_frame, textvariable=risk_var, width=8)
    risk_entry.pack(side="left")
    ttk.Label(risk_frame, text="(for dynamic sizing)", style="Sub.TLabel").pack(side="left", padx=8)

    # Account + Instrument
    acct_frame = ttk.LabelFrame(frame, text="NT8 Connection", padding=8)
    acct_frame.pack(fill="x", pady=(0, 8))

    inst_row = ttk.Frame(acct_frame)
    inst_row.pack(fill="x", pady=2)
    ttk.Label(inst_row, text="Instrument:", width=16).pack(side="left")
    inst_var = tk.StringVar(value="MNQ 06-26")
    ttk.Entry(inst_row, textvariable=inst_var, width=16).pack(side="left")

    acct_row = ttk.Frame(acct_frame)
    acct_row.pack(fill="x", pady=2)
    ttk.Label(acct_row, text="Account:", width=16).pack(side="left")
    acct_var = tk.StringVar(value="Sim101")
    ttk.Entry(acct_row, textvariable=acct_var, width=16).pack(side="left")

    # Second account (optional) — Lucid 2-account rotation
    acct2_row = ttk.Frame(acct_frame)
    acct2_row.pack(fill="x", pady=2)
    ttk.Label(acct2_row, text="Account 2:", width=16).pack(side="left")
    acct2_var = tk.StringVar(value="Sim102")
    ttk.Entry(acct2_row, textvariable=acct2_var, width=16).pack(side="left")

    rot_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(acct_frame, text="Enable 2-account rotation (Lucid: alternate trades Acct 1 / Acct 2)",
                    variable=rot_var).pack(anchor="w", pady=(4, 0))

    # Options
    opt_frame = ttk.Frame(frame)
    opt_frame.pack(fill="x", pady=(0, 8))

    paper_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(opt_frame, text="Paper mode (no orders sent to NT8)", variable=paper_var).pack(anchor="w")

    # Update fields when mode changes
    def update_risk_enabled():
        """target_risk is only relevant when contracts is dynamic."""
        if mode_var.get() != "custom":
            return
        is_dynamic = con_var.get().strip().lower() == "dynamic"
        risk_entry.config(state="normal" if is_dynamic else "disabled")

    def on_mode_change(*_):
        mode = mode_var.get()
        is_custom = mode == "custom"
        state = "normal" if is_custom else "disabled"
        tp_entry.config(state=state)
        tsl_entry.config(state=state)
        ttp_entry.config(state=state)
        con_entry.config(state=state)
        risk_entry.config(state=state)

        if not is_custom and mode in PRESETS:
            preset = PRESETS[mode]
            tp_var.set(f"{preset['tp_mult']:.2f}")
            tsl_var.set(f"{preset['trend_sl_mult']:.2f}")
            ttp_var.set(f"{preset['trend_tp_mult']:.2f}")
            if preset["contracts"] is None:
                con_var.set("dynamic")
            else:
                con_var.set(str(preset["contracts"]))
            risk_var.set(str(int(preset["target_risk"])))

        # In custom mode, honor contracts-vs-target_risk dependency
        if is_custom:
            update_risk_enabled()

    mode_var.trace_add("write", on_mode_change)
    con_var.trace_add("write", lambda *_: update_risk_enabled())
    on_mode_change()  # set initial state

    # Buttons
    btn_frame = ttk.Frame(frame)
    btn_frame.pack(fill="x", pady=(4, 0))

    def on_start():
        mode = mode_var.get()
        try:
            tp = float(tp_var.get())
        except ValueError:
            tp = 1.90

        try:
            trend_sl = float(tsl_var.get())
        except ValueError:
            trend_sl = 1.00

        try:
            trend_tp = float(ttp_var.get())
        except ValueError:
            trend_tp = 1.00

        con_str = con_var.get().strip().lower()
        if con_str == "dynamic":
            contracts = None
        else:
            try:
                contracts = int(con_str)
            except ValueError:
                contracts = None

        try:
            target_risk = float(risk_var.get())
        except ValueError:
            target_risk = 1000.0

        acct1 = acct_var.get().strip()
        acct2 = acct2_var.get().strip()
        accounts_list = None
        if rot_var.get() and acct2:
            accounts_list = [acct1, acct2]

        result[0] = TradingConfig(
            mode=mode,
            sl_mult=0.50,
            tp_mult=tp,
            trend_sl_mult=trend_sl,
            trend_tp_mult=trend_tp,
            instrument=inst_var.get().strip(),
            account=acct1,
            contracts=contracts,
            target_risk=target_risk,
            paper=paper_var.get(),
            accounts=accounts_list,
        )
        root.destroy()

    def on_cancel():
        root.destroy()

    ttk.Button(btn_frame, text="Start Trading", command=on_start).pack(side="left", padx=(0, 8))
    ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="left")

    # Info label
    info = ttk.Label(frame, text="Base: VWAP zone (ADX 15-30) | Trend: STD1 band (ADX 30+)", style="Sub.TLabel")
    info.pack(anchor="w", pady=(8, 0))

    root.mainloop()
    return result[0]


if __name__ == "__main__":
    config = show_config_gui()
    if config:
        print(f"Mode: {config.mode}")
        print(f"SL: {config.sl_mult} / TP: {config.tp_mult}")
        print(f"Instrument: {config.instrument}")
        print(f"Account: {config.account}")
        print(f"Contracts: {config.contracts or 'dynamic'}")
        print(f"Target risk: ${config.target_risk:.0f}")
        print(f"Paper: {config.paper}")
    else:
        print("Cancelled")
