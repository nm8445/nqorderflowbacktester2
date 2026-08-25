import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BG="#141414"; FG="#e8e8e8"; ACC="#4ec9b0"; MUT="#9aa0a6"; ORA="#f7a072"
plt.rcParams.update({"mathtext.fontset":"cm","font.family":"DejaVu Sans"})

fig = plt.figure(figsize=(12.6, 13.4)); fig.patch.set_facecolor(BG)
gs = GridSpec(2, 1, height_ratios=[3.15, 1.0], hspace=0.14,
              left=0.045, right=0.965, top=0.945, bottom=0.06)

ax = fig.add_subplot(gs[0]); ax.set_facecolor(BG); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.text(0.0, 1.005, "Rough-Vol algo - the equations", color=ACC, fontsize=23, weight="bold", va="top")
ax.text(0.0, 0.955, r"closed price series $P_t$ on 20-min bars  $\rightarrow$  z-score signal $z_t$", color=MUT, fontsize=12.5, va="top")

rows = [
 (0.86, r"$r_t \;=\; \ln\!\frac{P_t}{P_{t-1}}$",
        "1.  log return of each bar"),
 (0.735, r"$\varepsilon_t \;=\; \frac{\,r_t-\mu_t\,}{\sigma_t}\qquad \mu_t,\,\sigma_t=\mathrm{rolling\ mean,\ std\ over\ }400$",
        "2.  standardize returns into unit 'shocks'"),
 (0.60, r"$K(s)\;=\; s^{\,H-\frac{1}{2}}\;-\;(s-1)^{\,H-\frac{1}{2}}\,,\qquad H=0.40$",
        "3.  the ROUGH kernel  (fractional-BM / rough-Bergomi)"),
 (0.475, r"$X^{H}_t \;=\; \sum_{s=1}^{80} K(s)\,\varepsilon_{t-s}$",
        "4.  convolve shocks with the kernel  (long memory)"),
 (0.35, r"$v_t \;=\; V_0\,\exp\!\left(\eta\,X^{H}_t\right)\,,\qquad V_0=10^{-4},\ \ \eta=1$",
        "5.  rough variance path  (lognormal in the driver)"),
 (0.215, r"$z_t \;=\; \frac{\,v_t-\overline{v}_t\,}{\mathrm{sd}(v)_t}\ \ \Longrightarrow\ \ \mathrm{enter\ if}\ \ z_t>2.0$",
        "6.  z-score the variance (window 75) = the SIGNAL"),
 (0.085, r"$\mathrm{EMA}_t=\alpha\,C_t+(1-\alpha)\,\mathrm{EMA}_{t-1}\,,\ \ \alpha=\frac{2}{81}\qquad \delta_b=V^{\mathrm{buy}}_b-V^{\mathrm{sell}}_b$",
        "7.  EMA-80 = direction bias    .    order-flow delta = absorption filter"),
]
for y, eq, ann in rows:
    ax.text(0.055, y, eq, color=FG, fontsize=20.5, va="center")
    ax.text(0.055, y-0.055, ann, color=MUT, fontsize=11.3, va="center", style="italic")

ax.text(0.055, 0.008, r"stop / target: $\ \pm\,2\times\mathrm{ATR}$   (Wilder RMA of true range)",
        color=ORA, fontsize=12, va="center")
ax.plot([0.0,1.0],[0.93,0.93], color="#333", lw=1)

H, L = 0.40, 80
ks = np.arange(1, L+1, dtype=float)
K = ks**(H-0.5) - (ks-1)**(H-0.5); K[0]=1.0
axk = fig.add_subplot(gs[1]); axk.set_facecolor(BG)
axk.plot(ks, K, color=ACC, lw=2.2, label=r"rough kernel  $K(s)=s^{H-1/2}-(s-1)^{H-1/2}$,  $H=0.40$")
axk.axhline(1/L, color=ORA, lw=1.6, ls="--", label=r"flat moving-average weight $=1/80$")
axk.fill_between(ks, K, 1/L, where=(K>1/L), color=ACC, alpha=0.10)
axk.set_title("what 'rough' does: newest shocks weighted heavily, tail decays as a power law (never to zero)",
              color=FG, fontsize=12.5, pad=8)
axk.set_xlabel("lag  s  (bars back)", color=MUT, fontsize=11)
axk.set_ylabel("weight  K(s)", color=MUT, fontsize=11)
axk.set_ylim(-0.05, 1.05); axk.set_xlim(1, L)
for sp in axk.spines.values(): sp.set_color("#333")
axk.tick_params(colors=MUT, labelsize=9.5); axk.grid(alpha=0.12)
leg = axk.legend(loc="upper right", fontsize=10.5, framealpha=0.0)
for t in leg.get_texts(): t.set_color(FG)

out = r"C:\Users\njchi\AppData\Local\Temp\claude\C--trading-nqorderflowbacktester\3f2f0bc3-2656-41e4-9582-26aff6a8675a\scratchpad\rv_equations.png"
fig.savefig(out, dpi=145, facecolor=BG)
print("saved", out)
