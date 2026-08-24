"""W(k) audit figure: the region-mask fragility prior points at LOW frequencies (wrong); the boundary-mask prior
points at the HIGH frequencies thin structures need (the fix). Data from scripts/audit_fgtdr_W*.py. -> W_audit.png"""
import numpy as np, matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"]})
ctr = np.array([0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375])
region = np.array([0.346, 0.435, 0.278, 0.233, 0.203, 0.191, 0.233, 0.394])   # region mask: structure energy-share
bound = np.array([0.110, 0.228, 0.450, 0.650, 0.678, 0.566, 0.445, 0.437])    # boundary mask: structure energy-share
fig, ax = plt.subplots(figsize=(9.2, 5.4))
ax.axvspan(0.25, 0.75, color="#eef4ee", zorder=0)
ax.text(0.5, 0.045, "high spatial frequencies\n(thin structure boundaries)", ha="center", fontsize=9, color="#4a7a55")
ax.plot(ctr, region, "-o", color="#b04a3a", lw=2.4, ms=7, label="region mask  (current prior)")
ax.plot(ctr, bound, "-o", color="#2ca25f", lw=2.6, ms=7, label="boundary mask  (audit fix)")
ax.axhline(region.mean(), color="#b04a3a", ls=":", lw=1.2, alpha=.6)
ax.annotate("region → points LOW\nhigh/low ratio = 0.71  ✗", (0.9375, 0.394), (0.60, 0.30),
            fontsize=9.5, color="#b04a3a", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#b04a3a", lw=1.3))
ax.annotate("boundary → points HIGH\nhigh/low ratio = 4.03  ✓  (5.7× better)", (0.5625, 0.678), (0.36, 0.74),
            fontsize=9.5, color="#1c7a44", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2ca25f", lw=1.3))
ax.set_xlabel("radial frequency  ‖k‖  (0 = DC, 1 = Nyquist)", fontsize=11.5)
ax.set_ylabel("structure's share of k-space energy   $P_{frag}/P_{all}$", fontsize=11.5)
ax.set_xlim(0, 1); ax.set_ylim(0, 0.82); ax.set_axisbelow(True); ax.grid(alpha=.25)
ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=10.5, loc="upper right", frameon=False)
ax.set_title("The fragility prior was aimed the wrong way — and the boundary fix corrects it\n"
             "(real SKM-TEA knee, 148 structure-bearing slices)", fontsize=12.5, fontweight="bold")
fig.tight_layout(); fig.savefig("outputs/plots/W_audit.png", dpi=150, bbox_inches="tight")
print("wrote outputs/plots/W_audit.png")
