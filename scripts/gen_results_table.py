"""Regenerate the LIVE results status board in site/index.html from outputs/results/*.json + logs.
Self-updating: reads each metric file, decides done / running / planned, splices the table between
<!--RESULTS_TABLE_START--> and <!--RESULTS_TABLE_END-->. Re-run any time (also on a */30 cron):
    python scripts/gen_results_table.py
No GPU, no deps beyond stdlib."""
import os, json, time, glob

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "outputs/results")
LOGD = os.path.join(ROOT, "outputs/logs")

def have(f): return os.path.exists(os.path.join(RES, f))
def load(f):
    try: return json.load(open(os.path.join(RES, f)))
    except Exception: return None
def log_fresh(name, mins=25):
    p = os.path.join(LOGD, name)
    return os.path.exists(p) and (time.time() - os.path.getmtime(p)) < mins * 60
def running(name): return any(name in os.path.basename(p) for p in glob.glob("/proc/*/cmdline") if _cmd_has(p, name))
def _cmd_has(proc, name):
    try: return name in open(proc, "rb").read().decode("utf-8", "ignore")
    except Exception: return False

PILL = {"done": '<span class="pill done">done</span>',
        "running": '<span class="pill run">running</span>',
        "planned": '<span class="pill plan">planned</span>',
        "cut": '<span class="pill plan">cut</span>'}

def tr(label, desc, status, frag=False):
    c = ' class="frag"' if frag else ''
    return (f'      <tr{c}><td style="text-align:left">{label}</td>'
            f'<td style="text-align:left">{desc}</td><td>{PILL[status]}</td></tr>')

out = []

# ---- A: law-led core (analysis) ----
out.append('  <h3>A · The law-led core (analysis)</h3>')
out.append('  <div class="tbl"><table><thead><tr><th>Result</th><th>Finding</th><th>Status</th></tr></thead><tbody>')
for lab, desc, f in [
    ("Per-organ fragility (Dice R1&#8594;R8)", "tail organs collapse: adrenal &#8722;0.20, gallbladder &#8722;0.17 vs liver &#8722;0.02", "m0_fragility_dice.json"),
    ("Spectral fragility law (centroid &#8594; drop)", "r = 0.86 MRISeg &#183; 0.855 AMOS &#183; 0.841 &#177; 0.023 (5-fold CV)", "m0_law_v2.json"),
    ("Selection-corrected + cross-pipeline transfer", "centroid = TOP of 5 predictors (permutation-corrected p = 0.0015 / 0.0053); the law calibrated on one dataset predicts the OTHER&#8217;s per-organ drop <b>values</b> at R&#178; = 0.67&#8211;0.74 (2 scanners &#215; 2 seg pipelines &#8212; honest calibrated test, not the rank-artifact version)", "reconcile_stats.json"),
    ("Not just size / blur (predictor rank)", "centroid 0.86 &#8811; SA/V 0.85 &gt; size 0.66 &gt; blur 0.30", "debunk_obvious.json"),
    ("Double-jeopardy mechanism (Parseval)", "energy&#8594;error r = 0.978 real knee (44 cases) &#183; 0.871 sim abdominal", "skmtea_law_multicase.json"),
    ("R* safe-acceleration limits", "adrenals R* &#8776; 4.2&#8211;4.7 vs liver / kidney 8.5&#215;+", "m0_rstar.json"),
]:
    out.append(tr(lab, desc, "done" if have(f) else "planned"))
out.append('    </tbody></table></div>')

# ---- B: method / mitigation (real knee, R8) ----
out.append('  <h3>B · Method / mitigation (real knee k-space, R8)</h3>')
out.append('  <div class="tbl"><table><thead><tr><th>Experiment</th><th>Result</th><th>Status</th></tr></thead><tbody>')
out.append(tr("Mixed-R mitigation",
              "+0.134 vs classical-CS recon (p=2e-12, n=42); <b>+0.028 vs fair learned recon</b>",
              "done" if have("condseg_knee_full.json") else "planned"))
d = load("fgtdr.json")
if d:
    m = d["mean"]
    out.append(tr(f"FG-TDR 4-arm ablation (n={d.get('n_percase','?')})",
                  f"task {m['task_adapted']:.3f} &#183; recon-then-seg {m['recon_then_seg']:.3f} &#183; "
                  f"FG-TDR {m['FGTDR']:.3f} &#183; mixed-R <b>{m['mixedR']:.3f}</b>", "done"))
else:
    out.append(tr("FG-TDR 4-arm ablation", "&#8212;", "planned"))
out.append(tr("W(k) audit &#8212; region vs boundary prior",
              "aim ratio: region 0.71 (mis-pointed) &#8594; boundary <b>4.03</b> (5.7&#215; better)", "done"))
# --- the live one: FG-TDR boundary ablation ---
db = load("fgtdr_bnd.json")
if db:
    m = db["mean"]; vf = db.get("FGTDRbnd_vs_FGTDR", {}); vm = db.get("FGTDRbnd_vs_mixedR", {})
    def p(x): return f"{x:.1e}" if isinstance(x, (int, float)) else "n/a"
    desc = (f"task {m['task_adapted']:.3f} &#183; region {m['FGTDR']:.3f} &#183; <b>boundary {m['FGTDR_bnd']:.3f}</b> &#183; "
            f"mixed-R {m['mixedR']:.3f} &nbsp;|&nbsp; bnd&#8722;region {vf.get('delta',0):+.3f} (p={p(vf.get('p'))}) &#183; "
            f"bnd&#8722;mixedR {vm.get('delta',0):+.3f} (p={p(vm.get('p'))})")
    st = "done"
else:
    part = load("fgtdr_bnd_partial.json")
    n = len(part.get("seeds_done", [])) if part else 0
    if part or log_fresh("fgtdr_bnd.log") or running("condseg_knee_fgtdr"):
        desc = f"running &#8212; {n}/2 seeds checkpointed (auto-fills on completion)"; st = "running"
    else:
        desc = "region &#8212; &#183; boundary &#8212; &#183; vs mixed-R &#8212;"; st = "planned"
out.append(tr("FG-TDR boundary ablation (region-W vs boundary-W)", desc, st, frag=True))
out.append(tr("Full ablation ladder (7 arms, budget-matched)",
              "<b>CUT for the sprint</b> &#8212; ~18h GPU to sharpen a method (FG-TDR) that is already a documented NEGATIVE; GPU time went to the law/mechanism instead",
              "done" if have("ladder.json") else "cut"))
# --- FG-Seg make-or-break (the decisive novelty control) ---
dc = load("fgseg_control.json")
if dc:
    mb = dc.get("MAKE_OR_BREAK_centroid_vs_drop", {}); tm = dc.get("tail_mean", {})
    def pf(x): return f"{x:.1e}" if isinstance(x, (int, float)) else "n/a"
    desc = (f"tail Dice: <b>law {tm.get('centroid',0):.3f}</b> vs worst-organ {tm.get('drop',0):.3f} vs uniform {tm.get('uniform',0):.3f} "
            f"&nbsp;|&nbsp; <b>law&#8722;control {mb.get('delta_tail',0):+.3f}</b> (p={pf(mb.get('p'))})")
    stc = "done"
elif load("fgseg_control_partial.json") or log_fresh("queue.log"):
    desc = "running &#8212; law (centroid) vs worst-organ (drop) vs uniform weighting (auto-fills)"; stc = "running"
else:
    desc = "law-predicted (centroid) vs empirical worst-organ weighting &#8212; decides if the fragility-law novelty survives"; stc = "planned"
out.append(tr("&#9733; FG-Seg make-or-break (law vs worst-organ weighting)", desc, stc, frag=True))
out.append('    </tbody></table></div>')

# ---- C: WACV-required ----
out.append('  <h3>C · WACV-required (reviewer-hardened to-do)</h3>')
out.append('  <div class="tbl"><table><thead><tr><th>Experiment</th><th>Why (from the AC review)</th><th>Status</th></tr></thead><tbody>')
# --- the causal probe (final v3, width-matched): an INFORMATIVE null ---
dp = load("intervention_v3_power.json")
if dp:
    desc = (f"within-organ, width-matched full deletion: own-radius vs other-radii &#916;Dice "
            f"<b>{dp['mean_effect']:+.4f}</b>, 95% CI [{dp['ci95'][0]:+.3f}, {dp['ci95'][1]:+.3f}] &#8212; "
            f"excludes any effect &gt; {dp['ci95'][1]:.3f}; TOST equivalence p&lt;0.0001; powered 17&#215; below the "
            f"phenomenon. <b>INFORMATIVE NULL</b> &#8594; the frequency claim is predictive, not causal")
    sti = "done"
elif running("intervention_v3"):
    desc = "running &#8212; width-matched full-deletion causal probe (auto-fills)"; sti = "running"
else:
    desc = "within-organ ablation at own vs other spectral radii, matched energy &#8212; the causal test"; sti = "planned"
out.append(tr("&#9733; Causal frequency probe (informative null)", desc, sti, frag=True))
# --- the loop-closer: mixed-R rescues the fragile organs ---
out.append(tr("Mixed-R closes the loop (2 anatomies)",
              "abdominal nnU-Net mixed-R <b>+0.088</b> tail Dice @R8 (p=4&#215;10&#8315;&#8308;&#185;, 239/240) &#8212; but only <b>+0.019 vs a matched R8-trained baseline</b> (Focal-Tversky edges it); knee <b>+0.134</b> &#183; "
              "per-organ gain vs centroid &#961;=<b>0.824</b> (weakest-organ regularization) &#8594; helps the worst organs most",
              "done" if have("abdominal_mixedr_perorgan.json") else "planned", frag=True))
for lab, desc, f in [
    ("Frequency vs size (honest decomposition)", "freq-family is the largest variance block (freq-PC1 46&#8211;50% &gt; size 34&#8211;40% &gt; contrast); but centroid&#8217;s effect <b>over size</b> is METHOD-SENSITIVE &#8212; Pearson partial sig (p=0.008) yet rank-based Spearman &#961;=0.41/0.46 <b>n.s.</b> (p=0.19). Downgraded to &#8216;directional, entangled with size at n=13&#8217; (disclosed, not hidden)", "reconcile_stats.json"),
    ("Real-knee law replication <span class=\"src\">(already done)</span>", "SKM-TEA knee: centroid&#8594;drop fit exists (n=6) &#8212; but the per-structure law <b>inverts</b> on knee (patellar); only the mechanism transfers", "knee_law.json"),
    ("Frequency-sensitivity probe (causal, informative null)", "the within-organ equal-energy k-space ablation IS this probe: own vs other radius &#916;Dice +0.0013, TOST-equivalent-to-zero &#8594; predictive, not causal", "intervention_v3_power.json"),
    ("Whole-body generalization (TotalSeg-MRI, 41 structures)", "full-res law &#961;=0.532 (p=3.4e-4); W1 partial centroid&#8594;drop | difficulty <b>0.389</b> (perm p=0.0067), 0.66 gradual &#8212; dissociates from difficulty on the whole body (full-res only)", "totalseg_law_fullres.json"),
    ("Per-structure &#215; R matrix", "full organ&#215;R&#215;method table + Cohen's d + bootstrap CI + Holm/BH (pre-registered fragile set)", "perstruct_matrix.json"),
]:
    out.append(tr(lab, desc, "done" if have(f) else "planned"))
out.append('    </tbody></table></div>')

# ---- D: reviewer-hardening pass (2026-07-27, disk-verified) ----
out.append('  <h3>D &#183; Reviewer-hardening pass (2026-07-27, disk-verified)</h3>')
out.append('  <div class="tbl"><table><thead><tr><th>Result</th><th>Finding</th><th>Status</th></tr></thead><tbody>')
for lab, desc, f, fr in [
    ("Theory: derived energy-budget scalar &#934;_R", "&#934;_R (fraction of organ k-space energy discarded, a-priori) predicts the drop &#961;=<b>0.90</b>, <b>beating</b> the centroid proxy (0.86); provable monotonicity backbone (Parseval + stochastic dominance); Occam kills the multi-factor index (SA/V, contrast add nothing over &#934;_R)", "theory_validate.json", True),
    ("Energy &#8594; TASK(Dice) &#8212; escapes the Parseval tautology", "removed energy predicts <b>Dice</b>, not just image error: pooled &#961;=<b>0.952</b>; within a FIXED R &#961;=<b>0.94</b> at R6 &amp; R8", "energy_task_link.json", False),
    ("Metric blindness is metric-general (not SSIM-only)", "at R8 across 240 cases: PSNR&#8594;Dice &#961;=<b>&#8722;0.205</b>, SSIM&#8594;Dice &#961;=<b>&#8722;0.093</b> &#8212; both &#8776;0 while the metrics vary", "cpu_audit_extras.json", False),
    ("Law confidence intervals (report BOTH)", "honest <b>structure-level</b> bootstrap [0.553, 0.966] (n=13); case-clustered [0.824, 0.879] = subject-sampling noise only, not the headline", "cpu_audit_extras.json", False),
    ("Mixed-R vs a MATCHED baseline", "+0.088 is vs an OOD strawman; <b>+0.019</b> vs a matched R8-trained baseline (Focal-Tversky edges it). Direction bulletproof, magnitude modest", "abdominal_mixedr.json", False),
    ("Simulation-fidelity (physics realism)", "per-organ ordering preserved under a full complex multicoil forward (phase+coils+noise) &#961;=<b>0.978</b>; 4-rung decomposition built + GPU-gated", "complex_compare.json", False),
]:
    out.append(tr(lab, desc, "done" if have(f) else "planned", frag=fr))
out.append('    </tbody></table></div>')

stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime())
out.append(f'  <p class="math-note">Auto-generated from <code>outputs/results/</code> &#183; last refresh {stamp} '
           f'&#183; updates every 30&#8201;min as runs land.</p>')

frag = "\n".join(out)
idx = os.path.join(ROOT, "site/index.html")
html = open(idx, encoding="utf-8").read()
A0, B0 = "<!--RESULTS_TABLE_START-->", "<!--RESULTS_TABLE_END-->"
if A0 not in html or B0 not in html:
    raise SystemExit("markers not found in site/index.html — aborting (no change)")
pre, rest = html.split(A0, 1); _, post = rest.split(B0, 1)
open(idx, "w", encoding="utf-8").write(pre + A0 + "\n" + frag + "\n  " + B0 + post)
print(f"results board updated @ {stamp}")
