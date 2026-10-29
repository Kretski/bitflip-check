#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Dimitar Kretski
"""make_report.py - turns a bitflip_test.py result file (.jsonl) into a one-document Markdown report.

    python make_report.py results/mnv2_audit.jsonl                          -> mnv2_audit_report.md
    python make_report.py my_audit.jsonl --overhead bitflip_overhead.jsonl --client "Example Ltd" --out report.md

The report states what was tested (hashes), against what baseline (no protection), the numbers with intervals,
the paired comparison on the SAME flips, the limits of the method, and how to repeat the run.
It only restates what is in the file; it makes no claim beyond it. Standard library only.
"""
import argparse, datetime, json, math, os, re, sys


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def sign_test(b, c):
    n = b + c
    if n == 0: return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n)


def tool_version(here):
    try:
        head = open(os.path.join(here, "bitflip_test.py"), encoding="utf-8").read(400)
        m = re.search(r"bitflip_test\.py (v[0-9.]+)", head)
        return m.group(1) if m else "unknown"
    except OSError:
        return "unknown (bitflip_test.py not found next to this script)"


MNAME = {"none": "no protection (baseline)", "clamp": "clamp (range restriction)", "scrub": "scrub (checksum + golden copy)",
         "tmr": "tmr (3 copies, bitwise vote)", "scrub_scales": "scrub_scales (scales only)", "scrub_scales+clamp": "scrub_scales + clamp"}
ORDER = ["none", "clamp", "scrub_scales", "scrub_scales+clamp", "scrub", "tmr"]
CONSTRUCTION = {"scrub", "tmr"}


def build(rows, overhead_rows, client, src):
    here = os.path.dirname(os.path.abspath(__file__))
    out = []; A = out.append
    cleans = [r for r in rows if r.get("kind") == "clean_fmt"]
    rand = [r for r in rows if r.get("kind") == "rand"]
    if not rand: sys.exit("no random-flip trials in this file; run bitflip_test.py --mode inference first")
    done = [r for r in rows if r.get("kind") == "seed_done"]
    A("# bitflip-check report")
    A("")
    A(f"- Result file: `{os.path.basename(src)}`")
    A(f"- Generated: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} with `make_report.py`; test run by `bitflip_test.py` {tool_version(here)}")
    if client: A(f"- Prepared for: {client}")
    A("- Status: research-grade measurement on simulated storage faults. Not a safety assessment, certification or guarantee.")
    A("")
    archs = sorted({r.get("arch", "smallcnn") for r in rand})
    for arch in archs:
        ra = [r for r in rand if r.get("arch", "smallcnn") == arch]
        ca = [r for r in cleans if r.get("arch", "smallcnn") == arch]
        A(f"## Model: `{arch}`")
        A("")
        hs = {}
        for r in ca:
            for k in ("model_def_sha256", "weights_sha256", "model_file_sha256", "data_sha256"):
                if r.get(k): hs[k] = r[k]
        if hs:
            A("Artifacts tested (SHA-256):")
            A("")
            for k, v in hs.items(): A(f"- {k.replace('_sha256', '')}: `{v}`")
            A("")
        else:
            A("No artifact hashes recorded in this file (hashes are written by bitflip_test.py v2.10 or later for your own models)." if str(arch).startswith("custom:")
              else "Built-in model trained by `bitflip_test.py`; no external artifacts.")
            A("")
        for fmt in sorted({r["fmt"] for r in ra}):
            rf = [r for r in ra if r["fmt"] == fmt]
            cl = {r["seed"]: r["clean"] for r in ca if r["fmt"] == fmt}
            full = next((r.get("clean_fp32_full") for r in ca if r["fmt"] == fmt and r.get("clean_fp32_full") is not None), None)
            keffs = sorted({(r["k"], r.get("k_eff", r["k"])) for r in rf})
            nseeds = len({r["seed"] for r in rf})
            A(f"### Format `{fmt}`")
            A("")
            A(f"- Clean accuracy of this format on the evaluation subset: " + ", ".join(f"{v:.2f}%" for v in cl.values())
              + (f" (float32 model on the full evaluation set: {full:.2f}%)" if full is not None else ""))
            A("- Flips per memory image: " + ", ".join(f"reference k={k} -> {ke} used" for k, ke in keffs)
              + f"; trained models in this file: {nseeds}")
            A("")
            for k_ref in sorted({r["k"] for r in rf}):
                rk = [r for r in rf if r["k"] == k_ref]
                by = {}
                for r in rk: by.setdefault(r["method"], {})[(r["seed"], r["trial"])] = r
                A(f"Random flips, k = {k_ref} (accuracy drop measured against this format's clean accuracy):")
                A("")
                A("| Method | Trials | Mean drop (pts) | P(drop > 5 pts) | 95% CI | P(collapse, acc <= 15%) |")
                A("|---|---|---|---|---|---|")
                for m in [m for m in ORDER if m in by]:
                    v = list(by[m].values()); n = len(v)
                    drops = [r["clean"] - r["acc"] for r in v]; big = sum(1 for d in drops if d > 5)
                    lo, hi = wilson(big, n); col = sum(1 for r in v if r["acc"] <= 15) / n
                    star = " *" if m in CONSTRUCTION else ""
                    A(f"| {MNAME[m]}{star} | {n} | {sum(drops)/n:.2f} | {big/n*100:.1f}% | [{lo*100:.1f}, {hi*100:.1f}] | {col*100:.1f}% |")
                A("")
                if any(m in CONSTRUCTION for m in by):
                    A("\\* `scrub` and `tmr` restore the clean model by construction, under their assumptions (fault-free golden copy; no coincident flips of the same bit in `tmr`). A 0% there is not a measured fault-detection rate.")
                    A("")
                if "none" in by:
                    prs = []
                    for m in ("clamp", "scrub_scales"):
                        if m not in by: continue
                        keys = by["none"].keys() & by[m].keys()
                        b = sum(1 for q in keys if (by["none"][q]["clean"] - by["none"][q]["acc"]) > 5 and not (by[m][q]["clean"] - by[m][q]["acc"]) > 5)
                        c = sum(1 for q in keys if not (by["none"][q]["clean"] - by["none"][q]["acc"]) > 5 and (by[m][q]["clean"] - by[m][q]["acc"]) > 5)
                        prs.append((m, b, c, len(keys), sign_test(b, c)))
                    if prs:
                        A("Paired comparison on the same flips (trials with a drop above 5 pts that a protection removes or adds; exact two-sided sign test):")
                        A("")
                        A("| Protection | Removed | Added | Trials | p |")
                        A("|---|---|---|---|---|")
                        for m, b, c, n, p in prs: A(f"| {MNAME[m]} | {b} | {c} | {n} | {p:.3f} |")
                        A("")
                        if nseeds < 3:
                            A(f"This file contains {nseeds} trained model(s); the interval and p-value describe flip-to-flip noise of that model only, not variation between trainings.")
                            A("")
                if nseeds >= 2 and "none" in by:
                    A("Per trained model, P(drop > 5 pts), none / " + " / ".join(m for m in ("clamp", "scrub_scales") if m in by) + ":")
                    A("")
                    for sd in sorted({r["seed"] for r in rk}):
                        vals = []
                        for m in ["none"] + [m for m in ("clamp", "scrub_scales") if m in by]:
                            vv = [r for (s, _), r in by[m].items() if s == sd]
                            vals.append(f"{sum(1 for r in vv if r['clean'] - r['acc'] > 5) / max(1, len(vv)):.3f}")
                        A(f"- model {sd}: " + " / ".join(vals))
                    A("")
        if done:
            d = done[-1]
            A("### Reproduce")
            A("")
            A("```")
            if str(arch).startswith("custom:"):
                ms = ("--model-file FILE.pt" if hs.get("model_file_sha256") else "--model-def FILE.py:FUNC --weights FILE.pt") + " --data-npz FILE.npz"
            else:
                ms = f"--arch {arch}"
            fm = ",".join(d.get("fmts", []))
            ntr = max((len([1 for r in ra if r["fmt"] == f and r["method"] == "none" and r["k"] == d["ks"][0]]) for f in {r["fmt"] for r in ra}), default=0) // max(1, len({r["seed"] for r in ra})) if d.get("ks") else 0
            A(f"python bitflip_test.py --mode inference {ms} --formats {fm} {'--normalize-bits ' if d.get('norm') else ''}"
              f"{'--skip-bit-sweep ' if not d.get('sweep', True) else ''}--ks {','.join(map(str, d.get('ks', [])))} --trials {ntr} --out repeat.jsonl")
            A("python bitflip_test.py --analyze repeat.jsonl")
            A("```")
            A("")
            A("Replace the model placeholders with the same files (the SHA-256 values above identify them). Flip positions are random: a repeat gives different flips, so individual percentages differ by the noise shown in the intervals.")
            A("")
    if overhead_rows:
        A("## Cost of the protections (this machine, NumPy implementation)")
        A("")
        r0 = overhead_rows[0]
        A(f"Forward pass, batch 1: {r0['fwd_b1_ms']:.3f} ms. Times are for this script on one machine, not a prediction for target hardware.")
        A("")
        A("| Format | Operation | Median ms | x forward | Extra memory |")
        A("|---|---|---|---|---|")
        for r in overhead_rows:
            A(f"| {r['fmt']} | {r['op']} | {r['median_ms']:.3f} | {r['median_ms']/r['fwd_b1_ms']:.1f}x | {r['extra_bytes']/1024:.1f} KB ({100*r['extra_bytes']/max(1, r['image_bytes']):.1f}%) |")
        A("")
    A("## Limits of this measurement")
    A("")
    for t in ("Storage faults only: stored weights and float buffers are corrupted in the stated format and decoded to float32 for compute. No faults in activations, accumulators, compute units, control logic or instruction memory.",
              "`k` is the number of flips per memory image, not a physical fault rate. No ECC behaviour or radiation environment of any specific platform is modelled; hardware with ECC corrects single-bit flips before they reach the model.",
              "With `--normalize-bits`, every format receives the same number of flips per stored bit (a smaller image receives fewer flips).",
              "Intervals describe the randomness of the flips for the model(s) in this file. They do not cover variation between different trainings, architectures or datasets.",
              "`scrub` and `tmr` hold only under their stated assumptions.",
              "Timings come from a NumPy implementation on one machine.",
              "This is not a safety, reliability or certification statement."):
        A(f"- {t}")
    A("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Markdown report from a bitflip_test.py .jsonl result file")
    ap.add_argument("jsonl"); ap.add_argument("--overhead", default=None, help="optional overhead .jsonl from --mode overhead")
    ap.add_argument("--client", default=None, help="name shown under 'Prepared for' (optional)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.jsonl, encoding="utf-8-sig") if l.strip()]
    ov = [json.loads(l) for l in open(a.overhead, encoding="utf-8-sig") if l.strip()] if a.overhead else []
    ov = [r for r in ov if r.get("mode") == "overhead"]
    text = build(rows, ov, a.client, a.jsonl)
    out = a.out or os.path.splitext(a.jsonl)[0] + "_report.md"
    open(out, "w", encoding="utf-8").write(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
