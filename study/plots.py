"""Plots & summary for a PH study run.

Reads a run dir and produces:
  convergence.png        min/avg fitness (mean MSE) per generation
  method_perf.png        best & mean MSE per point-cloud / filtration / vectorizer
  population_monitor.png  which methods populate each generation (stacked fractions)
  numeric_trends.png     mean+/-std of numeric genes over generations
  summary.md             best pipeline (+ MSE/MAE/DA/R2/time), method rankings,
                         parameter tendencies, and PH-vs-no-PH-benchmark verdict
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CAT = ["point_cloud", "filtration", "vectorizer", "homology_dim"]
NUM = ["window", "dimension", "delay", "stride", "resolution"]


def _load(run):
    run = Path(run)
    gens = [json.loads(l) for l in (run / "generations.jsonl").read_text().splitlines()]
    evals = [json.loads(l) for l in (run / "evals.jsonl").read_text().splitlines()]
    return run, gens, evals


def plot_improvement(run, gens, max_gen=50):
    """Concise, formal improvement curve: % reduction of best mean-MSE vs generation 0."""
    g = [d["gen"] for d in gens]
    base = gens[0]["min"]
    imp = [(base - d["min"]) / base * 100.0 for d in gens]
    plt.figure(figsize=(6.5, 4))
    plt.plot(g, imp, "-o", color="#1b6b3a", lw=2, ms=4)
    plt.xlim(0, max_gen); plt.ylim(bottom=0)
    plt.xlabel("generation"); plt.ylabel("MSE reduction vs. generation 0  (%)")
    plt.title("Evolutionary improvement")
    plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(run / "improvement.png", dpi=130); plt.close()


def plot_vs_baseline(run, gens, max_gen=50):
    """Improvement of the best PH config over the no-PH mean baseline, in %.
    Positive = below the benchmark; the shaded band is the benchmark's +/-1 std
    (run-to-run noise) — only crossing above it is a genuine improvement."""
    bpath = Path(run) / "benchmark.json"
    if not bpath.exists():
        return
    bm = json.loads(bpath.read_text())["summary"]["mse"]
    base, sd = bm["mean"], bm["std"]
    g = [d["gen"] for d in gens]
    imp = [(base - d["min"]) / base * 100.0 for d in gens]
    band = sd / base * 100.0
    plt.figure(figsize=(6.5, 4))
    plt.axhspan(-band, band, color="tab:gray", alpha=0.18, label="no-PH baseline ±1σ (noise)")
    plt.axhline(0.0, color="tab:gray", ls="--", lw=1.2)
    plt.plot(g, imp, "-o", color="#1b4d7a", lw=2, ms=4, label="best PH vs no-PH")
    plt.xlim(0, max_gen)
    plt.xlabel("generation"); plt.ylabel("MSE reduction vs. no-PH baseline  (%)")
    plt.title("Improvement over the no-PH baseline"); plt.grid(alpha=0.3); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(run / "improvement_vs_baseline.png", dpi=130); plt.close()


def plot_convergence(run, gens):
    g = [d["gen"] for d in gens]
    plt.figure(figsize=(7, 4))
    plt.plot(g, [d["min"] for d in gens], "-o", label="best (min)", color="tab:green")
    plt.plot(g, [d["avg"] for d in gens], "-s", label="population avg", color="tab:blue")
    plt.fill_between(g, [d["avg"] - d["std"] for d in gens], [d["avg"] + d["std"] for d in gens],
                     alpha=0.15, color="tab:blue")
    plt.xlabel("generation"); plt.ylabel("fitness = mean test MSE (lower=better)")
    plt.title("Evolutionary convergence"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(run / "convergence.png", dpi=120); plt.close()


def _perf(evals, gene):
    by = defaultdict(list)
    for e in evals:
        by[str(e["genome"].get(gene))].append(e["mse"])
    methods = sorted(by, key=lambda m: np.min(by[m]))
    return methods, by


def plot_method_perf(run, evals):
    genes = ["point_cloud", "filtration", "vectorizer"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ranks = {}
    for ax, gene in zip(axes, genes):
        methods, by = _perf(evals, gene)
        best = [np.min(by[m]) for m in methods]; mean = [np.mean(by[m]) for m in methods]
        x = np.arange(len(methods))
        ax.bar(x - 0.2, best, 0.4, label="best", color="tab:green")
        ax.bar(x + 0.2, mean, 0.4, label="mean", color="tab:orange")
        ax.set_xticks(x); ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax.set_title(gene); ax.set_ylabel("test MSE"); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
        ranks[gene] = {m: dict(best=float(np.min(by[m])), mean=float(np.mean(by[m])), n=len(by[m])) for m in methods}
    plt.suptitle("Which PH methods perform best (by test MSE)")
    plt.tight_layout(); plt.savefig(run / "method_perf.png", dpi=120); plt.close()
    return ranks


def plot_population_monitor(run, gens):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, gene in zip(axes.ravel(), CAT):
        vals = sorted({str(ind["genome"][gene]) for d in gens for ind in d["population"]})
        series = {v: [] for v in vals}
        for d in gens:
            c = Counter(str(ind["genome"][gene]) for ind in d["population"]); tot = sum(c.values())
            for v in vals:
                series[v].append(c.get(v, 0) / tot)
        gnum = [d["gen"] for d in gens]
        ax.stackplot(gnum, [series[v] for v in vals], labels=vals, alpha=0.85)
        ax.set_title(gene); ax.set_xlabel("generation"); ax.set_ylabel("pop. fraction")
        ax.legend(fontsize=7, loc="upper left", ncol=2)
    plt.suptitle("Population monitor: which methods fill each generation")
    plt.tight_layout(); plt.savefig(run / "population_monitor.png", dpi=120); plt.close()


def plot_numeric_trends(run, gens):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, gene in zip(axes.ravel(), NUM):
        means = [np.mean([float(i["genome"][gene]) for i in d["population"]]) for d in gens]
        stds = [np.std([float(i["genome"][gene]) for i in d["population"]]) for d in gens]
        gnum = [d["gen"] for d in gens]; means, stds = np.array(means), np.array(stds)
        ax.plot(gnum, means, "-o", color="tab:purple")
        ax.fill_between(gnum, means - stds, means + stds, alpha=0.2, color="tab:purple")
        ax.set_title(gene); ax.set_xlabel("generation")
    axes.ravel()[-1].axis("off")
    plt.suptitle("Numeric gene convergence (mean +/- std)")
    plt.tight_layout(); plt.savefig(run / "numeric_trends.png", dpi=120); plt.close()


def write_summary(run, gens, evals, ranks):
    best = min(evals, key=lambda e: e["mse"])
    L = ["# PH study summary\n",
         f"- generations: {len(gens)-1}, unique evaluations: {len(evals)}",
         f"- **best fitness (mean test MSE): {best['mse']:.4f}**",
         f"- best metrics: MAE {best['mae']:.4f} · DA {best['da']:.3f} · R2 {best['r2']:.4f} · time {best['seconds']}s",
         f"- best genome: `{best['genome']}`\n"]
    # PH vs no-PH benchmark
    bpath = Path(run) / "benchmark.json"
    if bpath.exists():
        b = json.loads(bpath.read_text())["summary"]
        d = (b["mse"]["mean"] - best["mse"]) / b["mse"]["mean"] * 100
        L.append("## PH vs no-PH benchmark (10 runs, raw OHLCV)")
        L.append(f"- no-PH: MSE {b['mse']['mean']:.4f}±{b['mse']['std']:.4f} · MAE {b['mae']['mean']:.4f} "
                 f"· DA {b['da']['mean']:.3f} · R2 {b['r2']['mean']:.4f}")
        L.append(f"- best PH: MSE {best['mse']:.4f} -> PH **{'helps' if best['mse']<b['mse']['mean'] else 'does NOT help'}** "
                 f"({d:+.1f}% vs no-PH)\n")
    for gene in ["point_cloud", "filtration", "vectorizer"]:
        L.append(f"## {gene} ranking (by best MSE)")
        L.append("| method | best MSE | mean MSE | #evals |"); L.append("|---|--:|--:|--:|")
        for m, s in sorted(ranks[gene].items(), key=lambda kv: kv[1]["best"]):
            L.append(f"| {m} | {s['best']:.4f} | {s['mean']:.4f} | {s['n']} |")
        L.append("")
    last = gens[-1]["population"]
    L.append("## Final-generation tendencies")
    for gene in NUM:
        xs = [float(i["genome"][gene]) for i in last]
        L.append(f"- {gene}: mean {np.mean(xs):.1f} (range {min(xs):.0f}-{max(xs):.0f})")
    for gene in CAT:
        top = Counter(str(i["genome"][gene]) for i in last).most_common(1)[0]
        L.append(f"- {gene}: most common = {top[0]} ({top[1]}/{len(last)})")
    (run / "summary.md").write_text("\n".join(L), encoding="utf-8")
    return best


def make_all(run):
    run, gens, evals = _load(run)
    plot_improvement(run, gens)
    plot_vs_baseline(run, gens)
    plot_convergence(run, gens)
    ranks = plot_method_perf(run, evals)
    plot_population_monitor(run, gens)
    plot_numeric_trends(run, gens)
    best = write_summary(run, gens, evals, ranks)
    print(f"Plots + summary -> {run}")
    print(f"Best MSE {best['mse']:.4f}  {best['genome']['point_cloud']}/{best['genome']['filtration']}/{best['genome']['vectorizer']}")
    return run


if __name__ == "__main__":
    make_all(sys.argv[1] if len(sys.argv) > 1 else "study/runs/latest")
