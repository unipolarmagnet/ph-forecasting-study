"""Evolutionary search over persistent-homology pipelines (DEAP), DLinear fitness.

Searches which PH methods + parameters (via phlib) minimise mean test-MSE across the
12 stock datasets. Logs, per run directory:
  results.csv        one row per evaluated individual: key genes + MSE/MAE/DA/R2/time
  evals.jsonl        full record per individual (genome, per-dataset metrics)
  generations.jsonl  per-generation stats + FULL population (which individuals)
  population.csv     flat (gen, individual) table for monitoring the population
  logbook.csv, hall_of_fame.json, config.json, run.log
"""
import argparse
import csv
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
from deap import base, creator, tools

from study import genome as G
from study.fitness import StudyConfig, evaluate

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
# experiment datasets (the 12 sector CSVs are all present in data/; this study uses 3)
DATASETS = ["SPY", "AAPL", "JPM"]

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


def _logger(out):
    lg = logging.getLogger(f"study.{out.name}")
    lg.setLevel(logging.INFO); lg.handlers.clear(); lg.propagate = False
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(out / "run.log", encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt); lg.addHandler(h)
    return lg


def make_toolbox(sc):
    tb = base.Toolbox()
    tb.register("individual", lambda: creator.Individual(G.random_individual()))
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("evaluate", evaluate, sc=sc)
    tb.register("mate", tools.cxUniform, indpb=0.5)
    tb.register("mutate", G.mutate, indpb=0.2)
    tb.register("select", tools.selTournament, tournsize=3)
    return tb


def _stats(pop):
    f = [ind.fitness.values[0] for ind in pop]
    return dict(min=float(np.min(f)), avg=float(np.mean(f)), max=float(np.max(f)), std=float(np.std(f)))


def run(datasets, model="dlinear", pop_size=24, ngen=12, cxpb=0.6, mutpb=0.3, seed=2021,
        epochs=10, seq_len=64, pred_len=5, out_dir=None):
    random.seed(seed); np.random.seed(seed)
    out = Path(out_dir or (ROOT / "study" / "runs" / f"{model}_{time.strftime('%Y%m%d_%H%M%S')}"))
    out.mkdir(parents=True, exist_ok=True)
    log = _logger(out)

    sc = StudyConfig(datasets=datasets, model=model, epochs=epochs, seq_len=seq_len,
                     pred_len=pred_len, seed=seed,
                     log_path=out / "evals.jsonl", results_path=out / "results.csv")
    (out / "config.json").write_text(json.dumps(dict(
        datasets=[Path(d).stem for d in datasets], model=model, pop_size=pop_size, ngen=ngen,
        cxpb=cxpb, mutpb=mutpb, seed=seed, epochs=epochs, seq_len=seq_len, pred_len=pred_len,
        lr=sc.lr, genes=G.NAMES), indent=2))
    log.info(f"START model={model} pop={pop_size} ngen={ngen} datasets={len(datasets)} "
             f"epochs={epochs} seq_len={seq_len} pred_len={pred_len} seed={seed} device={sc.device}")

    tb = make_toolbox(sc)
    hof = tools.HallOfFame(10)
    gen_log = open(out / "generations.jsonl", "w")
    pop_csv = open(out / "population.csv", "w", newline="")
    pw = csv.writer(pop_csv); pw.writerow(["gen", "rank", "fitness_mse", "point_cloud",
                                           "filtration", "vectorizer", "homology_dim", "signature"])
    logbook = tools.Logbook(); logbook.header = ["gen", "nevals", "min", "avg", "max", "std", "gen_seconds"]

    t0 = time.time(); gen_t0 = t0
    sc.current_gen = 0
    pop = tb.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = tb.evaluate(ind)
    hof.update(pop)

    def record(gen, nevals):
        secs = time.time() - gen_t0
        st = _stats(pop)
        logbook.record(gen=gen, nevals=nevals, gen_seconds=round(secs, 1), **st)
        ranked = sorted(pop, key=lambda i: i.fitness.values[0])
        for rank, ind in enumerate(ranked):
            g = G.to_dict(ind)
            pw.writerow([gen, rank, f"{ind.fitness.values[0]:.6f}", g["point_cloud"],
                         g["filtration"], g["vectorizer"], g["homology_dim"], G.signature(ind)])
        pop_csv.flush()
        snap = dict(gen=gen, nevals=nevals, gen_seconds=round(secs, 1), **st,
                    population=[dict(genome=G.to_dict(i), fitness=i.fitness.values[0]) for i in pop],
                    best=dict(genome=G.to_dict(hof[0]), fitness=hof[0].fitness.values[0]))
        gen_log.write(json.dumps(snap, default=str) + "\n"); gen_log.flush()
        b = G.to_dict(hof[0])
        log.info(f"gen {gen:>2d}  min {st['min']:.4f}  avg {st['avg']:.4f}  "
                 f"best={b['point_cloud']}/{b['filtration']}/{b['vectorizer']}  "
                 f"| epoch {secs:.0f}s total {time.time()-t0:.0f}s  ({nevals} evals)")

    record(0, len(pop))
    for gen in range(1, ngen + 1):
        gen_t0 = time.time()
        sc.current_gen = gen
        offspring = [tb.clone(i) for i in tb.select(pop, len(pop))]
        for a, b in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                tb.mate(a, b); del a.fitness.values, b.fitness.values
        for m in offspring:
            if random.random() < mutpb:
                tb.mutate(m); del m.fitness.values
        invalid = [i for i in offspring if not i.fitness.valid]
        for ind in invalid:
            ind.fitness.values = tb.evaluate(ind)
        pop = tools.selBest(pop + offspring, pop_size)     # elitist (mu+lambda)
        hof.update(pop)
        record(gen, len(invalid))

    gen_log.close(); pop_csv.close()
    with open(out / "logbook.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=logbook.header); w.writeheader()
        for row in logbook:
            w.writerow({k: row[k] for k in logbook.header})
    hof_out = [dict(rank=i, genome=G.to_dict(ind), steps=G.decode(ind),
                    metrics={k: sc._fit_cache[G.signature(ind)][k] for k in ("mse", "mae", "da", "r2", "seconds")},
                    fitness=ind.fitness.values[0]) for i, ind in enumerate(hof)]
    (out / "hall_of_fame.json").write_text(json.dumps(hof_out, indent=2, default=str))

    total = time.time() - t0
    log.info(f"DONE in {total:.0f}s ({total/60:.1f} min). {sc.n_evals} unique evals. Output: {out}")
    log.info(f"BEST mse {hof[0].fitness.values[0]:.4f}  {G.to_dict(hof[0])}")
    return out, hof


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="*", default=DATASETS)
    p.add_argument("--model", default="dlinear", choices=["dlinear","patchtst","timemixer"])
    p.add_argument("--pop", type=int, default=24)
    p.add_argument("--ngen", type=int, default=50)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--pred_len", type=int, default=5)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--out_dir", default=None)
    a = p.parse_args()
    ds = [str(DATA / f"{t}.csv") if not str(t).endswith(".csv") else t for t in a.datasets]
    run(ds, model=a.model, pop_size=a.pop, ngen=a.ngen, epochs=a.epochs, seq_len=a.seq_len,
        pred_len=a.pred_len, seed=a.seed, out_dir=a.out_dir)
