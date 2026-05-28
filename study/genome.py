"""Genome for the evolutionary search over persistent-homology pipelines (phlib).

An individual is a fixed-length list of genes encoding the PH process:
  point cloud   : point_cloud, ph_source, window, stride, dimension, delay, width, pca_dim
  filtration    : filtration, metric, dtm_k, maxdim
  vectorization : vectorizer, homology_dim, resolution

decode(ind) -> [("ph_features", params)] for preprocessing.apply (cleanup auto-added).
Some genes are no-ops for some methods (e.g. metric only for rips; dimension/delay only
for point-cloud filtrations; pca_dim only for *_pca) — the study reports which matter.
"""
import random

GENES = [
    # --- point cloud ---
    ("point_cloud", "cat", ["takens", "sliding_window", "takens_pca", "sliding_window_pca"]),
    ("ph_source",   "cat", ["close", "log_return"]),
    ("window",      "int", (32, 160)),     # rolling window length
    ("stride",      "int", (3, 15)),       # rolling recompute cadence — KEY cost lever
                                           # (stride=1 recomputes PH every step: far too slow
                                           #  for expensive filtrations over 2500 pts x 12 sets)
    ("dimension",   "int", (2, 8)),        # Takens embedding dim
    ("delay",       "int", (1, 10)),       # Takens delay
    ("width",       "int", (10, 40)),      # sliding-window width
    ("pca_dim",     "int", (2, 5)),        # PCA target dim (*_pca)
    # --- filtration ---
    ("filtration",  "cat", ["vietoris_rips", "dtm_rips", "alpha", "witness",
                            "cubical", "upper_star", "periodic_cubical"]),
    ("metric",      "cat", ["euclidean", "manhattan", "chebyshev", "cosine"]),
    ("dtm_k",       "int", (3, 12)),
    ("maxdim",      "cat", [0, 1]),
    # --- vectorization ---
    ("vectorizer",  "cat", ["landscape", "silhouette", "betti", "image", "entropy",
                            "topological_vector", "complex_polynomial",
                            "persistence_image", "statistics"]),
    ("homology_dim", "cat", [0, 1]),
    ("resolution",  "int", (8, 40)),
]
NAMES = [g[0] for g in GENES]


def _rand_gene(kind, spec):
    if kind == "bool":
        return random.random() < 0.5
    if kind == "int":
        return random.randint(spec[0], spec[1])
    if kind == "cat":
        return random.choice(spec)
    raise ValueError(kind)


def random_individual():
    return [_rand_gene(k, s) for _, k, s in GENES]


def mutate(ind, indpb=0.2):
    """Per-gene mutation: ints take a small local step; categoricals resample."""
    for i, (_, kind, spec) in enumerate(GENES):
        if random.random() < indpb:
            if kind == "int":
                lo, hi = spec
                ind[i] = int(min(hi, max(lo, ind[i] + random.choice([-2, -1, 1, 2]))))
            else:
                ind[i] = _rand_gene(kind, spec)
    return (ind,)


def to_dict(ind):
    return dict(zip(NAMES, ind))


def decode(ind):
    """Genome -> preprocessing steps (PH-only; cleanup auto-appended by the pipeline)."""
    g = to_dict(ind)
    hd = int(g["homology_dim"])
    # 1-D signal filtrations have NO H1 (no loops); forcing H0 avoids an empty H1
    # diagram -> all-zero features (a degenerate "free model-capacity" win for the EA).
    if g["filtration"] in ("cubical", "lower_star", "upper_star", "periodic_cubical"):
        hd = 0
    params = dict(
        source=g["ph_source"], window=int(g["window"]), stride=int(g["stride"]),
        point_cloud=g["point_cloud"], filtration=g["filtration"], vectorizer=g["vectorizer"],
        homology_dim=hd, maxdim=max(int(g["maxdim"]), hd),
        dimension=int(g["dimension"]), delay=int(g["delay"]), width=int(g["width"]),
        pca_dim=int(g["pca_dim"]), metric=g["metric"], dtm_k=int(g["dtm_k"]),
        # one "resolution/size" knob drives every vectorizer that has one:
        resolution=int(g["resolution"]),          # landscape/silhouette/betti/image/entropy
        pixels=int(g["resolution"]),              # persistence_image (persim)
        n_coeffs=int(g["resolution"]),            # topological_vector / complex_polynomial
    )
    return [("ph_features", params)]


def signature(ind):
    return "|".join(f"{n}={v}" for n, v in zip(NAMES, ind))
