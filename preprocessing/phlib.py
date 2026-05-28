"""phlib — an extensive persistent-homology toolkit (a facade over real TDA libraries).

This module does NOT implement any persistent-homology algorithm itself. It is a
single, consistent facade that routes each step of the PH pipeline to the best
available, verified library:

    backend libraries:  gudhi · ripser · persim · scikit-learn (PCA)

The PH pipeline has three searchable stages, each a registry of named methods:

  1. POINT CLOUD   (1-D series -> points)            POINT_CLOUDS
        takens              gudhi TimeDelayEmbedding (Takens delay embedding)
        sliding_window      plain length-w windows as points
        takens_pca          Takens embedding reduced to pca_dim (sklearn PCA)
        sliding_window_pca  windowing reduced to pca_dim (SW1PerS-style)
        (cubical/upper_star/periodic filtrations skip this and use the raw signal)
  2. FILTRATION    (points/signal -> persistence diagrams)   FILTRATIONS
        vietoris_rips     ripser  (exact Vietoris-Rips, deterministic)
        dtm_rips          gudhi DTMRipsComplex (DTM-weighted Rips; outlier-robust)
        alpha             gudhi AlphaComplex   (Delaunay; squared-radius scale)
        witness           gudhi EuclideanWitnessComplex (deterministic max-min landmarks)
        cubical/lower_star gudhi CubicalComplex (sublevel-set: minima)
        upper_star        superlevel-set of the signal (maxima)
        periodic_cubical  gudhi PeriodicCubicalComplex (cyclic sublevel-set)
  3. VECTORIZATION (diagram -> fixed-length vector)          VECTORIZERS
        landscape silhouette betti image entropy topological_vector
        complex_polynomial persistence_image statistics

Plus diagram DISTANCES (bottleneck, wasserstein, sliced_wasserstein, heat_kernel).

Canonical diagram format throughout: a list of float arrays, one per homology
dimension k, each of shape (n_k, 2) = (birth, death), with infinite bars removed
(the standard input to every vectorizer). Use :func:`compute` to run the whole
pipeline, or call the registries directly.
"""
from __future__ import annotations

import inspect
import numpy as np

# ----------------------------------------------------------------------------- backends (lazy)
def _gudhi():
    import gudhi
    return gudhi


# ----------------------------------------------------------------------------- diagram utils
def _clean(arr):
    """Keep finite (birth, death) rows with death > birth."""
    a = np.asarray(arr, dtype=float).reshape(-1, 2)
    if a.size == 0:
        return np.empty((0, 2))
    a = a[np.isfinite(a).all(axis=1)]
    return a[a[:, 1] > a[:, 0]]


def _from_simplextree(st, maxdim):
    st.compute_persistence(persistence_dim_max=True)
    return [_clean(st.persistence_intervals_in_dimension(k)) for k in range(maxdim + 1)]


# ============================================================================= 1) POINT CLOUDS
def pc_takens(series, dimension=3, delay=1, stride=1, **_):
    """Takens delay embedding via gudhi.point_cloud.timedelay.TimeDelayEmbedding."""
    from gudhi.point_cloud.timedelay import TimeDelayEmbedding
    tde = TimeDelayEmbedding(dim=int(dimension), delay=int(delay), skip=int(stride))
    cloud = np.asarray(tde(np.asarray(series, dtype=float)))
    return cloud.reshape(-1, int(dimension))


def pc_sliding_window(series, width=20, stride=1, **_):
    """Each length-`width` window becomes a point in R^width (plain windowing)."""
    s = np.asarray(series, dtype=float)
    idx = range(0, len(s) - width + 1, int(stride))
    return np.stack([s[i:i + width] for i in idx]) if len(s) >= width else np.empty((0, width))


def _pca(cloud, pca_dim):
    from sklearn.decomposition import PCA          # existing library, deterministic solver
    k = min(int(pca_dim), cloud.shape[1], max(cloud.shape[0] - 1, 1))
    if cloud.shape[1] <= k or len(cloud) <= k:
        return cloud
    return PCA(n_components=k, svd_solver="full").fit_transform(cloud)


def pc_takens_pca(series, dimension=10, delay=1, stride=1, pca_dim=3, **_):
    """High-dimensional Takens embedding reduced to `pca_dim` via PCA (sklearn)."""
    return _pca(pc_takens(series, dimension=dimension, delay=delay, stride=stride), pca_dim)


def pc_sliding_window_pca(series, width=20, stride=1, pca_dim=3, **_):
    """Sliding-window embedding reduced to `pca_dim` via PCA (the SW1PerS-style cloud)."""
    return _pca(pc_sliding_window(series, width=width, stride=stride), pca_dim)


POINT_CLOUDS = {
    "takens": pc_takens,                          # gudhi TimeDelayEmbedding
    "sliding_window": pc_sliding_window,          # plain windowing
    "takens_pca": pc_takens_pca,                  # gudhi + sklearn PCA
    "sliding_window_pca": pc_sliding_window_pca,  # windowing + sklearn PCA
}
# filtrations that consume the raw 1-D signal instead of a point cloud:
SIGNAL_FILTRATIONS = {"cubical", "lower_star", "upper_star", "periodic_cubical"}


# ============================================================================= 2) FILTRATIONS
def filt_vietoris_rips(cloud, maxdim=1, threshold=np.inf, coeff=2, metric="euclidean", **_):
    """Vietoris-Rips via Ripser (fast, field standard)."""
    from ripser import ripser
    thr = float(threshold) if np.isfinite(threshold) else np.inf
    res = ripser(np.asarray(cloud, dtype=float), maxdim=int(maxdim),
                 thresh=thr, coeff=int(coeff), metric=metric)
    return [_clean(d) for d in res["dgms"]]


def filt_alpha(cloud, maxdim=1, **_):
    """Alpha complex via gudhi.AlphaComplex.

    The cloud is PCA-reduced to <=3D first: AlphaComplex builds a Delaunay
    triangulation, which is intractable (hours / hangs) above ~3-4 dimensions —
    and clouds here can be high-dimensional (sliding_window -> R^width,
    takens -> R^dimension). Alpha is only meaningful in low dimension, so this is
    the standard fix. NOTE: alpha filtration values are SQUARED circumradii
    (different scale from Rips radii); each filtration is internally self-consistent.
    """
    g = _gudhi()
    pts = np.asarray(cloud, dtype=float)
    if pts.shape[1] > 3:
        pts = _pca(pts, 3)                       # keep the Delaunay tractable
    st = g.AlphaComplex(points=pts).create_simplex_tree()
    return _from_simplextree(st, maxdim)


def filt_cubical(signal, maxdim=1, **_):
    """Sublevel-set (lower-star) filtration of the raw signal via gudhi.CubicalComplex.
    Captures the structure of the signal's minima/valleys."""
    g = _gudhi()
    cc = g.CubicalComplex(top_dimensional_cells=np.asarray(signal, dtype=float))
    cc.compute_persistence()
    return [_clean(cc.persistence_intervals_in_dimension(k)) for k in range(maxdim + 1)]


def filt_upper_star(signal, maxdim=1, **_):
    """Superlevel-set (upper-star) filtration of the raw signal (captures maxima).

    Computed as the sublevel-set of the negated signal, then mapped back to the
    signal's value scale: a (-f) bar (b,d) becomes the f bar (-d, -b)."""
    dgms = filt_cubical(-np.asarray(signal, dtype=float), maxdim=maxdim)
    return [_clean(np.c_[-d[:, 1], -d[:, 0]]) if len(d) else d for d in dgms]


def filt_periodic_cubical(signal, maxdim=1, **_):
    """Sublevel-set filtration with PERIODIC boundary via gudhi.PeriodicCubicalComplex
    (treats the 1-D signal as a cycle; an extra H1 loop reflects the periodicity)."""
    g = _gudhi()
    s = np.asarray(signal, dtype=float)
    cc = g.PeriodicCubicalComplex(top_dimensional_cells=s, periodic_dimensions=[True])
    cc.compute_persistence()
    return [_clean(cc.persistence_intervals_in_dimension(k)) for k in range(maxdim + 1)]


def filt_dtm_rips(cloud, maxdim=1, threshold=np.inf, dtm_k=4, dtm_q=2, **_):
    """DTM-weighted Rips via gudhi.DTMRipsComplex — a Vietoris-Rips filtration robust
    to noise/outliers (uses the distance-to-measure with k neighbours). Deterministic."""
    from gudhi.dtm_rips_complex import DTMRipsComplex
    pts = np.asarray(cloud, dtype=float)
    mf = float(threshold) if np.isfinite(threshold) else np.inf
    st = DTMRipsComplex(points=pts, k=int(dtm_k), q=float(dtm_q),
                        max_filtration=mf).create_simplex_tree(max_dimension=int(maxdim) + 1)
    return _from_simplextree(st, maxdim)


def filt_witness(cloud, maxdim=1, n_landmarks=50, witness_radius_frac=0.5, **_):
    """Landmark-based Euclidean witness complex via gudhi.

    max_alpha_square is set from the cloud extent (so components actually merge);
    `witness_radius_frac` scales it (squared fraction of the bounding-box diagonal).
    """
    g = _gudhi()
    pts = np.asarray(cloud, dtype=float)
    n_land = min(int(n_landmarks), len(pts))
    # deterministic max-min (farthest-point) landmarks — the standard, reproducible
    # choice for witness complexes (random landmarks would be non-deterministic).
    landmarks = np.asarray(g.subsampling.choose_n_farthest_points(
        points=pts, nb_points=n_land, starting_point=0))
    wc = g.EuclideanWitnessComplex(landmarks=landmarks, witnesses=pts)
    diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    rad = float((diag * float(witness_radius_frac)) ** 2 + 1e-9)
    st = wc.create_simplex_tree(max_alpha_square=rad, limit_dimension=int(maxdim))
    return _from_simplextree(st, maxdim)


FILTRATIONS = {
    "vietoris_rips":    filt_vietoris_rips,   # ripser (exact, fast, deterministic)
    "dtm_rips":         filt_dtm_rips,        # gudhi (DTM-weighted Rips, outlier-robust)
    "alpha":            filt_alpha,           # gudhi (Delaunay; squared-radius scale)
    "witness":          filt_witness,         # gudhi (deterministic max-min landmarks)
    "cubical":          filt_cubical,         # gudhi (sublevel-set of the raw signal)
    "lower_star":       filt_cubical,         # alias of cubical (minima)
    "upper_star":       filt_upper_star,      # gudhi (superlevel-set: maxima)
    "periodic_cubical": filt_periodic_cubical,  # gudhi (cyclic sublevel-set)
}


# ============================================================================= 3) VECTORIZERS
# Each takes a single finite (n,2) diagram (the chosen H_k) + params -> 1-D-ish vector.
def _rep(name):
    import gudhi.representations as R
    return getattr(R, name)


def _safe(fn, expected, dgm, *a, **k):
    """Empty diagram -> zeros(expected). Non-empty -> compute (errors propagate),
    padded/trimmed to `expected` for a consistent feature length."""
    if dgm is None or len(dgm) == 0:
        return np.zeros(int(expected))
    v = np.asarray(fn(dgm, *a, **k), dtype=float).ravel()
    if v.size != expected:
        out = np.zeros(int(expected)); out[:min(v.size, expected)] = v[:expected]; return out
    return v


def vec_landscape(dgm, num_landscapes=5, resolution=100, **_):
    R = _rep("Landscape")(num_landscapes=int(num_landscapes), resolution=int(resolution))
    return _safe(lambda d: R.fit_transform([d])[0], num_landscapes * resolution, dgm)


def vec_silhouette(dgm, resolution=100, power=1.0, **_):
    R = _rep("Silhouette")(resolution=int(resolution),
                           weight=lambda p: (p[1] - p[0]) ** float(power))
    return _safe(lambda d: R.fit_transform([d])[0], resolution, dgm)


def vec_betti(dgm, resolution=100, **_):
    R = _rep("BettiCurve")(resolution=int(resolution))
    return _safe(lambda d: R.fit_transform([d])[0], resolution, dgm)


def vec_image(dgm, resolution=20, bandwidth=1.0, weight_power=1.0, **_):
    r = int(resolution)
    R = _rep("PersistenceImage")(bandwidth=float(bandwidth), resolution=[r, r],
                                 weight=lambda p: (p[1] - p[0]) ** float(weight_power))
    return _safe(lambda d: R.fit_transform([d])[0], r * r, dgm)


def vec_entropy(dgm, mode="vector", resolution=100, normalized=True, **_):
    R = _rep("Entropy")(mode=mode, resolution=int(resolution), normalized=bool(normalized))
    exp = 1 if mode == "scalar" else int(resolution)
    return _safe(lambda d: R.fit_transform([d])[0], exp, dgm)


def vec_topological_vector(dgm, n_coeffs=10, **_):
    R = _rep("TopologicalVector")(threshold=int(n_coeffs))
    return _safe(lambda d: R.fit_transform([d])[0], n_coeffs, dgm)


def vec_complex_polynomial(dgm, n_coeffs=10, polynomial_type="R", **_):
    R = _rep("ComplexPolynomial")(threshold=int(n_coeffs), polynomial_type=polynomial_type)
    def go(d):
        c = np.asarray(R.fit_transform([d])[0]).ravel()
        return np.concatenate([c.real, c.imag])
    return _safe(go, 2 * n_coeffs, dgm)


def vec_image_persim(dgm, pixels=20, spread=1.0, **_):
    from persim import PersistenceImager
    r = int(pixels)
    def go(d):
        pim = PersistenceImager(pixel_size=1.0 / r, kernel_params={"sigma": [[spread, 0], [0, spread]]})
        img = np.asarray(pim.transform(d, skew=True)).ravel()
        out = np.zeros(r * r); out[:min(img.size, r * r)] = img[:r * r]; return out
    return _safe(go, r * r, dgm)


def vec_statistics(dgm, **_):
    """13 descriptive statistics of the diagram (pure numpy; no external lib).

    [count, total persistence, mean/std/max/min/median lifetime, mean/std birth,
     mean/std death, mean midpoint, persistence entropy]."""
    def go(d):
        b, de = d[:, 0], d[:, 1]
        life = de - b
        mid = 0.5 * (b + de)
        tot = float(life.sum())
        p = life / tot if tot > 0 else np.zeros_like(life)
        ent = float(-(p * np.log(p + 1e-12)).sum())
        return np.array([len(life), tot, life.mean(), life.std(), life.max(), life.min(),
                         np.median(life), b.mean(), b.std(), de.mean(), de.std(),
                         mid.mean(), ent], dtype=float)
    return _safe(go, 13, dgm)


VECTORIZERS = {
    "landscape":          vec_landscape,           # gudhi.representations
    "silhouette":         vec_silhouette,          # gudhi
    "betti":              vec_betti,               # gudhi
    "image":              vec_image,               # gudhi
    "entropy":            vec_entropy,             # gudhi
    "topological_vector": vec_topological_vector,  # gudhi
    "complex_polynomial": vec_complex_polynomial,  # gudhi
    "persistence_image":  vec_image_persim,        # persim
    "statistics":         vec_statistics,          # numpy descriptive summary
}


# ============================================================================= DISTANCES
def dist_bottleneck(dgm1, dgm2, **_):
    return float(_gudhi().bottleneck_distance(np.asarray(dgm1), np.asarray(dgm2)))


def dist_wasserstein(dgm1, dgm2, order=1.0, **_):
    from gudhi.hera import wasserstein_distance
    return float(wasserstein_distance(np.asarray(dgm1), np.asarray(dgm2), order=float(order)))


def dist_sliced_wasserstein(dgm1, dgm2, **_):
    import persim
    return float(persim.sliced_wasserstein(np.asarray(dgm1), np.asarray(dgm2)))


def dist_heat_kernel(dgm1, dgm2, sigma=0.4, **_):
    """Persistence heat-kernel (pseudo-)distance between two diagrams (persim.heat)."""
    import persim
    return float(persim.heat(np.asarray(dgm1), np.asarray(dgm2), sigma=float(sigma)))


DISTANCES = {
    "bottleneck": dist_bottleneck,                  # gudhi
    "wasserstein": dist_wasserstein,                # gudhi (hera)
    "sliced_wasserstein": dist_sliced_wasserstein,  # persim
    "heat_kernel": dist_heat_kernel,                # persim
}


# ============================================================================= pipeline
def diagrams(series, point_cloud="takens", filtration="vietoris_rips",
             dimension=3, delay=1, stride=1, width=20, pca_dim=3, maxdim=1,
             threshold=np.inf, coeff=2, metric="euclidean", **filt_params):
    """1-D series -> persistence diagrams (list of (n_k,2) arrays per homology dim)."""
    series = np.asarray(series, dtype=float).ravel()
    if filtration not in FILTRATIONS:
        raise ValueError(f"filtration {filtration!r}; choose {list(FILTRATIONS)}")
    if filtration in SIGNAL_FILTRATIONS:           # cubical/upper_star/periodic: raw signal
        return FILTRATIONS[filtration](series, maxdim=maxdim, **filt_params)
    if point_cloud not in POINT_CLOUDS:
        raise ValueError(f"point_cloud {point_cloud!r}; choose {list(POINT_CLOUDS)}")
    cloud = POINT_CLOUDS[point_cloud](series, dimension=dimension, delay=delay,
                                      stride=stride, width=width, pca_dim=pca_dim)
    return FILTRATIONS[filtration](cloud, maxdim=maxdim, threshold=threshold,
                                   coeff=coeff, metric=metric, **filt_params)


def vectorize(dgm, vectorizer="landscape", **params):
    """Single (n,2) diagram -> fixed-length feature vector via the chosen method."""
    if vectorizer not in VECTORIZERS:
        raise ValueError(f"vectorizer {vectorizer!r}; choose {list(VECTORIZERS)}")
    fn = VECTORIZERS[vectorizer]
    allowed = set(inspect.signature(fn).parameters) - {"dgm"}
    return np.asarray(fn(dgm, **{k: v for k, v in params.items() if k in allowed}))


def compute(series, point_cloud="takens", filtration="vietoris_rips",
            vectorizer="landscape", homology_dim=1, maxdim=1, **params):
    """Full pipeline: series -> diagrams -> vectorized features of H_{homology_dim}."""
    dgms = diagrams(series, point_cloud=point_cloud, filtration=filtration,
                    maxdim=max(maxdim, homology_dim), **params)
    dgm = dgms[homology_dim] if homology_dim < len(dgms) else np.empty((0, 2))
    feats = vectorize(dgm, vectorizer=vectorizer, **params)
    info = dict(point_cloud=point_cloud, filtration=filtration, vectorizer=vectorizer,
                homology_dim=homology_dim,
                diagram_points={k: int(len(d)) for k, d in enumerate(dgms)},
                feature_shape=list(np.shape(feats)), feature_len=int(np.size(feats)))
    return feats, dgms, info


def capabilities():
    """Return the registries (names) for point clouds, filtrations, vectorizers, distances."""
    return dict(point_clouds=list(POINT_CLOUDS), filtrations=list(FILTRATIONS),
                vectorizers=list(VECTORIZERS), distances=list(DISTANCES))
