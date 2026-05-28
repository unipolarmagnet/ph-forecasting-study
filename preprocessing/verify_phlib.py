"""Extensive correctness verification for phlib (ASCII output, reproducible).

Checks (each prints PASS/FAIL):
  A. Ground-truth topology  - recover Betti numbers of known shapes.
  B. Cross-engine agreement - ripser vs gudhi Rips give the same H1 diagram.
  C. Vectorizer correctness - cross-check phlib (gudhi/persim) against the
     independently GUDHI-verified `finalph`, plus convention-independent properties.
  D. Distance axioms+stability - d(D,D)=0, bottleneck<=wasserstein, VR 1-Lipschitz (factor 2).
  E. Takens embedding exactness - exact values vs hand computation.
  F. Cubical (sublevel-set) correctness on a known 1-D signal.
  G. Determinism - repeated runs identical on a generic (non-degenerate) cloud.
  H. Empty-diagram handling - vectorizers return zeros of the right length.

Run:  python preprocessing/verify_phlib.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phlib  # noqa: E402

# finalph is used ONLY here as an optional independent oracle for cross-checks;
# phlib itself does not depend on it. Skip those checks if it is unavailable.
_FP = Path(__file__).resolve().parent.parent / "vendor" / "finalph"
try:
    if str(_FP) not in sys.path:
        sys.path.insert(0, str(_FP))
    import finalph as _FPH  # noqa: F401
    HAVE_FP = True
except Exception:
    HAVE_FP = False

RNG = np.random.RandomState(0)
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  | ' + detail) if detail else ''}")


def betti(dgms, k, frac=0.5, floor=0.08):
    """# significant bars in dim k: per-dimension (lifetime > frac * max lifetime
    in that dim), but a dimension whose top bar is < floor * GLOBAL max lifetime is
    treated as pure noise (score 0). Handles both 'feature dominates' (sphere H2)
    and 'noise-only' (sphere H1) dimensions with one rule."""
    glob = max((float((d[:, 1] - d[:, 0]).max()) for d in dgms if len(d)), default=0.0)
    if k >= len(dgms) or len(dgms[k]) == 0 or glob <= 0:
        return 0
    life = dgms[k][:, 1] - dgms[k][:, 0]
    if life.max() < floor * glob:
        return 0
    return int((life > frac * life.max()).sum())


# ------------------------------------------------------------------ shapes
def circle(n=120, r=1.0, c=(0, 0)):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[c[0] + r * np.cos(t), c[1] + r * np.sin(t)]


def sphere(n=300):
    x = RNG.randn(n, 3)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def torus(n=1000, R=3.0, r=1.0):
    # random (non-degenerate) sampling; a regular grid creates Delaunay degeneracies
    u, v = RNG.rand(n) * 2 * np.pi, RNG.rand(n) * 2 * np.pi
    return np.c_[(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)]


# ================================================================== A. topology
def test_topology():
    print("\nA. GROUND-TRUTH TOPOLOGY (recover Betti numbers)")
    d = phlib.filt_vietoris_rips(circle(120), maxdim=1)
    check("circle  H1=1", betti(d, 1) == 1, f"got {betti(d,1)}")
    two = np.vstack([circle(80, c=(-2.6, 0)), circle(80, c=(2.6, 0))])
    d = phlib.filt_vietoris_rips(two, maxdim=1)
    check("two circles  H1=2", betti(d, 1) == 2, f"got {betti(d,1)}")
    fig8 = np.vstack([circle(90, c=(-1, 0)), circle(90, c=(1, 0))])
    d = phlib.filt_vietoris_rips(fig8, maxdim=1)
    check("figure-eight  H1=2", betti(d, 1) == 2, f"got {betti(d,1)}")
    blobs = np.vstack([RNG.randn(40, 2) * 0.15 + c for c in [(0, 0), (10, 0), (5, 9)]])
    d = phlib.filt_vietoris_rips(blobs, maxdim=1)
    comp = betti(d, 0) + 1                      # components = long finite H0 bars + 1 (the infinite)
    check("three clusters  components=3", comp == 3, f"got {comp}")
    d = phlib.filt_alpha(sphere(300), maxdim=2)
    check("sphere  H2=1", betti(d, 2) == 1, f"got {betti(d,2)}")
    check("sphere  H1=0 (noise only)", betti(d, 1) == 0, f"got {betti(d,1)}")
    d = phlib.filt_alpha(torus(1000), maxdim=2)
    # torus: one void (H2=1) and >=1 loop (H1>=1). Exact H1=2 is sampling/threshold
    # dependent (the two cycles differ in persistence by ~R/r), so assert robustly.
    check("torus  H2=1", betti(d, 2) == 1, f"got {betti(d,2)}")
    check("torus  H1>=1 (loops present)", betti(d, 1) >= 1, f"significant H1={betti(d,1)}")
    # new methods also recover the circle loop
    d = phlib.filt_dtm_rips(circle(120), maxdim=1, dtm_k=4)
    check("dtm_rips  circle H1=1", betti(d, 1) == 1, f"got {betti(d,1)}")
    cloud = phlib.POINT_CLOUDS["takens_pca"](np.sin(np.linspace(0, 30, 300)), dimension=10, delay=5, pca_dim=3)
    d = phlib.filt_vietoris_rips(cloud, maxdim=1)
    check("takens_pca  sine H1=1", betti(d, 1) == 1, f"got {betti(d,1)}")


# ================================================================== B. cross-engine
def test_cross_engine():
    print("\nB. CROSS-ENGINE AGREEMENT (ripser vs gudhi Rips)")
    import gudhi as g
    X = circle(60)
    dr = phlib.filt_vietoris_rips(X, maxdim=1, threshold=3.0)
    st = g.RipsComplex(points=X, max_edge_length=3.0).create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    dg = phlib._clean(st.persistence_intervals_in_dimension(1))
    bd = float(g.bottleneck_distance(dr[1], dg))
    check("ripser vs gudhi-Rips  H1 bottleneck ~ 0", bd < 1e-6, f"d_B={bd:.2e}")


# ================================================================== C. vectorizers
def test_vectorizers():
    print("\nC. VECTORIZER CORRECTNESS (properties + optional finalph cross-check)")
    import gudhi.representations as R
    D = np.array([[0.0, 2.0], [1.0, 4.0], [0.5, 1.5]])
    a, b = float(D[:, 0].min()), float(D[:, 1].max())
    res = 50
    life = D[:, 1] - D[:, 0]

    # convention-independent properties
    g_bc = np.asarray(R.BettiCurve(resolution=res, sample_range=[a, b]).fit_transform([D])[0])
    check("betti max == max overlap (3)", int(round(g_bc.max())) == 3, f"max={g_bc.max():.1f}")
    g_ls = np.asarray(R.Landscape(num_landscapes=3, resolution=res, sample_range=[a, b]).fit_transform([D])[0])
    L = g_ls.reshape(3, res)
    check("landscape layers non-increasing", bool((L[0] >= L[1] - 1e-9).all() and (L[1] >= L[2] - 1e-9).all()))
    g_H = float(np.asarray(R.Entropy(mode="scalar").fit_transform([D])[0]).ravel()[0])
    p = life / life.sum(); manual_H = float(-(p * np.log(p)).sum())
    check("entropy == manual -sum(p*ln p)", abs(g_H - manual_H) < 1e-6, f"{g_H:.6f} vs {manual_H:.6f}")

    # phlib statistics (numpy): [count, total persistence, ...]
    st = phlib.vectorize(D, vectorizer="statistics")
    check("statistics: count==3 and total==sum(life)",
          st[0] == 3 and np.isclose(st[1], life.sum()), f"count={st[0]:.0f} total={st[1]:.3f}")
    img = phlib.vectorize(D, vectorizer="persistence_image", pixels=16)
    check("persistence_image non-negative", bool((img >= -1e-9).all()), f"min={img.min():.3f}")

    # optional: phlib(gudhi) vectorizers == independently-verified finalph to 1e-6
    if HAVE_FP:
        import finalph as fph
        f_ls = np.asarray(fph.PersistenceLandscape(num_landscapes=3, resolution=res, sample_range=(a, b)).fit_transform(D)).ravel()
        check("landscape == finalph (1e-6)", np.allclose(g_ls, f_ls, atol=1e-6), f"maxdiff={np.abs(g_ls-f_ls).max():.1e}")
        f_bc = np.asarray(fph.BettiCurve(resolution=res, sample_range=(a, b)).fit_transform(D)).ravel()
        check("betti == finalph (1e-6)", np.allclose(g_bc, f_bc, atol=1e-6), f"maxdiff={np.abs(g_bc-f_bc).max():.1e}")
        f_H = float(np.asarray(fph.PersistenceEntropy().fit_transform(D)).ravel()[0])
        check("entropy == finalph (1e-6)", abs(g_H - f_H) < 1e-6, f"diff={abs(g_H-f_H):.1e}")
    else:
        print("  [skip] finalph oracle not available")


# ================================================================== D. distances
def test_distances():
    print("\nD. DISTANCE AXIOMS + STABILITY")
    X = circle(60); Y = X + 0.05 * RNG.randn(*X.shape)
    dx = phlib.filt_vietoris_rips(X, maxdim=1, threshold=3.0)[1]
    dy = phlib.filt_vietoris_rips(Y, maxdim=1, threshold=3.0)[1]
    for name in phlib.DISTANCES:
        check(f"{name}(D,D)=0", abs(phlib.DISTANCES[name](dx, dx)) < 1e-9)
    bn = phlib.DISTANCES["bottleneck"](dx, dy)
    w1 = phlib.DISTANCES["wasserstein"](dx, dy, order=1.0)
    check("bottleneck <= wasserstein(1)", bn <= w1 + 1e-9, f"d_B={bn:.4f} W1={w1:.4f}")
    eps = float(np.linalg.norm(X - Y, axis=1).max())
    check("stability  d_B <= 2*max_displacement (VR 1-Lipschitz)", bn <= 2 * eps + 1e-6,
          f"d_B={bn:.4f} 2*eps={2*eps:.4f}")


# ================================================================== E. embedding
def test_embedding():
    print("\nE. TAKENS EMBEDDING EXACTNESS")
    s = np.arange(10.0)
    c = phlib.POINT_CLOUDS["takens"](s, dimension=3, delay=2, stride=1)
    expected = np.array([[s[i], s[i + 2], s[i + 4]] for i in range(len(s) - 4)])
    check("takens(dim3,delay2) exact rows", c.shape == expected.shape and np.allclose(c, expected),
          f"shape {c.shape}, first {c[0] if len(c) else '-'}")


# ================================================================== F. cubical
def test_cubical():
    print("\nF. CUBICAL (sublevel-set) on a known 1-D signal")
    sig = np.array([0, 3, 1, 4, 0.5, 5, 2, 6, 0.0])      # local minima at values {0, 0.5, 1, 2}
    d0 = phlib.filt_cubical(sig, maxdim=1)[0]
    births = set(np.round(np.unique(d0[:, 0]), 3)) if len(d0) else set()
    check("cubical births at local minima", len(d0) >= 2 and births.issubset({0.0, 0.5, 1.0, 2.0}),
          f"births={sorted(births)}")


# ================================================================== G. determinism
def test_determinism():
    print("\nG. DETERMINISM (repeated runs identical, generic non-degenerate input)")
    sig = np.cumsum(RNG.randn(200))                       # random-walk signal (generic)
    cloud = phlib.POINT_CLOUDS["takens"](sig, dimension=3, delay=4)
    for f in ["vietoris_rips", "dtm_rips", "alpha", "witness",
              "cubical", "upper_star", "periodic_cubical"]:
        inp = sig if f in phlib.SIGNAL_FILTRATIONS else cloud
        d1 = phlib.FILTRATIONS[f](inp, maxdim=1)
        d2 = phlib.FILTRATIONS[f](inp, maxdim=1)
        ok = all(np.array_equal(np.sort(p, 0), np.sort(q, 0)) for p, q in zip(d1, d2))
        check(f"{f} deterministic", ok)
    for pc in ["takens_pca", "sliding_window_pca"]:        # PCA solver determinism
        c1 = phlib.POINT_CLOUDS[pc](sig, dimension=10, delay=4, width=20, pca_dim=3)
        c2 = phlib.POINT_CLOUDS[pc](sig, dimension=10, delay=4, width=20, pca_dim=3)
        check(f"{pc} deterministic", c1.shape == c2.shape and np.allclose(c1, c2))


# ================================================================== H. empty handling
def test_empty():
    print("\nH. EMPTY-DIAGRAM HANDLING (zeros of correct length)")
    empty = np.empty((0, 2))
    cases = [("landscape", 3 * 20), ("betti", 20), ("silhouette", 20),
             ("image", 20 * 20), ("statistics", 13), ("topological_vector", 8),
             ("persistence_image", 16 * 16)]
    for v, exp in cases:
        out = phlib.vectorize(empty, vectorizer=v, resolution=20, num_landscapes=3, pixels=16, n_coeffs=8)
        check(f"{v} empty -> zeros[{exp}]", out.shape == (exp,) and not out.any(), f"shape {out.shape}")


if __name__ == "__main__":
    for t in [test_topology, test_cross_engine, test_vectorizers, test_distances,
              test_embedding, test_cubical, test_determinism, test_empty]:
        try:
            t()
        except Exception as e:
            check(t.__name__ + " (crashed)", False, f"{type(e).__name__}: {e}")
    n, p = len(results), sum(results)
    print(f"\n==== {p}/{n} checks PASSED ====" + ("  ALL GOOD" if p == n else "  !! FAILURES ABOVE"))
    sys.exit(0 if p == n else 1)
