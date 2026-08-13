"""Validate a categorical chart palette — the computable data-viz checks.

A Python twin of the dataviz skill's validate_palette.js, kept in the repo
because this box has no node and `scripts/plotting/grok_curves.py` cites this
path for its palette verification. Same thresholds, same Machado-Oliveira-
Fernandes (2009) severity-1.0 CVD model, same OKLab dE x100 metric.

    venv/bin/python scripts/validate_palette.py "#2a78d6,#eb6834" --mode light
    venv/bin/python scripts/validate_palette.py "#3987e5,#d95926" --mode dark --pairs all

Exit 0 unless a check hard-FAILs. WARN bands (adjacent CVD 6-8, sub-3:1
contrast) do not fail; each is legal only with mandatory secondary encoding.
"""
import argparse, itertools, math, re, sys

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
DEFAULT_SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}

MACHADO = {
    "protan": ((0.152286, 1.052583, -0.204868), (0.114503, 0.786281, 0.099216),
               (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968), (0.280085, 0.672501, 0.047413),
               (-0.011820, 0.042940, 0.968881)),
    "tritan": ((1.255528, -0.076749, -0.178779), (-0.078411, 0.930809, 0.147602),
               (0.004733, 0.691367, 0.303900)),
}
WS = "[ \t\n\v\f\r   -     　]+"
_strip = lambda v: re.sub(f"^{WS}|{WS}$", "", v)
_ishex = lambda v: re.fullmatch(r"#?[0-9a-fA-F]{6}", v) is not None


def _s2lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin(h):
    h = h.strip().lstrip("#")
    return [_s2lin(int(h[i:i + 2], 16) / 255) for i in (0, 2, 4)]


def rel_lum(h):
    r, g, b = lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted((rel_lum(a), rel_lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab_from_lin(rgb):
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def oklch(h):
    L, a, b = oklab_from_lin(lin(h))
    return L, math.hypot(a, b)


def simulate(h, kind):
    r, g, b = lin(h)
    M = MACHADO[kind]
    return [min(1.0, max(0.0, M[i][0] * r + M[i][1] * g + M[i][2] * b)) for i in range(3)]


def delta_e(h1, h2, kind=None):
    a = oklab_from_lin(simulate(h1, kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2, kind) if kind else lin(h2))
    return 100 * math.dist(a, b)


def validate(palette, mode="light", surface=None, pairs="adjacent"):
    surface = surface or DEFAULT_SURFACE[mode]
    lo, hi = BAND[mode]
    rows, ok = [], True

    off = [(c, round(oklch(c)[0], 3)) for c in palette if not lo <= oklch(c)[0] <= hi]
    ok &= not off
    rows.append(("Lightness band", not off,
                 f"outside band: {off}" if off else f"all {len(palette)} inside L {lo}-{hi}"))

    low = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    ok &= not low
    rows.append(("Chroma floor", not low,
                 f"below {CHROMA_FLOOR}: {low}" if low else f"all >= {CHROMA_FLOOR}"))

    pl = (list(itertools.combinations(range(len(palette)), 2)) if pairs == "all"
          else [(i, i + 1) for i in range(len(palette) - 1)])
    worst_cvd, worst_cvd_pair = math.inf, None
    for i, j in pl:
        d = min(delta_e(palette[i], palette[j], "protan"),
                delta_e(palette[i], palette[j], "deutan"))
        if d < worst_cvd:
            worst_cvd, worst_cvd_pair = d, (palette[i], palette[j])
    cvd_ok = worst_cvd >= CVD_FLOOR
    ok &= cvd_ok
    note = f"worst {pairs} pair {worst_cvd_pair} dE {worst_cvd:.1f}"
    if worst_cvd < CVD_TARGET:
        note += f" -- WARN, in the {CVD_FLOOR}-{CVD_TARGET} band, secondary encoding REQUIRED"
    rows.append((f"CVD separation ({pairs})", cvd_ok, note))

    worst_n, worst_n_pair = math.inf, None
    for i, j in pl:
        d = delta_e(palette[i], palette[j])
        if d < worst_n:
            worst_n, worst_n_pair = d, (palette[i], palette[j])
    n_ok = worst_n >= NORMAL_FLOOR
    ok &= n_ok
    rows.append(("Normal-vision floor", n_ok,
                 f"worst pair {worst_n_pair} dE {worst_n:.1f} (floor {NORMAL_FLOOR})"))

    cr = [(c, round(contrast(c, surface), 2)) for c in palette]
    weak = [x for x in cr if x[1] < CONTRAST_MIN]
    rows.append((f"Contrast vs {surface}", True,
                 (f"WARN sub-{CONTRAST_MIN}:1 (relief rule -- labels or table view): {weak}"
                  if weak else f"all >= {CONTRAST_MIN}:1") + f" | {cr}"))
    return rows, ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("palette")
    ap.add_argument("--mode", default="light", choices=["light", "dark"])
    ap.add_argument("--surface", default=None)
    ap.add_argument("--pairs", default="adjacent", choices=["adjacent", "all"])
    a = ap.parse_args()
    pal = [c if c.startswith("#") else "#" + c
           for c in (_strip(x) for x in a.palette.split(",")) if c]
    bad = [c for c in pal if not _ishex(c)]
    if bad or not pal:
        sys.exit(f"not hex colors: {bad or 'empty palette'}")
    surf = a.surface
    if surf and not _ishex(_strip(surf)):
        sys.exit(f"surface not a hex color: {surf}")
    rows, ok = validate(pal, a.mode, surf, a.pairs)
    print(f"\n  palette {pal}  mode={a.mode}  surface={surf or DEFAULT_SURFACE[a.mode]}  pairs={a.pairs}\n")
    for name, passed, note in rows:
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<26} {note}")
    print()
    sys.exit(0 if ok else 1)
