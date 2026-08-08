"""Build a self-contained HTML page of train/test accuracy curves per setup.

Reads the per-run history JSONs the training loops write and renders one small
multiple per run: train and test accuracy against gradient steps, on a log x
axis so the memorise->generalise delay that defines grokking is legible.

Two annotations carry the science:
  * the dataset's grok threshold, as a horizontal rule -- it is 95% for modular,
    90% for MNIST and 85% for S5, and a t_grok is only meaningful next to the bar
    it was measured at;
  * T_grok, as a vertical rule -- the step at which test accuracy reaches that bar
    and never falls back.

    python scripts/plotting/grok_curves.py --out /tmp/curves.html --runs <id> <id> ...
"""

import argparse
import glob
import json
import math
import os


# Categorical slots 1 and 2 from the validated default palette, both modes.
# Verified with scripts/validate_palette.py: all six checks PASS in light and
# dark (worst adjacent CVD dE 24.7 / 26.8, normal-vision 33.6 / 31.8).
SERIES = [
    {"key": "train_acc", "label": "Train", "light": "#2a78d6", "dark": "#3987e5"},
    {"key": "test_acc", "label": "Test", "light": "#eb6834", "dark": "#d95926"},
]

SETUP_NAMES = {
    "A": "Quadratic MLP · mod-97",
    "B": "Nanda transformer · mod-113",
    "C": "Transformer · S₅",
    "D": "Quadratic MLP · S₅",
    "E": "Omnigrok MLP · MNIST-1k",
}


def load_run(run_id, runs_root="results/data/runs", hist_root="results/runs"):
    """(row, history) for one completed run id."""
    with open(os.path.join(runs_root, run_id + ".json")) as fh:
        row = json.load(fh)
    hits = glob.glob(os.path.join(hist_root, run_id, "history_*.json"))
    if not hits:
        raise FileNotFoundError(f"no history for {run_id} under {hist_root}")
    with open(hits[0]) as fh:
        history = json.load(fh)
    return row, history


def _thin(xs, ys, target=220):
    """Log-spaced downsample, always keeping the first and last point.

    An SVG path with 4000 nodes is slow to render and no more informative than
    one with 220; log spacing keeps resolution where a log x axis needs it.
    """
    n = len(xs)
    if n <= target:
        return list(zip(xs, ys))
    lo, hi = math.log10(max(xs[0], 1)), math.log10(max(xs[-1], 10))
    want = {0, n - 1}
    for i in range(target):
        t = 10 ** (lo + (hi - lo) * i / (target - 1))
        best = min(range(n), key=lambda j: abs(xs[j] - t))
        want.add(best)
    return [(xs[i], ys[i]) for i in sorted(want)]


def series_for(history):
    """(steps, {key: values}) with step 0 dropped -- a log axis has no zero."""
    steps = history.get("total_steps") or history.get("epoch") or []
    keep = [i for i, s in enumerate(steps) if s and s > 0]
    xs = [float(steps[i]) for i in keep]
    out = {}
    for spec in SERIES:
        vals = history.get(spec["key"]) or []
        if len(vals) >= len(steps):
            out[spec["key"]] = [float(vals[i]) for i in keep]
    return xs, out


def infer_setup(row):
    """Setup label from (dataset, model), for rows written before the tag existed.

    `setup` is a manifest tag, so runs banked earlier -- the whole pre-schema
    cohort -- do not carry one even though their identity is now recorded.
    """
    if row.get("setup"):
        return row["setup"]
    key = (row.get("dataset"), row.get("model"))
    return {("modular", "groknet"): "A", ("modular", "transformer"): "B",
            ("s5", "transformer"): "C", ("s5", "groknet"): "D",
            ("mnist", "mlp"): "E"}.get(key, "?")


def panel_payload(run_id, row, history, section=None):
    xs, ys = series_for(history)
    if not xs:
        return None
    thinned = {k: _thin(xs, v) for k, v in ys.items()}
    t_grok = row.get("t_grok")
    if isinstance(t_grok, str):                       # "inf" survives the JSON
        t_grok = float("inf")
    t_memo = row.get("t_memo")
    if isinstance(t_memo, str):
        t_memo = float("inf")
    setup = infer_setup(row)
    return {
        "id": run_id,
        "setup": setup,
        "setup_name": SETUP_NAMES.get(setup, "?"),
        "mode": row["mode"],
        "dataset": row.get("dataset"),
        "model": row.get("model"),
        "loss": row.get("loss"),
        "detail": _detail(row),
        "section": section,
        "threshold": row.get("grok_threshold"),
        "t_grok": None if t_grok in (None, float("inf")) else t_grok,
        # The DELAY is the phenomenon, so the panel has to be able to draw it.
        # t_grok alone shows when generalisation landed but not what it was
        # waiting for -- and "never trained" and "trained, never generalised"
        # are the same picture without t_memo.
        "t_memo": None if t_memo in (None, float("inf")) else t_memo,
        "final_train": row.get("final_train_acc"),
        "final_test": row.get("final_acc"),
        "wall_s": row.get("wall_s"),
        "series": {k: [[round(x, 1), round(y, 2)] for x, y in v]
                   for k, v in thinned.items()},
    }


def _detail(row):
    """The one-line config description under each panel title."""
    bits = []
    if row["mode"] == "federated":
        bits.append(f"K={row.get('num_clients')} · E={row.get('local_epochs')}"
                    f" · {row.get('partition')}")
    else:
        bits.append("centralized")
    if row.get("dataset") == "mnist":
        bits.append(f"n_train={row.get('n_train')} · batch={row.get('batch_size')}")
    else:
        bits.append(f"α={row.get('alpha')}")
    bits.append(f"{row.get('optimizer')} · {row.get('loss')}")
    # weight decay and seed are what distinguish cells inside a single sweep --
    # without them a wd control panel is indistinguishable from the arm it
    # controls, and two seeds of one cell read as one result.
    # hidden_width is a defining property of the config, not a detail: a capacity
    # sweep's panels are otherwise IDENTICAL in caption and the comparison the
    # page exists to show becomes unreadable.
    if row.get("hidden_width") not in (None, ""):
        bits.append(f"width {row.get('hidden_width')}")
    wd = row.get("weight_decay")
    if wd not in (None, ""):
        bits.append(f"wd={wd}")
    if row.get("seed") not in (None, ""):
        bits.append(f"seed {row.get('seed')}")
    return "  ·  ".join(bits)


def _section_for(row, args):
    """The heading a panel sits under.

    Defaults to the mode, which is what the page has always done. `--section-by`
    points it at any result-row field instead -- `experiment` splits a page into
    exp0 / exp1 / exp2, which is the comparison a reader of the campaign wants,
    and mode cannot express because a section may contain both.
    """
    if not getattr(args, "section_by", None):
        return {"centralized": "Centralized reference",
                "federated": "Federated (FedAvg)"}.get(row.get("mode"), row.get("mode"))
    key = str(row.get(args.section_by, "") or "other")
    return (args.section_labels or {}).get(key, key)


def build(panels, title, heading=None, standfirst=None):
    payload = json.dumps(panels, separators=(",", ":"))
    return (_TEMPLATE.replace("__DATA__", payload)
            .replace("__TITLE__", title)
            .replace("__HEADING__", heading or _DEFAULT_HEADING)
            .replace("__STANDFIRST__", standfirst or _DEFAULT_STANDFIRST))


_DEFAULT_HEADING = "Four new setups, both centrally and under FedAvg — every one groks"
_DEFAULT_STANDFIRST = """Grokking is the gap between the two curves: the model memorises its
  training set early (<b>blue</b> saturates) and only generalises much later
  (<b>orange</b> follows). The shaded band is that gap — from
  <b>t<sub>memo</sub></b> to <b>T<sub>grok</sub></b>. The
  horizontal rule is the dataset's threshold, which is a property of the dataset,
  not a constant: 95% for modular arithmetic, 90% for MNIST, 85% for S₅."""


_TEMPLATE = r"""<title>__TITLE__</title>
<style>
  /* Palette: the validated surfaces are warm-biased neutrals, not pure grey --
     #fcfcfb / #1a1a19. Series hues are categorical slots 1 and 2, and they are
     the ONLY chromatic ink on the page; everything else is text or rule. */
  .viz-root{
    color-scheme:light;
    --surface-1:#fcfcfb; --surface-2:#f4f4f1; --border:#e3e2dd;
    --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#82817b;
    --grid:#e8e7e2; --series-1:#2a78d6; --series-2:#eb6834;
    --serif:ui-serif,"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
    --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
    font-family:var(--sans);
    background:var(--surface-1); color:var(--text-primary);
    /* Full-bleed ground, content still capped at 1400px. The surface has to
       reach the viewport edges: an embedding host paints its OWN background in
       the viewer's theme behind this page, so a centred max-width box leaves
       that host ground showing down both margins -- which is a visible seam
       whenever the two themes' surfaces differ at all. Centring via padding
       rather than `margin:0 auto` keeps the measure without the gap. */
    padding:36px max(24px, calc((100% - 1400px) / 2)) 64px;
    min-height:100vh; box-sizing:border-box;
    line-height:1.5;
  }
  @media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{
    color-scheme:dark;
    --surface-1:#1a1a19; --surface-2:#242423; --border:#3a3a38;
    --text-primary:#fff; --text-secondary:#c3c2b7; --text-muted:#8f8e86;
    --grid:#2f2f2d; --series-1:#3987e5; --series-2:#d95926;
  }}
  :root[data-theme="dark"] .viz-root{
    color-scheme:dark;
    --surface-1:#1a1a19; --surface-2:#242423; --border:#3a3a38;
    --text-primary:#fff; --text-secondary:#c3c2b7; --text-muted:#8f8e86;
    --grid:#2f2f2d; --series-1:#3987e5; --series-2:#d95926;
  }
  .viz-root :focus-visible{outline:2px solid var(--series-1);outline-offset:3px;border-radius:4px}
  @media (prefers-reduced-motion:reduce){.viz-root *,.tipbox{transition:none!important}}

  .viz-root h1{font-family:var(--serif);font-size:27px;font-weight:600;
    margin:0 0 10px;letter-spacing:-.012em;text-wrap:balance;line-height:1.2}
  .viz-root .sub{font-size:14.5px;color:var(--text-secondary);margin:0 0 24px;
    max-width:70ch}
  .viz-root .sub b{color:var(--text-primary);font-weight:600}

  /* Verdict strip: the summary before the detail. */
  .strip{display:flex;gap:0;flex-wrap:wrap;border:1px solid var(--border);
    border-radius:10px;overflow:hidden;margin-bottom:26px;background:var(--surface-2)}
  .stat{flex:1 1 150px;padding:13px 16px;background:var(--surface-1);
    border-right:1px solid var(--border)}
  .stat:last-child{border-right:0}
  .stat .k{font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;
    color:var(--text-muted);font-weight:600;margin-bottom:3px}
  .stat .v{font-family:var(--serif);font-size:23px;font-weight:600;line-height:1.1}
  .stat .v small{font-size:13px;font-weight:400;color:var(--text-secondary);
    font-family:var(--sans)}

  .bar{display:flex;align-items:center;gap:20px;flex-wrap:wrap;
       padding-bottom:14px;margin-bottom:4px;border-bottom:1px solid var(--border)}
  .legend{display:flex;gap:18px;align-items:center}
  .lg{display:flex;align-items:center;gap:7px;font-size:13px;color:var(--text-secondary)}
  .lg i{width:18px;height:2px;border-radius:1px;display:block;flex:none}
  .toggle{margin-left:auto;display:flex;gap:2px;background:var(--surface-2);
          border:1px solid var(--border);border-radius:7px;padding:2px}
  .toggle button{font:inherit;font-size:12.5px;padding:4px 13px;border:0;border-radius:5px;
    background:transparent;color:var(--text-secondary);cursor:pointer}
  .toggle button[aria-pressed="true"]{background:var(--surface-1);color:var(--text-primary);
    box-shadow:0 1px 2px rgba(0,0,0,.09);font-weight:600}

  /* The two bands encode a real distinction, not decoration: same setups, one
     trained centrally and one under FedAvg. */
  .rowlabel{font-family:var(--mono);font-size:11px;font-weight:600;letter-spacing:.09em;
    text-transform:uppercase;color:var(--text-muted);margin:30px 0 13px;
    display:flex;align-items:center;gap:12px}
  .rowlabel::after{content:"";flex:1;height:1px;background:var(--border)}
  .grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(276px,1fr))}
  .card{background:var(--surface-1);border:1px solid var(--border);
    border-radius:10px;padding:15px 15px 11px}
  .card h3{margin:0 0 3px;font-size:14.5px;font-weight:600;font-family:var(--serif)}
  .card h3 em{font-style:normal;color:var(--text-muted);font-weight:400}
  .card .cfg{margin:0 0 4px;font-family:var(--mono);font-size:10.5px;
             color:var(--text-muted);font-variant-numeric:tabular-nums}
  .card svg{display:block;width:100%;height:auto;overflow:visible;touch-action:none}
  .verdict{display:flex;align-items:center;gap:8px;font-size:12px;
           color:var(--text-secondary);margin-top:8px;flex-wrap:wrap}
  .verdict .num{font-family:var(--mono);font-variant-numeric:tabular-nums}
  .pill{font-size:10.5px;font-weight:600;padding:2px 8px;border-radius:99px;
        background:var(--surface-2);border:1px solid var(--border);
        color:var(--text-primary);letter-spacing:.02em}
  .tick{font-size:10px;fill:var(--text-muted);font-variant-numeric:tabular-nums}

  .tipbox{position:fixed;pointer-events:none;z-index:40;background:var(--surface-1);
    border:1px solid var(--border);border-radius:8px;padding:9px 11px;font-size:12px;
    box-shadow:0 4px 18px rgba(0,0,0,.17);opacity:0;transition:opacity .1s;min-width:138px}
  .tipbox .x{color:var(--text-muted);font-size:11px;margin-bottom:6px;
             font-family:var(--mono);font-variant-numeric:tabular-nums}
  .tipbox .r{display:flex;align-items:center;gap:8px;margin-top:3px}
  .tipbox .r i{width:13px;height:2px;border-radius:1px;flex:none}
  .tipbox .r b{margin-left:auto;font-family:var(--mono);
    font-variant-numeric:tabular-nums;font-weight:600}
  .tipbox .r span{color:var(--text-secondary)}

  .tablewrap{overflow-x:auto;margin-top:18px}
  table{border-collapse:collapse;width:100%;font-size:13px;min-width:760px;
        font-variant-numeric:tabular-nums}
  th,td{text-align:right;padding:8px 11px;border-bottom:1px solid var(--border);white-space:nowrap}
  th:first-child,td:first-child{text-align:left}
  td:nth-child(3),td:nth-child(5){font-family:var(--mono);font-size:12px}
  th{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;
     color:var(--text-muted);font-weight:600}
  .hidden{display:none}
  .note{font-size:12.5px;color:var(--text-muted);margin-top:26px;line-height:1.65;
    max-width:76ch;border-top:1px solid var(--border);padding-top:16px}
  .note b{color:var(--text-secondary)}
  .note p{margin:0 0 9px}
</style>

<div class="viz-root">
  <h1>__HEADING__</h1>
  <p class="sub">__STANDFIRST__</p>

  <div class="strip" id="strip"></div>

  <div class="bar">
    <div class="legend">
      <span class="lg"><i style="background:var(--series-1)"></i>Train accuracy</span>
      <span class="lg"><i style="background:var(--series-2)"></i>Test accuracy</span>
    </div>
    <div class="toggle" role="group" aria-label="View">
      <button id="bChart" aria-pressed="true">Charts</button>
      <button id="bTable" aria-pressed="false">Table</button>
    </div>
  </div>

  <div id="chartView"></div>
  <div id="tableView" class="hidden"><div class="tablewrap"></div></div>

  <div class="note" id="footnote"></div>
</div>
<div class="tipbox" id="tip" role="status" aria-live="polite"></div>

<script>
const PANELS = __DATA__;
const SER = [
  {key:"train_acc", label:"Train", css:"--series-1"},
  {key:"test_acc",  label:"Test",  css:"--series-2"}
];
const W=320,H=186,ML=34,MR=14,MT=10,MB=26;
const PW=W-ML-MR, PH=H-MT-MB;
const fmtStep = v => v>=1000 ? (v/1000).toFixed(v>=10000?0:1)+"k" : String(Math.round(v));

function scales(p){
  let lo=Infinity, hi=-Infinity;
  for(const k in p.series) for(const [x] of p.series[k]){ if(x<lo)lo=x; if(x>hi)hi=x; }
  lo=Math.max(lo,1);
  const L0=Math.log10(lo), L1=Math.log10(hi);
  return {
    x: v => ML + PW*(Math.log10(Math.max(v,lo))-L0)/((L1-L0)||1),
    y: v => MT + PH*(1 - Math.max(0,Math.min(100,v))/100),
    lo, hi, L0, L1
  };
}

function panelSVG(p){
  const s=scales(p);
  const ns="http://www.w3.org/2000/svg";
  const svg=document.createElementNS(ns,"svg");
  svg.setAttribute("viewBox",`0 0 ${W} ${H}`);
  svg.setAttribute("role","img");
  svg.setAttribute("tabindex","0");
  svg.setAttribute("aria-label",
    `${p.setup_name}, ${p.mode}. Final train ${p.final_train.toFixed(1)} percent, `+
    `test ${p.final_test.toFixed(1)} percent.`+
    (p.t_grok?` Grokked at ${Math.round(p.t_grok)} steps.`:" Did not reach the threshold."));

  const mk=(t,a)=>{const e=document.createElementNS(ns,t);
    for(const k in a) e.setAttribute(k,a[k]); return e;};

  // horizontal gridlines - solid hairlines, one step off surface
  [0,50,100].forEach(v=>{
    svg.appendChild(mk("line",{x1:ML,x2:W-MR,y1:s.y(v),y2:s.y(v),
      stroke:"var(--grid)","stroke-width":1}));
    const t=mk("text",{x:ML-6,y:s.y(v)+3.2,"text-anchor":"end"});
    t.setAttribute("class","tick"); t.textContent=v; svg.appendChild(t);
  });

  // the dataset's grok threshold
  if(p.threshold!=null){
    svg.appendChild(mk("line",{x1:ML,x2:W-MR,y1:s.y(p.threshold),y2:s.y(p.threshold),
      stroke:"var(--text-muted)","stroke-width":1,"stroke-opacity":.5}));
    const t=mk("text",{x:W-MR,y:s.y(p.threshold)-4,"text-anchor":"end"});
    t.setAttribute("class","tick");
    t.textContent=`grok bar ${p.threshold}%`; svg.appendChild(t);
  }

  // x ticks at decade boundaries
  for(let d=Math.ceil(s.L0); d<=Math.floor(s.L1); d++){
    const v=10**d, x=s.x(v);
    svg.appendChild(mk("line",{x1:x,x2:x,y1:MT+PH,y2:MT+PH+3,
      stroke:"var(--grid)","stroke-width":1}));
    const t=mk("text",{x:x,y:H-MB+15,"text-anchor":"middle"});
    t.setAttribute("class","tick"); t.textContent=fmtStep(v); svg.appendChild(t);
  }
  const xl=mk("text",{x:ML+PW/2,y:H-2,"text-anchor":"middle"});
  xl.setAttribute("class","tick"); xl.textContent="gradient steps (log)";
  svg.appendChild(xl);

  // The delay band: memorised -> generalised. This gap IS grokking, so it is
  // drawn as an extent rather than left implicit between two rules. Neutral
  // fill, never a series hue -- the hues mean train/test and nothing else.
  if(p.t_memo && p.t_grok && p.t_grok>p.t_memo){
    const x0=s.x(p.t_memo), x1=s.x(p.t_grok);
    svg.appendChild(mk("rect",{x:x0,y:MT,width:Math.max(0,x1-x0),height:PH,
      fill:"var(--text-primary)","fill-opacity":.055}));
    const mid=(x0+x1)/2, w=x1-x0;
    if(w>54){                       // only label when it fits without collision
      const t=mk("text",{x:mid,y:MT+PH-6,"text-anchor":"middle"});
      t.setAttribute("class","tick");
      t.textContent="delay "+fmtStep(p.t_grok-p.t_memo); svg.appendChild(t);
    }
  }

  // t_memo rule - dashed, the opening of the band
  if(p.t_memo){
    const x=s.x(p.t_memo);
    svg.appendChild(mk("line",{x1:x,x2:x,y1:MT,y2:MT+PH,
      stroke:"var(--text-muted)","stroke-width":1,"stroke-opacity":.55,
      "stroke-dasharray":"3 3"}));
    const t=mk("text",{x:Math.min(x+4,W-MR-2),y:MT+PH-18});
    t.setAttribute("class","tick");
    t.textContent="t_memo "+fmtStep(p.t_memo); svg.appendChild(t);
  }

  // T_grok rule - solid, so it is not confused with a projection
  if(p.t_grok){
    const x=s.x(p.t_grok);
    svg.appendChild(mk("line",{x1:x,x2:x,y1:MT,y2:MT+PH,
      stroke:"var(--text-muted)","stroke-width":1,"stroke-opacity":.65}));
    const t=mk("text",{x:Math.min(x+4,W-MR-2),y:MT+9});
    t.setAttribute("class","tick");
    t.textContent="T_grok "+fmtStep(p.t_grok); svg.appendChild(t);
  }

  // series - 2px, round caps
  SER.forEach(spec=>{
    const pts=p.series[spec.key]; if(!pts) return;
    const d=pts.map(([x,y],i)=>(i?"L":"M")+s.x(x).toFixed(2)+" "+s.y(y).toFixed(2)).join(" ");
    svg.appendChild(mk("path",{d,fill:"none",stroke:`var(${spec.css})`,
      "stroke-width":2,"stroke-linejoin":"round","stroke-linecap":"round"}));
    const [ex,ey]=pts[pts.length-1];
    svg.appendChild(mk("circle",{cx:s.x(ex),cy:s.y(ey),r:4,
      fill:`var(${spec.css})`,stroke:"var(--surface-1)","stroke-width":2}));
  });

  // crosshair + hit layer
  const cross=mk("line",{x1:0,x2:0,y1:MT,y2:MT+PH,stroke:"var(--text-muted)",
    "stroke-width":1,"stroke-opacity":0});
  svg.appendChild(cross);
  const dots=SER.map(spec=>{
    const c=mk("circle",{r:4.5,fill:`var(${spec.css})`,stroke:"var(--surface-1)",
      "stroke-width":2,opacity:0}); svg.appendChild(c); return c;});
  const hit=mk("rect",{x:ML,y:MT,width:PW,height:PH,fill:"transparent"});
  svg.appendChild(hit);

  const tip=document.getElementById("tip");
  const base=p.series[SER[0].key]||p.series[SER[1].key];
  function show(clientX, clientY, frac){
    const target=10**(s.L0+(s.L1-s.L0)*frac);
    let bi=0,bd=Infinity;
    base.forEach(([x],i)=>{const d=Math.abs(Math.log10(Math.max(x,1))-Math.log10(target));
      if(d<bd){bd=d;bi=i;}});
    const xv=base[bi][0];
    cross.setAttribute("x1",s.x(xv)); cross.setAttribute("x2",s.x(xv));
    cross.setAttribute("stroke-opacity",.45);
    tip.replaceChildren();
    const xd=document.createElement("div"); xd.className="x";
    xd.textContent=`step ${Math.round(xv).toLocaleString()}`; tip.appendChild(xd);
    SER.forEach((spec,si)=>{
      const arr=p.series[spec.key]; if(!arr){dots[si].setAttribute("opacity",0);return;}
      const pt=arr[Math.min(bi,arr.length-1)];
      dots[si].setAttribute("cx",s.x(pt[0])); dots[si].setAttribute("cy",s.y(pt[1]));
      dots[si].setAttribute("opacity",1);
      const r=document.createElement("div"); r.className="r";
      const i=document.createElement("i"); i.style.background=`var(${spec.css})`;
      const nm=document.createElement("span"); nm.textContent=spec.label;
      const b=document.createElement("b"); b.textContent=pt[1].toFixed(1)+"%";
      r.append(i,nm,b); tip.appendChild(r);
    });
    tip.style.opacity=1;
    tip.style.left=Math.min(clientX+14, innerWidth-165)+"px";
    tip.style.top=Math.max(8, clientY-64)+"px";
  }
  function hide(){tip.style.opacity=0;cross.setAttribute("stroke-opacity",0);
    dots.forEach(d=>d.setAttribute("opacity",0));}
  svg.addEventListener("pointermove",ev=>{
    const r=svg.getBoundingClientRect();
    const frac=((ev.clientX-r.left)/r.width*W-ML)/PW;
    if(frac<-0.05||frac>1.05){hide();return;}
    show(ev.clientX,ev.clientY,Math.max(0,Math.min(1,frac)));
  });
  svg.addEventListener("pointerleave",hide);
  svg.addEventListener("blur",hide);
  svg.addEventListener("focus",()=>{const r=svg.getBoundingClientRect();
    show(r.left+r.width*0.8, r.top+30, 0.8);});
  return svg;
}

function card(p){
  const el=document.createElement("div"); el.className="card";
  const h=document.createElement("h3");
  const em=document.createElement("em"); em.textContent=p.setup+" ";
  h.append(em, document.createTextNode(p.setup_name));
  const c=document.createElement("p"); c.className="cfg"; c.textContent=p.detail;
  el.append(h,c,panelSVG(p));
  const v=document.createElement("div"); v.className="verdict";
  const pill=document.createElement("span"); pill.className="pill";
  pill.textContent = p.t_grok ? "grokked" : "not within budget";
  const txt=document.createElement("span"); txt.className="num";
  txt.textContent = `final ${p.final_train.toFixed(1)}% train · `+
                    `${p.final_test.toFixed(1)}% test`;
  v.append(pill,txt); el.appendChild(v);
  return el;
}

// Verdict strip - the summary, before any detail.
(function(){
  const strip=document.getElementById("strip");
  const grokked=PANELS.filter(p=>p.t_grok).length;
  const setups=new Set(PANELS.map(p=>p.setup));
  const fed=PANELS.filter(p=>p.mode==="federated");
  const fedGrok=fed.filter(p=>p.t_grok).length;
  const wall=PANELS.reduce((a,p)=>a+(p.wall_s||0),0);
  [["Setups shown", setups.size, "architectures × tasks"],
   ["Runs grokked", `${grokked}/${PANELS.length}`, "reached the dataset bar"],
   ["Federated", `${fedGrok}/${fed.length}`, "FedAvg, K=5, E=5, iid"],
   ["Compute", (wall/60).toFixed(0), "minutes, one seed each"]
  ].forEach(([k,v,note])=>{
    const d=document.createElement("div"); d.className="stat";
    const kk=document.createElement("div"); kk.className="k"; kk.textContent=k;
    const vv=document.createElement("div"); vv.className="v";
    vv.append(document.createTextNode(String(v)));
    const s=document.createElement("small"); s.textContent=" "+note;
    vv.appendChild(s); d.append(kk,vv); strip.appendChild(d);
  });
})();

const chartView=document.getElementById("chartView");
// Sections come from the payload and keep its order, so the page can be split
// by whatever the comparison actually is -- experiment, mode, setup -- instead
// of always by mode. Panels carry `section` (see --section-by).
(function(){
  const order=[]; const byS=new Map();
  PANELS.forEach(p=>{
    const k=p.section||p.mode;
    if(!byS.has(k)){ byS.set(k,[]); order.push(k); }
    byS.get(k).push(p);
  });
  order.forEach(k=>{
    const rows=byS.get(k); if(!rows.length) return;
    const lab=document.createElement("div"); lab.className="rowlabel"; lab.textContent=k;
    const g=document.createElement("div"); g.className="grid";
    rows.forEach(p=>g.appendChild(card(p)));
    chartView.append(lab,g);
  });
})();

// table view - every plotted value is reachable without hovering
const tv=document.getElementById("tableView").querySelector(".tablewrap");
const tb=document.createElement("table");
const head=document.createElement("tr");
["Setup","Mode","Config","Grok bar","t_memo","T_grok","Delay","Final train","Final test","Wall"]
  .forEach(t=>{const th=document.createElement("th");th.textContent=t;head.appendChild(th);});
tb.appendChild(document.createElement("thead")).appendChild(head);
const body=document.createElement("tbody");
PANELS.forEach(p=>{
  const tr=document.createElement("tr");
  [p.setup+" · "+p.setup_name, p.mode, p.detail,
   p.threshold!=null?p.threshold+"%":"—",
   p.t_memo?Math.round(p.t_memo).toLocaleString():"never memorised",
   p.t_grok?Math.round(p.t_grok).toLocaleString():"not reached",
   (p.t_memo&&p.t_grok)?Math.round(p.t_grok-p.t_memo).toLocaleString():"—",
   p.final_train.toFixed(1)+"%", p.final_test.toFixed(1)+"%",
   p.wall_s!=null?Math.round(p.wall_s)+"s":"—"
  ].forEach(t=>{const td=document.createElement("td");td.textContent=t;tr.appendChild(td);});
  body.appendChild(tr);
});
tb.appendChild(body); tv.appendChild(tb);

const bC=document.getElementById("bChart"), bT=document.getElementById("bTable");
const tvWrap=document.getElementById("tableView");
bC.onclick=()=>{chartView.classList.remove("hidden");tvWrap.classList.add("hidden");
  bC.setAttribute("aria-pressed","true");bT.setAttribute("aria-pressed","false");};
bT.onclick=()=>{tvWrap.classList.remove("hidden");chartView.classList.add("hidden");
  bT.setAttribute("aria-pressed","true");bC.setAttribute("aria-pressed","false");};

// Caveats belong on the page, not in a covering note.
[["How to read the x axis",
  "Both modes are plotted on the compute-matched total_steps axis: a federated round "+
  "of E local steps across K clients touches the same number of per-sample gradients "+
  "as E full-batch steps, so the axes are comparable by gradient work. They are NOT "+
  "comparable by parameter-update count — FedAvg performs K×E updates per round to "+
  "centralized training's E."],
 ["Why the federated runs reach the bar sooner",
  "Every new setup uses AdamW, and client optimizer state is rebuilt each round by "+
  "default, so each round is E bias-corrected cold-start Adam steps. That is a genuine "+
  "no-op for the original GD setup but not here, and it is the most likely explanation "+
  "for the gap — not a federated speedup. The s5_fl_probe manifest carries a 12-run "+
  "A/B on persist_local_opt_state to settle it."],
 ["What this is and is not",
  "One seed per cell, chosen to demonstrate that each setup trains and groks end to "+
  "end. It is not a survival estimate: no fraction-grokked, no confidence interval, "+
  "and no claim about which setup is faster. Those need the 5-seed cells in the "+
  "staged manifests."]
].forEach(([h,t])=>{
  const p=document.createElement("p");
  const b=document.createElement("b"); b.textContent=h+". ";
  p.append(b, document.createTextNode(t));
  document.getElementById("footnote").appendChild(p);
});
</script>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Grokking across setups")
    ap.add_argument("--heading", default=None,
                    help="visible h1; defaults to the four-setups headline")
    ap.add_argument("--standfirst", default=None,
                    help="paragraph under the h1 (HTML allowed)")
    ap.add_argument("--section-by", default=None,
                    help="result-row field to split the page on, e.g. `experiment`. "
                         "Default: mode.")
    ap.add_argument("--section-labels", default=None,
                    help="rename sections, ';'-separated so labels may "
                         "contain commas: 'width=exp0 - width;boundary=exp1 - cliff'")
    ap.add_argument("--runs-root", default="results/data/runs")
    ap.add_argument("--hist-root", default="results/runs")
    args = ap.parse_args()

    # must happen BEFORE any panel is built: _section_for reads it per row
    if args.section_labels:
        args.section_labels = dict(
            kv.split("=", 1) for kv in args.section_labels.split(";") if "=" in kv)

    panels = []
    for run_id in args.runs:
        try:
            row, history = load_run(run_id, args.runs_root, args.hist_root)
        except FileNotFoundError as exc:
            print(f"  skip {run_id}: {exc}")
            continue
        payload = panel_payload(run_id, row, history,
                                _section_for(row, args))
        if payload is None:
            print(f"  skip {run_id}: empty history")
            continue
        panels.append(payload)
        print(f"  {payload['setup']} {payload['mode'][:4]}: "
              f"t_grok={payload['t_grok']} test={payload['final_test']:.1f}% "
              f"({len(next(iter(payload['series'].values())))} pts)")

    if args.section_by:
        # keep the caller's run order inside a section; order sections by first
        # appearance, so `--runs` fully determines the layout
        seen = {}
        for pl in panels:
            seen.setdefault(pl["section"], len(seen))
        panels.sort(key=lambda pl: seen[pl["section"]])
    else:
        panels.sort(key=lambda p: (p["mode"] != "centralized", p["setup"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(build(panels, args.title, args.heading, args.standfirst))
    print(f"\nWrote {len(panels)} panels -> {args.out}")


if __name__ == "__main__":
    main()
