"""Every run in the results table, as one panel per configuration.

`grok_curves.py` renders a handful of chosen runs with full annotation. This is
the other view: the whole corpus at once, one small multiple per *cell* (a config
minus its seed) with every seed drawn, so the curve shape and the seed spread are
legible together. That second part matters -- setup B's T_grok is bimodal across
seeds while setup A's is tight to 6%, and a per-run view hides that.

Panels are titled with only the fields that VARY inside their campaign, so the
label carries the axis being swept rather than restating constants.

    python scripts/plotting/grok_atlas.py --out atlas.html
"""

import argparse
import collections
import csv
import glob
import json
import os

from grok_curves import SETUP_NAMES, _thin, infer_setup, series_for

# Order campaigns by narrative, not alphabetically: what each setup does on its
# own first, then what federation does to it, then the diagnostics.
GROUP_ORDER = [
    ("central_anchor", "Centralized α ladders — where each setup's cliff is"),
    ("c_alpha", "Setup C — does a higher α rescue it?"),
    ("c_capacity", "Setup C — does more capacity rescue it?"),
    ("mnist_wd_band", "MNIST — the Omnigrok weight-decay band"),
    ("mnist_working_point", "MNIST — delay vs shardability"),
    ("wd_grid", "Weight-decay grid (anchor setup)"),
    ("poly_pilot", "Polynomial task gate"),
    ("fl_probe", "FL probe — first federated run of each new setup"),
    ("grok_confirm_fl", "FL confirmation runs"),
    ("mnist_fl", "Federated MNIST"),
    ("probe_rerun", "Probe cells re-run at 5× budget"),
    ("adam_restart", "AdamW optimizer-restart A/B"),
    ("k50_hparam", "K=50 failure — local step size"),
    ("k50_ladder", "K=50 failure — where it breaks"),
    ("probe", "T1 probe (anchor setup)"),
    ("k_fixed_total", "T2 K-breakdown (anchor setup)"),
    ("boundary", "T2 boundary campaign (anchor setup)"),
]

# Fields that can distinguish two cells. Seed is deliberately absent.
CELL_KEYS = ["dataset", "model", "loss", "task", "p", "group_n", "alpha", "n_train",
             "batch_size", "hidden_width", "n_layers", "n_heads", "d_mlp",
             "init_scale", "num_clients", "local_epochs", "num_rounds",
             "fraction_train", "partition", "dirichlet_alpha", "strategy",
             "server_lr", "lr", "weight_decay", "epochs", "mode",
             "persist_local_opt_state"]

SHORT = {"num_clients": "K", "local_epochs": "E", "alpha": "α",
         "n_train": "n", "batch_size": "bs", "hidden_width": "d", "n_heads": "h",
         "d_mlp": "mlp", "weight_decay": "wd", "num_rounds": "R", "epochs": "ep",
         "dirichlet_alpha": "dir", "persist_local_opt_state": "persist",
         "fraction_train": "frac", "server_lr": "slr", "init_scale": "init",
         "n_layers": "L", "group_n": "S", "p": "p", "lr": "lr"}

# Values that speak for themselves — printed bare, with no "key=" prefix.
BARE = {"partition", "task", "strategy"}

# The setup letter and name are already on every card, so repeating the fields
# they encode would spend the label on constants.
LABEL_SKIP = {"dataset", "model", "loss", "mode"}


def _label(key, varying):
    bits = []
    for i in varying:
        name, value = CELL_KEYS[i], key[i]
        if value == "" or name in LABEL_SKIP:
            continue
        bits.append(value if name in BARE else f"{SHORT.get(name, name)}={value}")
    return "  ".join(bits) or "single cell"


def _manifest_backfill(rows, manifest_dir="manifests"):
    """Fill cell keys the CSV lacks from the spec that produced each run.

    Runs banked before a field entered the result row carry a blank for it, and a
    blank collapses distinct configs into one panel — the capacity sweep would
    show 2 cells for 24 runs because n_heads/d_mlp were missing, which is exactly
    the two numbers that sweep exists to compare. The spec is the authority: the
    run id is a hash of it, so the join is exact.
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from fedgrok.manifest import load_manifest

    spec_by_id = {}
    for path in sorted(glob.glob(os.path.join(manifest_dir, "*.jsonl"))):
        for spec in load_manifest(path):
            spec_by_id.setdefault(spec["id"], spec)

    filled = 0
    for r in rows:
        spec = spec_by_id.get(r["id"])
        if not spec:
            continue
        for k in CELL_KEYS:
            if not r.get(k) and spec.get(k) is not None:
                r[k] = str(spec[k])
                filled += 1
    return filled


def build_cells(csv_path, hist_root, max_points):
    rows = list(csv.DictReader(open(csv_path)))
    n = _manifest_backfill(rows)
    if n:
        print(f"  backfilled {n} missing cell-key values from the manifests")
    by_group = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by_group[r.get("group") or "(ungrouped)"][
            tuple(r.get(k, "") for k in CELL_KEYS)].append(r)

    out = []
    for group, cells in by_group.items():
        # Which fields vary is computed WITHIN each setup, not across the whole
        # campaign. A section like central_anchor holds four setups, so lr, width
        # and init_scale all "vary" across it -- but only because the setups
        # differ, not because anything is being swept. Labelling on the section-
        # wide set buries the actual axis (alpha) in six constants.
        setup_of = {key: infer_setup(runs[0]) for key, runs in cells.items()}
        varying_for = {}
        for setup in set(setup_of.values()):
            mine = [k for k in cells if setup_of[k] == setup]
            varying_for[setup] = [i for i, _ in enumerate(CELL_KEYS)
                                  if len({k[i] for k in mine}) > 1]
        panels = []
        for key, runs in cells.items():
            varying = varying_for[setup_of[key]]
            series = []
            for r in runs:
                hits = glob.glob(os.path.join(hist_root, r["id"], "history_*.json"))
                if not hits:
                    continue
                with open(hits[0]) as fh:
                    history = json.load(fh)
                xs, ys = series_for(history)
                if not xs:
                    continue
                series.append({
                    "seed": r.get("seed"),
                    "grok": r["t_grok"] != "inf",
                    **{k: [[int(x), round(y, 1)] for x, y in _thin(xs, v, max_points)]
                       for k, v in ys.items()},
                })
            if not series:
                continue
            groks = sum(1 for r in runs if r["t_grok"] != "inf")
            finite = [float(r["t_grok"]) for r in runs if r["t_grok"] != "inf"]
            r0 = runs[0]
            panels.append({
                "label": _label(key, varying),
                "setup": infer_setup(r0),
                "setup_name": SETUP_NAMES.get(infer_setup(r0), "?"),
                "mode": r0["mode"],
                "threshold": float(r0["grok_threshold"]) if r0.get("grok_threshold") else None,
                "n": len(runs), "grokked": groks,
                "median": sorted(finite)[len(finite) // 2] if finite else None,
                "series": series,
            })
        panels.sort(key=lambda p: (p["setup"], p["label"]))
        out.append({"group": group,
                    "title": dict(GROUP_ORDER).get(group, group),
                    "panels": panels,
                    "runs": sum(p["n"] for p in panels)})
    order = {g: i for i, (g, _) in enumerate(GROUP_ORDER)}
    out.sort(key=lambda s: order.get(s["group"], 99))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="results/data/runs_v2.csv")
    ap.add_argument("--hist-root", default="results/runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-points", type=int, default=44)
    args = ap.parse_args()

    sections = build_cells(args.csv, args.hist_root, args.max_points)
    n_panels = sum(len(s["panels"]) for s in sections)
    n_runs = sum(s["runs"] for s in sections)
    payload = json.dumps(sections, separators=(",", ":"))
    html = _TEMPLATE.replace("__DATA__", payload)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(html)
    for s in sections:
        print(f"  {s['group']:22s} {len(s['panels']):3d} cells, {s['runs']:3d} runs")
    print(f"\n  {n_panels} panels over {n_runs} runs -> {args.out} "
          f"({len(html)/1e6:.1f} MB)")


_TEMPLATE = r"""<title>Grokking atlas — every run</title>
<style>
  .viz-root{
    color-scheme:light;
    --surface-1:#fcfcfb; --surface-2:#f4f4f1; --border:#e3e2dd;
    --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#82817b;
    --grid:#e8e7e2; --series-1:#2a78d6; --series-2:#eb6834;
    --serif:ui-serif,"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
    --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
    font-family:var(--sans); background:var(--surface-1); color:var(--text-primary);
    padding:34px 22px 70px; max-width:1520px; margin:0 auto; line-height:1.5;
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
  h1{font-family:var(--serif);font-size:26px;font-weight:600;margin:0 0 9px;
     letter-spacing:-.012em;text-wrap:balance}
  .sub{font-size:14px;color:var(--text-secondary);margin:0 0 20px;max-width:74ch}
  .sub b{color:var(--text-primary);font-weight:600}
  .controls{position:sticky;top:0;z-index:20;background:var(--surface-1);
    display:flex;gap:16px;align-items:center;flex-wrap:wrap;
    padding:11px 0;margin-bottom:6px;border-bottom:1px solid var(--border)}
  .legend{display:flex;gap:16px}
  .lg{display:flex;align-items:center;gap:7px;font-size:12.5px;color:var(--text-secondary)}
  .lg i{width:18px;height:2px;border-radius:1px;display:block;flex:none}
  label.f{font-size:12.5px;color:var(--text-secondary);display:flex;align-items:center;gap:6px}
  select{font:inherit;font-size:12.5px;padding:3px 7px;border:1px solid var(--border);
    border-radius:6px;background:var(--surface-1);color:var(--text-primary)}
  .count{margin-left:auto;font-size:12.5px;color:var(--text-muted);font-family:var(--mono)}
  section{margin-top:30px}
  section h2{font-family:var(--serif);font-size:17px;font-weight:600;margin:0 0 3px}
  section .meta{font-family:var(--mono);font-size:11px;color:var(--text-muted);
    letter-spacing:.04em;text-transform:uppercase;margin:0 0 12px}
  .grid{display:grid;gap:13px;grid-template-columns:repeat(auto-fill,minmax(232px,1fr))}
  .card{border:1px solid var(--border);border-radius:9px;padding:10px 11px 7px;
    background:var(--surface-1)}
  .card .t{font-family:var(--mono);font-size:10.5px;color:var(--text-primary);
    margin-bottom:1px;word-break:break-word;line-height:1.35}
  .card .s{font-size:10px;color:var(--text-muted);margin-bottom:3px}
  .card svg{display:block;width:100%;height:auto;overflow:visible;touch-action:none}
  .card .v{display:flex;gap:6px;align-items:center;font-size:10.5px;
    color:var(--text-secondary);margin-top:4px;font-family:var(--mono)}
  .dot{width:7px;height:7px;border-radius:99px;flex:none}
  .tick{font-size:9px;fill:var(--text-muted)}
  .tipbox{position:fixed;pointer-events:none;z-index:60;background:var(--surface-1);
    border:1px solid var(--border);border-radius:8px;padding:8px 10px;font-size:12px;
    box-shadow:0 4px 18px rgba(0,0,0,.17);opacity:0;transition:opacity .1s;min-width:150px}
  .tipbox .x{color:var(--text-muted);font-size:11px;margin-bottom:5px;font-family:var(--mono)}
  .tipbox .r{display:flex;align-items:center;gap:8px;margin-top:3px}
  .tipbox .r i{width:13px;height:2px;border-radius:1px;flex:none}
  .tipbox .r b{margin-left:auto;font-family:var(--mono);font-weight:600}
  .tipbox .r span{color:var(--text-secondary)}
  .note{font-size:12.5px;color:var(--text-muted);margin-top:30px;line-height:1.65;
    max-width:78ch;border-top:1px solid var(--border);padding-top:15px}
  .hidden{display:none}
</style>

<div class="viz-root">
  <h1>Grokking atlas — every run in the results table</h1>
  <p class="sub">One panel per <b>configuration</b>, with every seed drawn: train
  accuracy in blue, test in orange, one line per seed. Seeds are overlaid rather
  than averaged because the spread is itself a result — setup&nbsp;B's grok time is
  bimodal across seeds while setup&nbsp;A's is tight to 6%. Faint horizontal rule
  is the dataset's grok threshold. X is gradient steps, log scale.</p>

  <div class="controls">
    <div class="legend">
      <span class="lg"><i style="background:var(--series-1)"></i>Train</span>
      <span class="lg"><i style="background:var(--series-2)"></i>Test</span>
    </div>
    <label class="f">Setup <select id="fSetup"></select></label>
    <label class="f">Mode <select id="fMode"></select></label>
    <label class="f">Outcome <select id="fOut"></select></label>
    <span class="count" id="count"></span>
  </div>

  <div id="body"></div>
  <div class="note" id="note"></div>
</div>
<div class="tipbox" id="tip" role="status" aria-live="polite"></div>

<script>
const DATA = __DATA__;
const W=250,H=132,ML=25,MR=8,MT=7,MB=17,PW=W-ML-MR,PH=H-MT-MB;
const SER=[{k:"train_acc",n:"Train",c:"--series-1"},{k:"test_acc",n:"Test",c:"--series-2"}];
const tip=document.getElementById("tip");

function panelSVG(p){
  const ns="http://www.w3.org/2000/svg";
  let lo=Infinity, hi=-Infinity;
  p.series.forEach(s=>SER.forEach(sp=>(s[sp.k]||[]).forEach(([x])=>{
    if(x<lo)lo=x; if(x>hi)hi=x;})));
  lo=Math.max(lo,1); const L0=Math.log10(lo), L1=Math.log10(Math.max(hi,lo*10));
  const X=v=>ML+PW*(Math.log10(Math.max(v,lo))-L0)/((L1-L0)||1);
  const Y=v=>MT+PH*(1-Math.max(0,Math.min(100,v))/100);
  const mk=(t,a)=>{const e=document.createElementNS(ns,t);for(const k in a)e.setAttribute(k,a[k]);return e;};
  const svg=mk("svg",{viewBox:`0 0 ${W} ${H}`,role:"img",tabindex:"0"});
  svg.setAttribute("aria-label",
    `${p.label}. ${p.grokked} of ${p.n} seeds grokked.`);
  [0,50,100].forEach(v=>{
    svg.appendChild(mk("line",{x1:ML,x2:W-MR,y1:Y(v),y2:Y(v),stroke:"var(--grid)","stroke-width":1}));
    const t=mk("text",{x:ML-4,y:Y(v)+3,"text-anchor":"end"});
    t.setAttribute("class","tick"); t.textContent=v; svg.appendChild(t);
  });
  if(p.threshold!=null)
    svg.appendChild(mk("line",{x1:ML,x2:W-MR,y1:Y(p.threshold),y2:Y(p.threshold),
      stroke:"var(--text-muted)","stroke-width":1,"stroke-opacity":.42}));
  for(let d=Math.ceil(L0); d<=Math.floor(L1); d++){
    const x=X(10**d);
    const t=mk("text",{x:x,y:H-4,"text-anchor":"middle"});
    t.setAttribute("class","tick");
    t.textContent=(10**d>=1000)?(10**d/1000)+"k":10**d; svg.appendChild(t);
  }
  // One line per seed. Thinner and slightly transparent than a single-series
  // chart: overlaid at full weight a 5-seed bundle reads as one solid blob.
  const many=p.series.length>1;
  p.series.forEach(s=>SER.forEach(sp=>{
    const pts=s[sp.k]; if(!pts||!pts.length) return;
    const d=pts.map(([x,y],i)=>(i?"L":"M")+X(x).toFixed(1)+" "+Y(y).toFixed(1)).join(" ");
    svg.appendChild(mk("path",{d,fill:"none",stroke:`var(${sp.c})`,
      "stroke-width":many?1.4:2,"stroke-opacity":many?.8:1,
      "stroke-linejoin":"round","stroke-linecap":"round"}));
  }));
  const cross=mk("line",{x1:0,x2:0,y1:MT,y2:MT+PH,stroke:"var(--text-muted)",
    "stroke-width":1,"stroke-opacity":0}); svg.appendChild(cross);
  svg.appendChild(mk("rect",{x:ML,y:MT,width:PW,height:PH,fill:"transparent"}));
  const base=p.series[0][SER[0].k]||p.series[0][SER[1].k];
  svg.addEventListener("pointermove",ev=>{
    const r=svg.getBoundingClientRect();
    let f=((ev.clientX-r.left)/r.width*W-ML)/PW;
    if(f<-0.05||f>1.05){hide();return;}
    f=Math.max(0,Math.min(1,f));
    const target=10**(L0+(L1-L0)*f);
    let bi=0,bd=Infinity;
    base.forEach(([x],i)=>{const dd=Math.abs(Math.log10(Math.max(x,1))-Math.log10(target));
      if(dd<bd){bd=dd;bi=i;}});
    cross.setAttribute("x1",X(base[bi][0])); cross.setAttribute("x2",X(base[bi][0]));
    cross.setAttribute("stroke-opacity",.45);
    tip.replaceChildren();
    const xd=document.createElement("div"); xd.className="x";
    xd.textContent=`step ${base[bi][0].toLocaleString()} · ${p.series.length} seed(s)`;
    tip.appendChild(xd);
    SER.forEach(sp=>{
      const vals=p.series.map(s=>(s[sp.k]||[])[Math.min(bi,(s[sp.k]||[]).length-1)])
                         .filter(Boolean).map(v=>v[1]);
      if(!vals.length) return;
      const row=document.createElement("div"); row.className="r";
      const i=document.createElement("i"); i.style.background=`var(${sp.c})`;
      const nm=document.createElement("span"); nm.textContent=sp.n;
      const b=document.createElement("b");
      b.textContent = vals.length>1
        ? `${Math.min(...vals).toFixed(0)}–${Math.max(...vals).toFixed(0)}%`
        : `${vals[0].toFixed(1)}%`;
      row.append(i,nm,b); tip.appendChild(row);
    });
    tip.style.opacity=1;
    tip.style.left=Math.min(ev.clientX+14, innerWidth-172)+"px";
    tip.style.top=Math.max(8, ev.clientY-62)+"px";
  });
  function hide(){tip.style.opacity=0;cross.setAttribute("stroke-opacity",0);}
  svg.addEventListener("pointerleave",hide);
  svg.addEventListener("blur",hide);
  return svg;
}

function card(p){
  const el=document.createElement("div"); el.className="card";
  el.dataset.setup=p.setup; el.dataset.mode=p.mode;
  el.dataset.outcome = p.grokked===p.n ? "all" : (p.grokked===0 ? "none" : "partial");
  const t=document.createElement("div"); t.className="t"; t.textContent=p.label;
  const s=document.createElement("div"); s.className="s";
  s.textContent=`${p.setup} · ${p.setup_name}`;
  el.append(t,s,panelSVG(p));
  const v=document.createElement("div"); v.className="v";
  const d=document.createElement("span"); d.className="dot";
  d.style.background = p.grokked===p.n ? "var(--series-1)"
                     : (p.grokked===0 ? "var(--text-muted)" : "var(--series-2)");
  const txt=document.createElement("span");
  txt.textContent = `${p.grokked}/${p.n} grokked` +
    (p.median!=null ? `  ·  T≈${Math.round(p.median).toLocaleString()}` : "");
  v.append(d,txt); el.appendChild(v);
  return el;
}

const body=document.getElementById("body");
DATA.forEach(sec=>{
  const s=document.createElement("section"); s.dataset.group=sec.group;
  const h=document.createElement("h2"); h.textContent=sec.title;
  const m=document.createElement("p"); m.className="meta";
  m.textContent=`${sec.group} · ${sec.panels.length} configs · ${sec.runs} runs`;
  const g=document.createElement("div"); g.className="grid";
  sec.panels.forEach(p=>g.appendChild(card(p)));
  s.append(h,m,g); body.appendChild(s);
});

// One filter row, scoping every section — never per-chart controls.
const setups=[...new Set(DATA.flatMap(s=>s.panels.map(p=>p.setup)))].sort();
function fill(sel,opts){opts.forEach(([v,l])=>{const o=document.createElement("option");
  o.value=v;o.textContent=l;sel.appendChild(o);});}
fill(document.getElementById("fSetup"),
     [["all","All"],...setups.map(s=>[s,`${s} · ${(DATA.flatMap(d=>d.panels)
       .find(p=>p.setup===s)||{}).setup_name||s}`])]);
fill(document.getElementById("fMode"),
     [["all","All"],["centralized","Centralized"],["federated","Federated"]]);
fill(document.getElementById("fOut"),
     [["all","All"],["all_grok","All seeds grokked"],["partial","Partial"],
      ["none","None grokked"]]);
function apply(){
  const su=fSetup.value, mo=fMode.value, ou=fOut.value;
  let shown=0;
  document.querySelectorAll("section").forEach(sec=>{
    let any=0;
    sec.querySelectorAll(".card").forEach(c=>{
      const ok=(su==="all"||c.dataset.setup===su)
            && (mo==="all"||c.dataset.mode===mo)
            && (ou==="all"|| (ou==="all_grok"&&c.dataset.outcome==="all")
                          || (ou==="partial"&&c.dataset.outcome==="partial")
                          || (ou==="none"&&c.dataset.outcome==="none"));
      c.classList.toggle("hidden",!ok); if(ok){any++;shown++;}
    });
    sec.classList.toggle("hidden",any===0);
  });
  document.getElementById("count").textContent=`${shown} configs shown`;
}
["fSetup","fMode","fOut"].forEach(id=>document.getElementById(id).onchange=apply);
apply();

[["Seeds are overlaid, not averaged","A cell with 5 seeds draws 5 blue and 5 orange lines. Where they bundle tightly the setup is reproducible; where they fan out the grok time is seed-dependent, which changes how many seeds a downstream comparison needs."],
 ["Curves are thinned for rendering","Log-uniform downsampling to ≤44 points per series. Grokked counts and median T_grok come from the recorded result rows, not the drawn points."],
 ["The x axis is compute-matched","Federated runs plot total_steps — local steps weighted by participation — so a federated panel and a centralized one are comparable by gradient work, though not by parameter-update count."]
].forEach(([h,t])=>{const p=document.createElement("p");
  const b=document.createElement("b"); b.textContent=h+". ";
  p.append(b,document.createTextNode(t)); document.getElementById("note").appendChild(p);});
</script>
"""


if __name__ == "__main__":
    main()
