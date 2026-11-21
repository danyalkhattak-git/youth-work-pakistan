# =====================================================================================
# Children & Youth at Work: one-screen interactive district dashboard (PBS LFS 2020-21)
# Run:  python app.py  ->  http://127.0.0.1:8050      (or from Jupyter:  %run app.py)
# Designed for a full-screen 1920x1080 display (press F11 in the browser).
# Reads the CSVs + districts.topojson in data/, produced by prep/lfs_dashboard_prep.py
# =====================================================================================
import json, os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, ctx

try:                                                       # data/ folder next to this file (or the notebook's folder)
    BASE = Path(__file__).resolve().parent
except NameError:
    BASE = Path.cwd()
DATA_DIR = os.environ.get("DASH_DATA_DIR", str(BASE / "data"))
DEFAULT_PROVINCE, DEFAULT_DISTRICT = "Khyber Pakhtunkhwa", "KARAK"   # district "__avg__" = province view

# ------------------------------------------------------------------ data
rd = lambda f: pd.read_csv(os.path.join(DATA_DIR, f))
dim_d, dim_i, scores = rd("dim_district.csv"), rd("dim_indicator.csv"), rd("district_scores.csv")
summ, fact, trans = rd("district_summary.csv"), rd("fact_indicators.csv"), rd("transition.csv")
dim_i = dim_i.sort_values("sort").reset_index(drop=True)
NAME = dict(zip(dim_d.district, dim_d.district_name))
DPROV = dict(zip(dim_d.district, dim_d.province))
IND = dim_i.set_index("indicator_id")
F = fact.set_index(["geo_level", "geo_key", "indicator_id", "sex"]).sort_index()
SUMM = summ.set_index("district")
PROVS = ["Balochistan", "Sindh", "Khyber Pakhtunkhwa", "Punjab"]
PROV_Y = {"Punjab": 1, "Khyber Pakhtunkhwa": 2, "Sindh": 3, "Balochistan": 4}
PROV_SHORT = {"Khyber Pakhtunkhwa": "KP", "Punjab": "Punjab", "Sindh": "Sindh", "Balochistan": "Balochistan"}


def topo_to_geo(path, obj="districts"):
    """Decode quantised TopoJSON into GeoJSON (no extra libraries)."""
    t = json.load(open(path))
    sx, sy = t["transform"]["scale"]
    tx, ty = t["transform"]["translate"]
    arcs = []
    for a in t["arcs"]:
        x = y = 0
        pts = []
        for dx, dy in a:
            x += dx
            y += dy
            pts.append([x * sx + tx, y * sy + ty])
        arcs.append(pts)

    def ring(ids):
        out = []
        for i in ids:
            seg = arcs[i] if i >= 0 else arcs[~i][::-1]
            out.extend(seg if not out else seg[1:])
        return out

    feats = []
    for g in t["objects"][obj]["geometries"]:
        if g["type"] == "Polygon":
            coords = [ring(r) for r in g["arcs"]]
        elif g["type"] == "MultiPolygon":
            coords = [[ring(r) for r in p] for p in g["arcs"]]
        else:
            continue
        feats.append({"type": "Feature", "id": g["properties"]["district"], "properties": g["properties"],
                      "geometry": {"type": g["type"], "coordinates": coords}})
    return {"type": "FeatureCollection", "features": feats}


GEO = topo_to_geo(os.path.join(DATA_DIR, "districts.topojson"))
KX = np.cos(np.radians(30))      # simple equirectangular projection for Pakistan's latitude


def ring_area(r):
    x, y = np.array(r).T
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


SHAPES = []                      # (key, xs, ys, cx, cy, area): outer rings only, drawn largest first
for f in GEO["features"]:
    polys = [f["geometry"]["coordinates"]] if f["geometry"]["type"] == "Polygon" else f["geometry"]["coordinates"]
    xs, ys, best, area = [], [], None, 0
    for poly in polys:
        outer = poly[0]
        a = ring_area(outer)
        area += a
        if best is None or a > ring_area(best):
            best = outer
        xs += [p[0] * KX for p in outer] + [None]
        ys += [p[1] for p in outer] + [None]
    bx, by = np.array(best).T
    SHAPES.append((f["id"], xs, ys, bx.mean() * KX, by.mean(), area))
SHAPES.sort(key=lambda t: -t[5])
MAP_KEYS = [t[0] for t in SHAPES]

# ------------------------------------------------------------------ style
FONT = "Segoe UI, Inter, Helvetica, Arial, sans-serif"
BG, PANEL, FRAME = "#0D1321", "#0F1626", "#7B86D9"
TXT, MUTED, GRID = "#F2F5FA", "#A9B4C6", "#1F2A40"
SLATE, MAUVE, PINK, SALMON, SKY, PALE = "#3E5C80", "#7A5C82", "#D66F8E", "#F57A7A", "#8FB8DE", "#E8EEF6"
GOLD, WORSE, BETTER = "#FFD166", "#F57A7A", "#6FD6A8"
GIRL, BOY = PINK, SKY
MAP_SCALE = ["#EEF2F8", "#C9B6D9", "#D66F8E", "#A23B72"]
STATUS = [("School", SKY), ("School + work", "#9BD08F"), ("Work", SALMON), ("Neither", MAUVE)]
FRAMED = dict(background=PANEL, border=f"2px solid {FRAME}", borderRadius="14px", boxSizing="border-box")
PLAIN = {"C1": "Out of school (age 10-14)", "C2": "Out of school (age 15-17)", "L1": "Children working (10-14)",
         "L2": "Teens working (15-17)", "L3": "Working, not in school (10-17)", "Y1": "Not in work or school (15-24)",
         "Y2": "Jobless, looking for work (15-24)", "W1": "Women in work (15-64)",
         "G1": "Girls-boys gap: not in work/school", "G2": "Girls-boys gap: out of school"}
SHORT = {"C1": "Out of school 10-14", "C2": "Out of school 15-17", "L1": "Working 10-14", "L2": "Working 15-17",
         "L3": "Working, no school", "Y1": "No work or school", "Y2": "Jobless, looking", "W1": "Women in work",
         "G1": "Gap: no work/school", "G2": "Gap: out of school"}
MEANING = {"C1": "of children aged 10-14 are not in school", "C2": "of teens aged 15-17 are not in school",
           "L1": "of children aged 10-14 are working", "L2": "of teens aged 15-17 are working",
           "L3": "of 10-17 year olds work and are not in school", "Y1": "of 15-24 year olds are not in work or school",
           "Y2": "of young job-seekers can't find work", "W1": "of women aged 15-64 work or seek work",
           "G1": "points more girls than boys are not in work or school",
           "G2": "points more girls than boys are out of school"}


def base(fig, **kw):
    kw.setdefault("margin", dict(l=4, r=4, t=4, b=4))
    fig.update_layout(font=dict(family=FONT, color=TXT, size=15), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", autosize=True, showlegend=kw.pop("showlegend", False),
                      uniformtext=dict(minsize=13, mode="show"),
                      hoverlabel=dict(font_family=FONT, font_size=15, bgcolor=BG, bordercolor=FRAME), **kw)
    return fig


def unit(ind):
    return " pp" if IND.loc[ind, "unit"] == "pp" else "%"


def fmt(v, ind, d=1):
    return "n/a" if v is None or pd.isna(v) else f"{v:.{d}f}{unit(ind).strip() if unit(ind) == '%' else ' pts'}"


def get(level, key, ind, sex):
    k = (level, key, ind, sex)
    return F.loc[k] if k in F.index else None


def val(level, key, ind, sex):
    r = get(level, key, ind, sex)
    return np.nan if r is None else (r.value_shown if level == "District" else r.value)


def count(level, key, ind, sex):
    r = get(level, key, ind, sex)
    return np.nan if r is None else r.est_count


def big(v):
    if v is None or pd.isna(v):
        return "–"
    return f"{v / 1e6:.2f}M" if v >= 1e6 else (f"{v / 1e3:.0f}K" if v >= 1e3 else f"{v:.0f}")


def selection(prov, dist):
    if dist and dist != "__avg__":
        return "district", dist, DPROV[dist]
    if prov and prov != "Pakistan":
        return "province", None, prov
    return "national", None, None


def geo_of(mode, key, prov):
    return {"district": ("District", key), "province": ("Province", prov), "national": ("National", "Pakistan")}[mode]


def area_name(mode, key, prov):
    return NAME[key] if mode == "district" else (prov if mode == "province" else "Pakistan")


def dist_rows(ind, sex):
    d = fact[(fact.geo_level == "District") & (fact.indicator_id == ind) & (fact.sex == sex) & fact.value_shown.notna()].copy()
    d["need"] = d.value_shown * IND.loc[ind, "need_direction"]
    d["name"] = d.geo_key.map(NAME)
    return d


def status_by_age(mode, key, prov, sex):
    lvl, gk = geo_of(mode, key, prov)
    t = trans[(trans.geo_level == lvl) & (trans.geo_key == gk) & (trans.sex == sex)].copy()
    t["status"] = t.status.map({"In school only": "School", "In school and working": "School + work",
                                "Working only": "Work", "Neither": "Neither"})
    pv = t.pivot_table(index="age", columns="status", values="share").reindex(columns=[s for s, _ in STATUS]).fillna(0)
    return pv.div(pv.sum(axis=1), axis=0) * 100


# ------------------------------------------------------------------ figures
def spark(y, x=None, dot=None):
    x = list(range(len(y))) if x is None else x
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color="#D5DCE6", width=2, shape="spline"),
                               fill="tozeroy", fillcolor="rgba(213,220,230,0.18)", hoverinfo="skip"))
    if dot is not None:
        fig.add_trace(go.Scatter(x=[dot[0]], y=[dot[1]], mode="markers", marker=dict(size=10, color=SALMON),
                                 hoverinfo="skip"))
    return base(fig, xaxis=dict(visible=False), yaxis=dict(visible=False, rangemode="tozero"))


def fig_worst(ind, sex, mode, key, prov):
    d = dist_rows(ind, sex)
    if mode != "national":
        d = d[d.province == prov]
    d = d.sort_values("need", ascending=False).reset_index(drop=True)
    d["rank"] = d.index + 1
    top = d.head(8)
    if mode == "district" and key in set(d.geo_key) and key not in set(top.geo_key):
        top = pd.concat([top, d[d.geo_key == key]])
    top = top.iloc[::-1]
    cols = [GOLD if k == key else PINK for k in top.geo_key]      # worst districts one colour, selected district gold
    fig = go.Figure(go.Bar(x=top.value_shown, y=[f"{r}. {n}" for r, n in zip(top["rank"], top.name)], orientation="h",
                           marker=dict(color=cols), text=[f"{v:.0f}%" for v in top.value_shown],
                           textposition="outside", cliponaxis=False, textfont=dict(size=15, color=TXT),
                           customdata=top.geo_key, hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    ref = val("National", "Pakistan", ind, sex) if mode == "national" else val("Province", prov, ind, sex)
    if pd.notna(ref):
        fig.add_vline(x=ref, line=dict(color=PALE, width=1.5, dash="dot"))
    return base(fig, bargap=0.25, margin=dict(l=4, r=40, t=4, b=4), xaxis=dict(visible=False),
                yaxis=dict(tickfont=dict(size=15, color=TXT), automargin=True))


def fig_gender(mode, key, prov):
    ids = ["C1", "C2", "L2", "Y1", "Y2"][::-1]
    lvl, gk = geo_of(mode, key, prov)
    labs = [SHORT[i] for i in ids]
    fig = go.Figure()
    for sx, colr, nm in [("Male", BOY, "Boys"), ("Female", GIRL, "Girls")]:
        v = [val(lvl, gk, i, sx) for i in ids]
        fig.add_trace(go.Bar(y=labs, x=v, orientation="h", name=nm, marker_color=colr,
                             text=[f"{x:.0f}%" if pd.notna(x) else "" for x in v], textposition="outside",
                             cliponaxis=False, textfont=dict(size=14, color=TXT),
                             hovertemplate=nm + ": %{x:.1f}%<extra></extra>"))
    return base(fig, barmode="group", bargap=0.25, bargroupgap=0.06, margin=dict(l=4, r=40, t=4, b=4),
                xaxis=dict(visible=False, range=[0, 110]), yaxis=dict(tickfont=dict(size=15, color=TXT), automargin=True))


def fig_rings(mode, key, prov, sex):
    pv = status_by_age(mode, key, prov, sex)
    bands = [("Age 10-14", 10, 14, SLATE), ("Age 15-17", 15, 17, MAUVE), ("Age 18-24", 18, 24, PINK)]
    fig = go.Figure()
    for j, (lab, a, b, colr) in enumerate(bands):
        sch = (pv.loc[a:b, "School"] + pv.loc[a:b, "School + work"]).mean() if len(pv) else np.nan
        sch = 0 if pd.isna(sch) else sch
        pad = 0.115 * j
        fig.add_trace(go.Pie(values=[sch * 0.75, (100 - sch) * 0.75, 25], hole=0.80, sort=False, direction="clockwise",
                             rotation=0, marker=dict(colors=[colr, "#C9D1DD", "rgba(0,0,0,0)"], line=dict(width=0)),
                             textinfo="none", hoverinfo="skip", showlegend=False,
                             domain=dict(x=[0.02 + pad, 0.98 - pad], y=[0.02 + pad, 0.98 - pad])))
        fig.add_annotation(x=0.47, y=0.985 - j * 0.115 * 0.96, xref="paper", yref="paper", xanchor="right",
                           showarrow=False, text=f"{lab}  <b>{sch:.0f}%</b>",
                           font=dict(size=15, color=TXT))
    return base(fig)


def fig_treemap(ind, sex, mode, key, prov):
    d = dist_rows(ind, sex)
    if mode != "national":
        d = d[d.province == prov]
    d["size"] = d.est_count if unit(ind) == "%" else d.geo_key.map(dict(zip(dim_d.district, dim_d.population_est)))
    d = d[d["size"].fillna(0) > 0].sort_values("size", ascending=False)
    labels = [f"<b>{n}</b><br>{big(c)}" if unit(ind) == "%" else f"<b>{n}</b>" for n, c in zip(d.name, d["size"])]
    line_w = [4 if k == key else 1 for k in d.geo_key]
    line_c = [GOLD if k == key else PANEL for k in d.geo_key]
    fig = go.Figure(go.Treemap(labels=labels, parents=[""] * len(d), values=d["size"], customdata=d.geo_key,
                               marker=dict(colors=d.value_shown, colorscale=[PALE, SKY, PINK, MAUVE, SLATE],
                                           line=dict(width=line_w, color=line_c), pad=dict(t=0, l=0, r=0, b=0)),
                               textfont=dict(size=15), tiling=dict(pad=2), branchvalues="total", pathbar=dict(visible=False),
                               hovertemplate="%{label}<br>rate: %{color:.1f}<extra></extra>", root_color=PANEL))
    return base(fig, margin=dict(l=2, r=2, t=2, b=2))


def fig_map(ind, sex, mode, key, prov):
    from plotly.colors import sample_colorscale
    d = dist_rows(ind, sex).set_index("geo_key")
    scale = MAP_SCALE if IND.loc[ind, "need_direction"] == 1 else MAP_SCALE[::-1]
    vmin, vmax = (d.value_shown.min(), d.value_shown.max()) if len(d) else (0, 1)
    fig = go.Figure()
    for k, xs, ys, cx, cy, _ in SHAPES:
        v = d.value_shown.get(k, np.nan)
        colr = "#3A4560" if pd.isna(v) else sample_colorscale(scale, [(v - vmin) / ((vmax - vmin) or 1)])[0]
        fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", fillcolor=colr, mode="lines",
                                 line=dict(color=PANEL, width=0.6), hoverinfo="skip"))
    if mode != "national":
        for k, xs, ys, *_ in SHAPES:
            if DPROV.get(k) == prov:
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=SKY, width=1.6), hoverinfo="skip"))
    if mode == "district":
        for k, xs, ys, cx, cy, _ in SHAPES:
            if k == key:
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=GOLD, width=4), hoverinfo="skip"))
                fig.add_annotation(x=cx, y=cy, text=f"<b>{NAME[key]}</b>", showarrow=True, arrowhead=0, ax=60, ay=-35,
                                   arrowcolor=GOLD, font=dict(color=GOLD, size=16), bgcolor="rgba(13,19,33,.85)")
    hk = [t[0] for t in SHAPES]
    fig.add_trace(go.Scatter(x=[t[3] for t in SHAPES], y=[t[4] for t in SHAPES], mode="markers",
                             marker=dict(size=18, opacity=0), customdata=hk, hoverinfo="text",
                             hovertext=[f"<b>{NAME.get(k, k)}</b>: {fmt(d.value_shown.get(k, np.nan), ind)}" for k in hk]))
    return base(fig, hovermode="closest", dragmode="pan", margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1))


def fig_sankey(mode, key, prov):
    lvl, gk = geo_of(mode, key, prov)
    pops, shares = {}, {}
    for sx in ["Female", "Male"]:
        v, c = val(lvl, gk, "Y1", sx), count(lvl, gk, "Y1", sx)
        if pd.isna(v) or pd.isna(c) or v == 0:
            return None
        pops[sx] = c / (v / 100)
        pv = status_by_age(mode, key, prov, sx)
        shares[sx] = pv.loc[15:24].mean() / 100
    nodes = [f"Young people 15-24<br>{big(sum(pops.values()))}", f"Girls {big(pops['Female'])}",
             f"Boys {big(pops['Male'])}"]
    out = {}
    for st, _ in STATUS:
        out[st] = sum(pops[sx] * shares[sx][st] for sx in pops)
    nodes += [f"{st} {big(out[st])}" for st, _ in STATUS]
    src, tgt, vals, lcol = [0, 0], [1, 2], [pops["Female"], pops["Male"]], ["rgba(214,111,142,.45)", "rgba(143,184,222,.45)"]
    for j, sx in enumerate(["Female", "Male"]):
        for k, (st, colr) in enumerate(STATUS):
            src.append(1 + j)
            tgt.append(3 + k)
            vals.append(pops[sx] * shares[sx][st])
            lcol.append("rgba(214,111,142,.35)" if sx == "Female" else "rgba(143,184,222,.35)")
    fig = go.Figure(go.Sankey(
        arrangement="snap", textfont=dict(size=15, color=TXT, family=FONT),
        node=dict(label=nodes, pad=14, thickness=16, line=dict(width=0),
                  color=[SLATE, GIRL, BOY] + [c for _, c in STATUS]),
        link=dict(source=src, target=tgt, value=vals, color=lcol, hovertemplate="%{value:,.0f}<extra></extra>")))
    return base(fig, margin=dict(l=4, r=4, t=8, b=8))


def swarm_offsets(v, dy, dx, step=0.02):
    """Greedy beeswarm: place each dot (sorted by value) at the smallest sideways offset that doesn't overlap."""
    cands = sorted({0.0} | {sg * k * step for k in range(1, 200) for sg in (1, -1)}, key=abs)
    placed, out = [], np.zeros(len(v))
    for i in np.argsort(v):
        near = [(po, pv) for po, pv in placed if abs(v[i] - pv) < dy]
        for off in cands:
            if all(((off - po) / dx) ** 2 + ((v[i] - pv) / dy) ** 2 >= 1 for po, pv in near):
                break
        placed.append((off, v[i]))
        out[i] = off
    return out


def fig_swarm(ind, sex, mode, key, prov):
    d = dist_rows(ind, sex)
    if not len(d):
        return base(go.Figure())
    lo, hi = d.value_shown.min(), d.value_shown.max()
    dy = (hi - lo or 1) * 0.0135                # dot diameter in value units (tuned for the tall right-hand panel)
    order = ["Punjab", "Khyber Pakhtunkhwa", "Sindh", "Balochistan"]
    pcol = {"Punjab": SKY, "Khyber Pakhtunkhwa": PINK, "Sindh": "#9BD08F", "Balochistan": "#B9A3E3"}
    fig, sel_xy = go.Figure(), None
    for j, p in enumerate(order):
        g = d[d.province == p].reset_index(drop=True)
        if not len(g):
            continue
        x = j + np.clip(swarm_offsets(g.value_shown.to_numpy(), dy, 0.115, step=0.01), -0.47, 0.47)
        dim = mode != "national" and p != prov          # fade the other provinces when one is selected
        sel = [k == key for k in g.geo_key]
        if any(sel):
            sel_xy = (x[sel.index(True)], g.value_shown[sel.index(True)])
        fig.add_trace(go.Scatter(
            x=x, y=g.value_shown, mode="markers", customdata=g.geo_key,
            marker=dict(size=[14 if s_ else 9 for s_ in sel], color=[GOLD if s_ else pcol[p] for s_ in sel], opacity=0.5 if dim else 1,
                        line=dict(width=[2 if s_ else 0.6 for s_ in sel], color=[BG if s_ else PANEL for s_ in sel])),
            hovertext=[f"<b>{n}</b>: {fmt(v, ind)}" for n, v in zip(g.name, g.value_shown)], hoverinfo="text"))
        ref = val("Province", p, ind, sex)
        if pd.notna(ref):
            fig.add_shape(type="line", x0=j - 0.42, x1=j + 0.42, y0=ref, y1=ref, line=dict(color=PALE, width=1.5, dash="dot"))
    if mode == "district" and sel_xy:
        fig.add_annotation(x=sel_xy[0], y=sel_xy[1], text=f"<b>{NAME[key]}</b>", showarrow=True,
                           arrowhead=0, ax=0, ay=-28, arrowcolor=GOLD, font=dict(color=GOLD, size=14),
                           bgcolor="rgba(13,19,33,.85)")
    pad = (hi - lo or 1) * 0.06
    return base(fig, hovermode="closest", margin=dict(l=4, r=4, t=4, b=4),
                xaxis=dict(tickvals=list(range(4)), ticktext=[PROV_SHORT[p] for p in order], range=[-0.55, 3.55],
                           tickfont=dict(size=14, color=TXT), showgrid=False, zeroline=False, fixedrange=True),
                yaxis=dict(range=[lo - pad, hi + pad], gridcolor=GRID, zeroline=False, tickfont=dict(size=13, color=MUTED),
                           ticksuffix="%" if unit(ind) == "%" else "", fixedrange=True))


# ------------------------------------------------------------------ text
def kpi_spark(ind, sex, mode, key, prov):
    d = dist_rows(ind, sex)
    pool = d if mode == "national" else d[d.province == prov]
    pool = pool.sort_values("value_shown").reset_index(drop=True)
    dot = None
    if mode == "district" and key in set(pool.geo_key):
        i = int(pool.index[pool.geo_key == key][0])
        dot = (i, pool.value_shown.iloc[i])
    return spark(list(pool.value_shown), dot=dot)


def heads_up(ind, sex, mode, key, prov):
    lvl, gk = geo_of(mode, key, prov)
    area = area_name(mode, key, prov)
    v = val(lvl, gk, ind, sex)
    pts = []
    pts.append([html.B(fmt(v, ind)), f" {MEANING[ind]} in {area}."])
    if mode != "national":
        clvl, ck, cname = ("Province", prov, prov) if mode == "district" else ("National", "Pakistan", "Pakistan")
        c = val(clvl, ck, ind, sex)
        if pd.notna(v) and pd.notna(c):
            worse = (v - c) * IND.loc[ind, "need_direction"] > 0
            pts.append([f"That is ", html.B(f"{abs(v - c):.1f} points {'worse' if worse else 'better'}",
                                            style=dict(color=WORSE if worse else BETTER)),
                        f" than {cname} ({c:.1f}%)."])
    if mode == "district":
        d = dist_rows(ind, sex)
        pool = d[d.province == prov].sort_values("need", ascending=False).reset_index(drop=True)
        r = pool.index[pool.geo_key == key]
        if len(r):
            pts.append([f"Ranks ", html.B(f"{r[0] + 1} of {len(pool)}"), f" districts in {PROV_SHORT[prov]} (1 = worst)."])
    f, m = val(lvl, gk, "Y1", "Female"), val(lvl, gk, "Y1", "Male")
    if pd.notna(f) and pd.notna(m):
        pts.append(["Young women: ", html.B(f"{f:.0f}%", style=dict(color=GIRL)), " not in work or school, vs ",
                    html.B(f"{m:.0f}%", style=dict(color=BOY)), " of young men."])
    pv = status_by_age(mode, key, prov, sex)
    if 16 in pv.index:
        pts.append([html.B(f"{pv.loc[16, 'School'] + pv.loc[16, 'School + work']:.0f}%"), " are still in school at age 16."])
    return html.Ul([html.Li(p, style=dict(marginBottom="12px")) for p in pts],
                   style=dict(paddingLeft="20px", margin="8px 0 0 0"))


# ------------------------------------------------------------------ app
def dist_options(prov):
    if prov == "Pakistan":
        return [{"label": "All districts", "value": "__avg__"}]
    ds = dim_d[dim_d.province == prov].sort_values("district_name")
    return [{"label": "Province average", "value": "__avg__"}] + \
           [{"label": n, "value": k} for k, n in zip(ds.district, ds.district_name)]


app = Dash(__name__, title="Children & Youth at Work")
app.index_string = app.index_string.replace("</head>", """<style>
html, body {margin:0; background:#0D1321; overflow:hidden;}
.flt .Select-control, .flt .Select-menu-outer {background:#C9D1DD !important; border:none !important; font-size:15px;}
</style></head>""")
GCFG = {"displayModeBar": False, "responsive": True}
H = lambda t, id_=None: html.Div(t, id=id_, style=dict(fontSize="19px", fontWeight=400, color=TXT, marginBottom="4px",
                                                       whiteSpace="nowrap", overflow="hidden", textOverflow="ellipsis"))
graph = lambda id_, cfg=GCFG: dcc.Graph(id=id_, config=cfg, style=dict(flex="1 1 auto", minHeight=0, height="100%"))


def cell(title_id, graph_id, area, cfg=GCFG):
    return html.Div([H("", title_id), graph(graph_id, cfg)],
                    style=dict(gridArea=area, display="flex", flexDirection="column", minHeight=0, minWidth=0))


def kpi(n):
    return html.Div([html.Div(id=f"k{n}_lab", style=dict(fontSize="18px", color=TXT, textAlign="center")),
                     html.Div(id=f"k{n}_val", style=dict(fontSize="30px", fontWeight=700, color=TXT, textAlign="center",
                                                          lineHeight="1.15")),
                     dcc.Graph(id=f"k{n}_sp", config=GCFG, style=dict(flex="1 1 auto", minHeight=0))],
                    style={**FRAMED, "display": "flex", "flexDirection": "column", "padding": "8px 14px"})


flabel = lambda t: html.Div(t, style=dict(fontSize="15px", fontWeight=600, color=TXT, marginBottom="4px"))
app.layout = html.Div(style=dict(display="grid", gridTemplateColumns="300px 1fr", gridTemplateRows="150px 1fr", gap="12px",
                                 padding="12px", height="100vh", boxSizing="border-box", background=BG, fontFamily=FONT,
                                 color=TXT), children=[
    html.Div([html.Div("Children & Youth at Work", style=dict(fontSize="26px", lineHeight="1.15")),
              html.Div("Pakistan · LFS 2020-21", style=dict(fontSize="16px", color=MUTED, marginTop="6px"))],
             style={**FRAMED, "background": "#241B2F", "padding": "18px", "display": "flex", "flexDirection": "column",
                    "justifyContent": "center"}),
    html.Div(style=dict(display="grid", gridTemplateColumns="1fr 1fr 1fr 1.5fr", gap="12px", minHeight=0), children=[
        kpi(1), kpi(2), kpi(3),
        html.Div(className="flt", style={**FRAMED, "padding": "12px 16px", "display": "grid", "gap": "8px 14px",
                                         "gridTemplateColumns": "1fr 1fr"}, children=[
            html.Div([flabel("Province"), dcc.Dropdown(id="prov", clearable=False, searchable=False, value=DEFAULT_PROVINCE,
                      style=dict(color=BG), options=[{"label": "Pakistan", "value": "Pakistan"}] +
                                                    [{"label": p, "value": p} for p in PROVS])]),
            html.Div([flabel("District"), dcc.Dropdown(id="dist", clearable=False, value=DEFAULT_DISTRICT,
                      style=dict(color=BG), options=dist_options(DEFAULT_PROVINCE))]),
            html.Div([flabel("Indicator"), dcc.Dropdown(id="ind", clearable=False, searchable=False, value="Y1",
                      style=dict(color=BG), options=[{"label": PLAIN[i], "value": i} for i in dim_i.indicator_id])]),
            html.Div([flabel("Sex"), dcc.Dropdown(id="sex", clearable=False, searchable=False, value="All",
                      style=dict(color=BG), options=["All", "Female", "Male"])]),
        ]),
    ]),
    html.Div([html.Div(id="hu_title", style=dict(fontSize="19px", fontWeight=600)),
              html.Div(id="headsup", style=dict(fontSize="17px", lineHeight="1.4", color=TXT)),
              html.Div("Click a district on the map, bars, treemap or dot chart to select it.",
                       style=dict(fontSize="14px", color=MUTED, marginTop="auto")),
              html.Div("Source: PBS LFS 2020-21 (weighted)", style=dict(fontSize="14px", color=MUTED, marginTop="6px"))],
             style={**FRAMED, "padding": "16px 18px", "display": "flex", "flexDirection": "column", "overflowY": "auto"}),
    html.Div(style={**FRAMED, "padding": "14px 16px", "display": "grid", "gap": "14px 22px", "minHeight": 0,
                    "gridTemplateColumns": "0.95fr 0.95fr 1.05fr 1.05fr", "gridTemplateRows": "1fr 1fr",
                    "gridTemplateAreas": '"worst rings map swarm" "gender tree sankey swarm"'}, children=[
        cell("t_worst", "worst", "worst"), cell("t_rings", "rings", "rings"),
        cell("t_map", "map", "map", cfg={**GCFG, "scrollZoom": True}),
        cell("t_gender", "gender", "gender"), cell("t_tree", "tree", "tree"), cell("t_sankey", "sankey", "sankey"),
        cell("t_swarm", "swarm", "swarm"),
    ]),
])


@app.callback(Output("prov", "value"), Output("dist", "options"), Output("dist", "value"),
              Input("prov", "value"), Input("dist", "value"), Input("worst", "clickData"), Input("map", "clickData"),
              Input("tree", "clickData"), Input("swarm", "clickData"), prevent_initial_call=True)
def sync_selection(prov, dist, c_worst, c_map, c_tree, c_swarm):
    trig = ctx.triggered_id
    clicked = {"worst": c_worst, "map": c_map, "tree": c_tree, "swarm": c_swarm}.get(trig)
    if clicked:
        k = clicked["points"][0].get("customdata")
        if k in DPROV:
            prov, dist = DPROV[k], k
    elif trig == "prov":
        dist = "__avg__"
    if prov == "Pakistan":
        dist = "__avg__"
    return prov, dist_options(prov), dist


@app.callback(
    [Output(f"k{n}_{x}", "children") for n in (1, 2, 3) for x in ("lab", "val")] +
    [Output(f"k{n}_sp", "figure") for n in (1, 2, 3)] +
    [Output("hu_title", "children"), Output("headsup", "children")] +
    [Output(f"t_{g}", "children") for g in ("worst", "rings", "map", "gender", "tree", "sankey", "swarm")] +
    [Output(g, "figure") for g in ("worst", "rings", "map", "gender", "tree", "sankey", "swarm")],
    Input("prov", "value"), Input("dist", "value"), Input("sex", "value"), Input("ind", "value"))
def render(prov, dist, sex, ind):
    mode, key, p = selection(prov, dist)
    area = area_name(mode, key, p)
    short = PROV_SHORT.get(p, "Pakistan")
    lvl, gk = geo_of(mode, key, p)
    pv = status_by_age(mode, key, p, sex)
    school = (pv["School"] + pv["School + work"]) if len(pv) else pd.Series(dtype=float)
    work = (pv["Work"] + pv["School + work"]) if len(pv) else pd.Series(dtype=float)
    k_lab = [PLAIN[ind].split(" (")[0], "In school (age 10-24)", "Working (age 10-24)"]
    k_val = [fmt(val(lvl, gk, ind, sex), ind), f"{school.mean():.0f}%" if len(school) else "n/a",
             f"{work.mean():.0f}%" if len(work) else "n/a"]
    sparks = [kpi_spark(ind, sex, mode, key, p), spark(list(school.values), list(school.index)),
              spark(list(work.values), list(work.index))]
    sx = "" if sex == "All" else f" · {sex}"
    titles = [f"Worst districts in {short}{sx}" if mode != "national" else f"Worst districts in Pakistan{sx}",
              f"Still in school, by age{sx}", f"{SHORT[ind]} by district{sx}",
              "Girls vs boys", f"Where affected young people live{sx}", "What 15-24 year olds are doing",
              f"Every district, by province{sx}"]
    sank = fig_sankey(mode, key, p)
    if sank is None:
        sank = base(go.Figure()).add_annotation(text="Sample too small for this view", showarrow=False,
                                                font=dict(size=16, color=MUTED))
    figs = [fig_worst(ind, sex, mode, key, p), fig_rings(mode, key, p, sex), fig_map(ind, sex, mode, key, p),
            fig_gender(mode, key, p), fig_treemap(ind, sex, mode, key, p), sank,
            fig_swarm(ind, sex, mode, key, p)]
    gender_title = [titles[3] + "  ", html.Span("■ Girls  ", style=dict(color=GIRL, fontSize="16px")),
                    html.Span("■ Boys", style=dict(color=BOY, fontSize="16px"))]
    titles[3] = gender_title
    out = []
    for lab, v in zip(k_lab, k_val):
        out += [lab, v]
    return out + sparks + [f"Heads-up: {area}", heads_up(ind, sex, mode, key, p)] + titles + figs


server = app.server                                          # for gunicorn / Render deployment

if __name__ == "__main__":
    try:
        get_ipython()                                        # noqa: F821  (defined only inside Jupyter/IPython)
        app.run(jupyter_mode="tab", port=8050, debug=False)
    except NameError:
        app.run(host="127.0.0.1", port=8050, debug=False)
