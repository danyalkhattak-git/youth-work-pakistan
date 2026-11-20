# Afghanistan Health Cluster 4W (2022 - Jul 2023) -> compact JSON for the HTML dashboard
# Usage: python who_prep.py who_data_clean.xlsx afg_adm1_simplified.geojson out.json
import sys, json
import numpy as np, pandas as pd

SRC, GEO, OUT = sys.argv[1:4]
d = pd.read_excel(SRC)
d.columns = [c.strip() for c in d.columns]
FT = [c for c in d.columns if c.startswith("Type of health")][0]
ORG, FAC = "Reporting organizations", "ID of health facility"
n_raw = len(d)

# ---- 1. data quality: duplicates, missing totals --------------------------------------
dup_mask = d.drop(columns=["sort_ID"]).duplicated()
n_dup = int(dup_mask.sum())
d = d[~dup_mask].copy()
n_missing = int(d.TOTAL.isna().sum())
d = d[d.TOTAL.notna()].copy()

# ---- 2. harmonise indicator names --------------------------------------------------
VARIANTS = {
    "# of outpatient consultations (primary health care)": "# of primary health care consultations",
    "# of Mental Health and Psychosocial Support (MHPSS) consultations": "# of mental health consultation supported",
    "# of people attended health education/awareness sessions": "# of people attended health education & awareness sessions",
    "# of outpatient referrals": "# of patients referred out from PHC facilities",
    "# of health care workers trained": "# of health staff trained",
}
var_rows = [dict(original=k, merged_into=v, rows=int((d.indicator_name == k).sum())) for k, v in VARIANTS.items()]
d["ind"] = d.indicator_name.replace(VARIANTS)
SHORT = {
    "# of primary health care consultations": "Primary health care consultations",
    "# of Ante-Natal Care (ANC) visits": "Antenatal care (ANC) visits",
    "# of Post-Natal Care (PNC) visits": "Postnatal care (PNC) visits",
    "# of deliveries attended by a skilled birth attendant": "Deliveries with a skilled birth attendant",
    "# of children under 2 years of age received Penta3 vaccine": "Children under 2 given Penta3",
    "# of children under 2 years of age received Measles vaccine": "Children under 2 given measles vaccine",
    "# of mental health consultation supported": "Mental health (MHPSS) consultations",
    "# of trauma care cases treated (conflict)": "Trauma cases treated: conflict",
    "# of trauma care cases treated (non-conflict)": "Trauma cases treated: non-conflict",
    "# of people attended health education & awareness sessions": "People in health education sessions",
    "# of Noncommunicable diseases (NCDs) consultations": "NCD consultations",
    "# of outpatient consultations (other than primary health care)": "Other outpatient consultations",
    "# of patients referred out from PHC facilities": "Referrals out of PHC facilities",
    "# of child bearing age women who received TT2 vaccine": "Women of childbearing age given TT2",
    "#of pregnant women who received TT2 vaccine": "Pregnant women given TT2",
    "# of diarrhea cases treated with oral rehydration solution (ORS) in children under the age of 5 years": "Child diarrhoea cases treated with ORS",
    "# of trauma care cases referred (non-conflict)": "Trauma referrals: non-conflict",
    "# of trauma care cases referred (conflict)": "Trauma referrals: conflict",
    "# of health staff trained": "Health staff trained",
    "# of community health workers trained": "Community health workers trained",
    "# of persons reached with Risk Communication and Community Engagement (RCCE) messages for COVID-19": "RCCE messages: COVID-19",
    "# of persons reached with Risk Communication and Community Engagement (RCCE) messages for communicable diseases": "RCCE messages: communicable diseases",
    "# of patients receiving outpatient care at secondary health care facilities": "Secondary care outpatients",
}
zero_inds = sorted(SHORT.get(i, i).replace("# of ", "") for i, t in d.groupby("ind").TOTAL.sum().items() if t == 0)
drop = ~d.ind.isin(SHORT)
n_drop_rows, n_drop_inds = int(drop.sum()), int(d.loc[drop, "ind"].nunique())
n_sar = int((d.admin1name == "Sar-e-pul").sum())
d = d[~drop].copy()
modal = d.groupby("ind").activity_type_name.agg(lambda s: s.mode().iloc[0])
n_misact = int((d.activity_type_name != d.ind.map(modal)).sum())
d["ind_s"] = d.ind.map(SHORT)

# ---- 3. outliers: > 10x the same facility's median for the indicator, and >= 500 --------
grp = d.groupby([FAC, "ind"]).TOTAL
med = grp.transform("median")
cnt = grp.transform("size")
d["ratio"] = np.where(med > 0, d.TOTAL / med, np.nan)
d["outlier"] = ((cnt >= 3) & (d.TOTAL >= 500) & (d.ratio >= 10)).astype(int)
OUTL = (d[d.outlier == 1].sort_values("TOTAL", ascending=False).head(15)
        .assign(month=lambda x: x.report_year.astype(str) + "-" + x.report_month_number.astype(str).str.zfill(2))
        [["month", ORG, "admin1name", "ind_s", "TOTAL", "ratio"]])
OUTL = [[r.month, r[ORG], r.admin1name, r.ind_s, int(r.TOTAL), round(float(r.ratio), 1)] for _, r in OUTL.iterrows()]
n_out = int(d.outlier.sum())

# ---- 4. dimensions ---------------------------------------------------------------
PNAME = {"Ghazni": "Ghanzi", "Hilmand": "Helmand", "Hirat": "Herat", "Jawzjan": "Jowzjan", "Nimroz": "Nimruz",
         "Paktya": "Paktia", "Panjsher": "Panjshir", "Sar-e-Pul": "Sar-e Pol", "Sar-e-pul": "Sar-e Pol"}
d["prov"] = d.admin1name.replace(PNAME)
d["ym"] = (d.report_year - 2022) * 12 + d.report_month_number - 1          # 0 = Jan 2022 ... 18 = Jul 2023
MONTHS = [f"{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][i % 12]} {2022 + i // 12}" for i in range(19)]
PROVS = sorted(d.prov.unique())
REG = d.groupby("prov").Region_Name.agg(lambda s: s.mode().iloc[0]).to_dict()
INDS = list(d.groupby("ind_s").TOTAL.sum().sort_values(ascending=False).index)
ORGS = list(d[ORG].value_counts().index)
FTS = list(d[FT].value_counts().index)
pi, ii, oi, fi = ({v: k for k, v in enumerate(x)} for x in (PROVS, INDS, ORGS, FTS))

# cube 1: month x province x indicator x org x outlier -> girls, boys, women, men
c1 = d.groupby(["ym", "prov", "ind_s", ORG, "outlier"])[["GIRLS", "BOYS", "WOMEN", "MEN"]].sum().reset_index()
C1 = [c1.ym.tolist(), c1.prov.map(pi).tolist(), c1.ind_s.map(ii).tolist(), c1[ORG].map(oi).tolist(), c1.outlier.tolist()] + \
     [c1[c].round().astype(int).tolist() for c in ["GIRLS", "BOYS", "WOMEN", "MEN"]]
# cube 2: month x province x indicator x facility type x outlier -> total
c2 = d.groupby(["ym", "prov", "ind_s", FT, "outlier"]).TOTAL.sum().reset_index()
C2 = [c2.ym.tolist(), c2.prov.map(pi).tolist(), c2.ind_s.map(ii).tolist(), c2[FT].map(fi).tolist(), c2.outlier.tolist(),
      c2.TOTAL.round().astype(int).tolist()]
# facilities and organisations: month bitmask per (facility, province) and (org, province)
mask = lambda s: int(sum(1 << int(m) for m in set(s)))
fm = d.groupby([FAC, "prov"]).ym.agg(mask).reset_index()
om = d.groupby([ORG, "prov"]).ym.agg(mask).reset_index()
FACM = [fm.prov.map(pi).tolist(), fm.ym.tolist()]
ORGM = [om[ORG].map(oi).tolist(), om.prov.map(pi).tolist(), om.ym.tolist()]

# ---- 5. reporting calendar and timeliness ------------------------------------------
d["created"] = pd.to_datetime(d.createdAt, errors="coerce")
me = pd.to_datetime(dict(year=d.report_year, month=d.report_month_number, day=1)) + pd.offsets.MonthEnd(0)
d["lag"] = (d.created - me).dt.days
n_nodate = int(d.created.isna().sum())
n_early = int((d.lag < 0).sum())
rep = d.groupby([ORG, "ym"]).size().unstack(fill_value=0).reindex(columns=range(19), fill_value=0)
act = {}
for o, g in d.groupby(ORG):                                           # months a project of this org was active
    months = set()
    for s, e in g[["project_start_date", "project_end_date"]].drop_duplicates().itertuples(index=False):
        if pd.isna(s) or pd.isna(e):
            continue
        for m in range(19):
            ms = pd.Timestamp(2022 + m // 12, m % 12 + 1, 1)
            if s <= ms + pd.offsets.MonthEnd(0) and e >= ms:
                months.add(m)
    act[o] = months
CAL = []
for o in ORGS[:25]:
    a = act.get(o, set())
    r = rep.loc[o].tolist()
    reported_active = sum(1 for m in a if r[m] > 0)
    lag = d.loc[d[ORG] == o, "lag"].dropna()
    CAL.append(dict(org=o, cells=[(2 if r[m] > 0 else (1 if m in a else 0)) for m in range(19)],
                    completeness=round(reported_active / len(a) * 100) if a else None,
                    lag=float(lag.median()) if len(lag) else None))
TIME = []
for m in range(19):
    s = d.loc[d.ym == m, "lag"].dropna()
    TIME.append([float(s.median()) if len(s) else None, round(float((s <= 15).mean() * 100), 1) if len(s) else None])

# ---- 6. map: project and simplify the boundaries ----------------------------------
g = json.load(open(GEO))
KX = np.cos(np.radians(34))
allx, ally = [], []
def rings(geom):
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    return [p[0] for p in polys]
for f in g["features"]:
    for r in rings(f["geometry"]):
        allx += [p[0] * KX for p in r]; ally += [p[1] for p in r]
x0, x1, y0, y1 = min(allx), max(allx), min(ally), max(ally)
SC = 1000 / (x1 - x0)
PATHS, CENT = {}, {}
for f in g["features"]:
    nm = f["properties"]["shapeName"]
    parts, big = [], None
    for r in rings(f["geometry"]):
        pts = [((p[0] * KX - x0) * SC, (y1 - p[1]) * SC) for p in r]
        parts.append("M" + "L".join(f"{a:.1f} {b:.1f}" for a, b in pts) + "Z")
        if big is None or len(pts) > len(big):
            big = pts
    PATHS[nm] = "".join(parts)
    CENT[nm] = [round(float(np.mean([p[0] for p in big])), 1), round(float(np.mean([p[1] for p in big])), 1)]

out = dict(MONTHS=MONTHS, PROVS=PROVS, REG=REG, INDS=INDS, ORGS=ORGS, FTS=FTS, C1=C1, C2=C2, FACM=FACM, ORGM=ORGM,
           CAL=CAL, TIME=TIME, OUTL=OUTL, VAR=var_rows, MAP=dict(paths=PATHS, cent=CENT, w=1000, h=round((y1 - y0) * SC)),
           QA=dict(raw=n_raw, dup=n_dup, missing=n_missing, outliers=n_out, variant_rows=sum(v["rows"] for v in var_rows),
                   orgs=len(ORGS), zero_inds=zero_inds, drop_rows=n_drop_rows, drop_inds=n_drop_inds, sar=n_sar, misact=n_misact,
                   nodate=n_nodate, early=n_early, lag_med=float(d.lag.median()), ontime=round(float((d.lag.dropna() <= 15).mean() * 100), 1), facilities=int(d[FAC].nunique()), kept=len(d)))
if "--anon" in sys.argv:                                   # replace partner names with random codes
    rng = np.random.default_rng(7); codes = rng.permutation(len(ORGS)) + 1
    code = {o: f"Partner {c:02d}" for o, c in zip(ORGS, codes)}
    out["ORGS"] = [code[o] for o in ORGS]
    for r in out["CAL"]: r["org"] = code[r["org"]]
    for r in out["OUTL"]: r[1] = code[r[1]]
json.dump(out, open(OUT, "w"), separators=(",", ":"))
print({k: v for k, v in out["QA"].items()}, "| inds", len(INDS), "| provs", len(PROVS), "| c1 rows", len(c1), "| map provs", len(PATHS))
print("missing in map:", set(PROVS) - set(PATHS))
print(OUTL[:5]); print(CAL[0]); print(TIME[:3])
