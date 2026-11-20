# LFS 2024-25 (and 2020-21 for change) -> province-level tables for Power BI
# Usage:
#   python lfs_province_prep.py "<LFS 2024-25.dta>" "<LFS2020-21.dta or lfs2021_slim.csv.gz>" <output_dir>
# 2024-25 web file covers ages 15-64 only, so all indicators are for ages 15+.
# Definitions (both rounds):
#   employed   = worked for pay, ran a business, helped in family business, or had a job but was absent (S5C1-S5C4)
#   enrolled   = S4C10 != 1 (1 = currently not enrolled)
#   unemployed = not employed, available (S9C6), and looked for work in last 4 weeks or is a future starter (S9C1/S9C5)
#                2020-21 uses availability codes S9C6 in {1, 2}, as in lfs2021_disability_lfp.py
#   housework  = did unpaid housework (S10C10_13a) or care (S10C10_11a) in the reference week (2024-25 only)
# CIs: linearised, PSU = EBCode; strata = province x rural/urban (2024-25), district (2020-21)
import sys, os
import numpy as np, pandas as pd

SRC24, SRC21, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
PROV = {1: "Khyber Pakhtunkhwa", 2: "Punjab", 3: "Sindh", 4: "Balochistan"}
AREA = {1: "Rural", 2: "Urban"}


def load(src, cols):
    if src.endswith(".dta"):
        import pyreadstat
        d, _ = pyreadstat.read_dta(src, usecols=cols, encoding="latin1", apply_value_formats=False)
    else:
        d = pd.read_csv(src, usecols=cols, low_memory=False)
    return d


def prepare(d, year):
    for c in d.columns:
        if c not in ("District", "EBCode"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[(d.Weights > 0) & (d.S4C6 >= 15) & (d.S4C6 <= 64)].copy()
    d["province"] = d.Province.map(PROV)
    d["area"] = d.Region.map(AREA)
    d["sex"] = np.where(d.S4C5 == 2, "Female", "Male")
    d["age"] = d.S4C6
    d["w"] = d.Weights
    d["psu"] = d.EBCode.astype(str)
    d["stratum"] = (d.District.astype(str).str.upper() if "District" in d else d.province + "_" + d.area)
    emp = (d.S5C1 == 1) | (d.S5C2 == 1) | (d.S5C3 == 1) | (d.S5C4 == 1)
    avail = d.S9C6.isin([1, 2]) if year == "2020-21" else (d.S9C6 == 1)
    unemp = (~emp) & avail & ((d.S9C1 == 1) | (d.S9C5 == 1))
    d["emp"], d["unemp"], d["lf"] = emp.astype(float), unemp.astype(float), (emp | unemp).astype(float)
    d["enrolled"] = np.where(d.S4C10.isna(), np.nan, (d.S4C10 != 1).astype(float))
    d["oos"] = 1 - d.enrolled
    d["work_oos"] = d.emp * d.oos
    d["neet"] = ((d.emp == 0) & (d.enrolled == 0)).astype(float)
    d.loc[d.enrolled.isna(), "neet"] = np.nan
    if "S10C10_13a" in d:
        d["housework"] = ((d.S10C10_13a == 1) | (d.S10C10_11a == 1)).astype(float)
        d["hw_hours"] = d.S10C10_13b.fillna(0).clip(upper=112) + d.S10C10_11b.fillna(0).clip(upper=112)
    d["year"] = year
    return d


c24 = ["EBCode", "Province", "Region", "Weights", "S4C5", "S4C6", "S4C10", "S5C1", "S5C2", "S5C3", "S5C4",
       "S9C1", "S9C5", "S9C6", "S10C10_11a", "S10C10_11b", "S10C10_13a", "S10C10_13b"]
c21 = ["EBCode", "Province", "District", "Region", "Weights", "S4C5", "S4C6", "S4C10", "S5C1", "S5C2", "S5C3",
       "S5C4", "S9C1", "S9C5", "S9C6"]
D = {"2024-25": prepare(load(SRC24, c24), "2024-25"), "2020-21": prepare(load(SRC21, c21), "2020-21")}

a = lambda d, lo, hi: (d.age >= lo) & (d.age <= hi)
IND = [  # id, plain name, short label, meaning, theme, universe, outcome, direction, sdg, in 2020-21
    ("T1", "Out of school (age 15-17)", "Out of school 15-17", "of teens aged 15-17 are not enrolled in education",
     "Education", lambda d: a(d, 15, 17), "oos", 1, "4.1", 1),
    ("T2", "Teens working (age 15-17)", "Working 15-17", "of teens aged 15-17 are working",
     "Child & teen work", lambda d: a(d, 15, 17), "emp", 1, "8.7", 1),
    ("T3", "Working, not in school (15-17)", "Working, no school", "of teens aged 15-17 work and are not in school",
     "Child & teen work", lambda d: a(d, 15, 17), "work_oos", 1, "8.7", 1),
    ("Y1", "Not in work or education (15-24)", "No work or school", "of 15-24 year olds are not in work or education",
     "Youth", lambda d: a(d, 15, 24), "neet", 1, "8.6.1", 1),
    ("Y2", "Youth unemployment (15-24)", "Youth unemployment", "of young people in the labour force can't find work",
     "Youth", lambda d: a(d, 15, 24) & (d.lf == 1), "unemp", 1, "8.5.2", 1),
    ("W1", "Women in the labour force (15-64)", "Women in labour force", "of women aged 15-64 work or seek work",
     "Women", lambda d: a(d, 15, 64) & (d.sex == "Female"), "lf", -1, "8.5", 1),
    ("H1", "Unpaid housework or care (15-24)", "Unpaid housework/care", "of 15-24 year olds did unpaid housework or care last week",
     "Women", lambda d: a(d, 15, 24), "housework", 1, "5.4.1", 0),
]


def est(s, y):
    s = s[s[y].notna()]
    n = len(s)
    if n == 0:
        return np.nan, np.nan, 0, 0.0
    W = s.w.sum()
    p = (s.w * s[y]).sum() / W
    z = s.w * (s[y] - p) / W
    zc = z.groupby([s.stratum, s.psu]).sum()
    var = 0.0
    for _, g in zc.groupby(level=0):
        k = len(g)
        if k > 1:
            var += k / (k - 1) * ((g - g.mean()) ** 2).sum()
    return p, np.sqrt(var), n, W


rows = []
for year, d in D.items():
    geos = [("Pakistan", d)] + [(p, g) for p, g in d.groupby("province")]
    for geo, g in geos:
        for area in ["All", "Rural", "Urban"]:
            ga = g if area == "All" else g[g.area == area]
            for iid, name, short, mean, theme, uni, y, direc, sdg, in21 in IND:
                if year == "2020-21" and not in21:
                    continue
                u = ga[uni(ga)]
                for sex in ["All", "Female", "Male"]:
                    if iid == "W1" and sex == "Male":
                        continue
                    s = u if sex == "All" else u[u.sex == sex]
                    p, se, n, W = est(s, y)
                    rows.append(dict(year=year, province=geo, area=area, sex=sex, indicator_id=iid,
                                     value=p * 100, ci_lo=max(0, (p - 1.96 * se) * 100), ci_hi=min(100, (p + 1.96 * se) * 100),
                                     n=n, est_count=p * W))
F = pd.DataFrame(rows)
# gender gap (girls minus boys) in not in work or education, as its own indicator
gp = F[(F.indicator_id == "Y1") & (F.sex != "All")].pivot_table(index=["year", "province", "area"], columns="sex",
                                                                    values="value").reset_index()
gp = gp.assign(sex="All", indicator_id="G1", value=gp.Female - gp.Male, ci_lo=np.nan, ci_hi=np.nan, n=np.nan, est_count=np.nan)
F = pd.concat([F, gp[F.columns]], ignore_index=True)
F["reliability"] = np.where(F.n.isna(), "n/a", np.where(F.n < 50, "Suppressed", np.where((F.ci_hi - F.ci_lo) / 2 > 10, "Use with caution", "Reliable")))
F.loc[F.reliability == "Suppressed", ["value", "ci_lo", "ci_hi", "est_count"]] = np.nan
F["year_sort"] = F.year.map({"2020-21": 1, "2024-25": 2})
F = F.round({"value": 2, "ci_lo": 2, "ci_hi": 2, "est_count": 0})
F.to_csv(f"{OUT}/fact_province.csv", index=False)

DI = pd.DataFrame([dict(indicator_id=i, name=n, short=sh, meaning=m, theme=t, need_direction=dr, sdg=sd,
                        in_2020_21=x, unit="%", sort=k + 1) for k, (i, n, sh, m, t, _, _, dr, sd, x) in enumerate(IND)] +
                  [dict(indicator_id="G1", name="Gender gap: not in work or education (girls minus boys)",
                        short="Gender gap, no work/school", meaning="points more young women than young men are not in work or education",
                        theme="Women", need_direction=1, sdg="5", in_2020_21=1, unit="pts", sort=len(IND) + 1)])
DI.to_csv(f"{OUT}/dim_indicator.csv", index=False)

pd.DataFrame(dict(province=["Pakistan", "Khyber Pakhtunkhwa", "Punjab", "Sindh", "Balochistan"],
                  short=["Pakistan", "KP", "Punjab", "Sindh", "Balochistan"], sort=[0, 1, 2, 3, 4],
                  is_province=[0, 1, 1, 1, 1])).to_csv(f"{OUT}/dim_province.csv", index=False)

# what young people are doing, by age band (2024-25)
d = D["2024-25"]
d = d[a(d, 15, 24) & d.enrolled.notna()].copy()
d["status"] = np.select([(d.enrolled == 1) & (d.emp == 0), (d.enrolled == 1) & (d.emp == 1), (d.enrolled == 0) & (d.emp == 1)],
                        ["In education", "Education + work", "Working"], "Neither")
d["age_band"] = pd.cut(d.age, [14, 17, 20, 24], labels=["15-17", "18-20", "21-24"]).astype(str)
srows = []
for geo, g in [("Pakistan", d)] + [(p, x) for p, x in d.groupby("province")]:
    for area in ["All", "Rural", "Urban"]:
        ga = g if area == "All" else g[g.area == area]
        for sex in ["All", "Female", "Male"]:
            s0 = ga if sex == "All" else ga[ga.sex == sex]
            for band, s in s0.groupby("age_band"):
                W = s.w.sum()
                for k, st in enumerate(["In education", "Education + work", "Working", "Neither"]):
                    srows.append(dict(province=geo, area=area, sex=sex, age_band=band, status=st, status_order=k + 1,
                                      share=round((s.w * (s.status == st)).sum() / W * 100, 2), n=len(s)))
pd.DataFrame(srows).to_csv(f"{OUT}/age_status.csv", index=False)

# weekly hours of unpaid housework + care, 15-24, by sex (2024-25)
hrows = []
d = D["2024-25"]
d = d[a(d, 15, 24)]
for geo, g in [("Pakistan", d)] + [(p, x) for p, x in d.groupby("province")]:
    for area in ["All", "Rural", "Urban"]:
        ga = g if area == "All" else g[g.area == area]
        for sex in ["Female", "Male"]:
            s = ga[ga.sex == sex]
            hrows.append(dict(province=geo, area=area, sex=sex, hours=round((s.w * s.hw_hours).sum() / s.w.sum(), 1), n=len(s)))
pd.DataFrame(hrows).to_csv(f"{OUT}/housework_hours.csv", index=False)

chk = F[(F.area == "All") & (F.sex == "All")].pivot_table(index="indicator_id", columns=["year", "province"], values="value").round(1)
print(chk.to_string())
print(pd.DataFrame(hrows).query("area=='All'").pivot(index="province", columns="sex", values="hours"))
