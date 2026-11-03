import sys, re
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
import sta348_hmm_v1 as m
F = m.FACTORS
mz, momz, wk, momd = sys.argv[2:6]
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)

def raw_info(path):
    txt = m._read_text(path); lines = txt.splitlines()
    data = [l for l in lines if re.match(r"^\s*\d{6,8}\s*,", l)]
    ann = [l for l in lines if re.match(r"^\s*\d{4}\s*,", l)]
    miss = sum(1 for l in data if "-99.99" in l or "-999" in l)
    hdr = [l for l in lines if l.startswith(",")]
    vint = re.search(r"(\d{6}) CRSP", txt)
    return len(lines), len(data), len(ann), miss, hdr, vint.group(1) if vint else None
print("== RAW FILES")
for p in (mz, momz, wk, momd):
    nl, nd, na, miss, hdr, v = raw_info(p)
    df = m.read_french_daily(p)
    print(p.split('/')[-1], "| lines", nl, "| data rows", nd, "| annual rows", na, "| missing codes in data", miss,
          "| header", hdr, "| vintage", v, "| first", df.index[0].date(), "last", df.index[-1].date(), "| parsed", len(df), "| NaN", int(df.isna().sum().sum()))

ffm = m.read_french_daily(mz); mm = m.read_french_daily(momz)
ffw = m.read_french_daily(wk); md = m.read_french_daily(momd)
print("\n== MONTHLY JOIN: FF3 months", len(ffm), ffm.index[0].date(), "-", ffm.index[-1].date(),
      "| MOM months", len(mm), mm.index[0].date(), "-", mm.index[-1].date())
mon = m.load_french(mz, momz)
print("joined", len(mon), mon.index[0].date(), mon.index[-1].date(), "| FF3 months dropped (no MOM):", len(ffm) - len(mon))
gaps = pd.Series(mon.index).diff().dt.days.dropna()
print("month gaps (days) unique:", sorted(gaps.unique()), "| duplicates", mon.index.duplicated().sum())
exp = pd.date_range(mon.index[0], mon.index[-1], freq="ME"); print("missing months vs calendar:", len(exp.difference(mon.index)))

print("\n== WEEKLY JOIN: FF3 weeks", len(ffw), ffw.index[0].date(), "-", ffw.index[-1].date(),
      "| daily MOM days", len(md), md.index[0].date(), "-", md.index[-1].date())
wkly = m.load_french(wk, momd)
weeks = ffw.index
mdd = md["Mom"].dropna()
mdd = mdd[(mdd.index > weeks[0] - pd.Timedelta(days=7)) & (mdd.index <= weeks[-1])]
k = weeks[np.searchsorted(weeks.values, mdd.index.values, side="left")]
dpw = mdd.groupby(k).size()
print("weekly joined", len(wkly), wkly.index[0].date(), wkly.index[-1].date(), "| FF3 weeks without MOM:", len(ffw) - len(wkly))
print("first FF3 week with MOM days:", dpw.index[0].date(), "days", dpw.iloc[0], "| kept first week", wkly.index[0].date())
print("trading days per week:", dpw.loc[wkly.index].value_counts().sort_index().to_dict())
print("daily MOM days after last weekly date (dropped):", int((md.index > weeks[-1]).sum()), md.index[-1].date())
print("weekly date weekday counts:", pd.Series(wkly.index.dayofweek).map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat'}).value_counts().to_dict())
print("week gaps days:", pd.Series(wkly.index).diff().dt.days.dropna().value_counts().sort_index().to_dict())
low = dpw.loc[wkly.index]; print("weeks with <=3 trading days:", [(d.date(), int(n)) for d, n in low[low <= 3].items()])

# sanity: daily MOM compounded to months vs monthly MOM
mdm = (1 + md["Mom"]).resample("ME").prod() - 1
j = pd.concat([mdm.rename("daily_comp"), mm["Mom"].rename("monthly")], axis=1, join="inner").dropna()
print("\nMOM check: daily-compounded vs monthly file: months", len(j), "corr %.3f" % j.corr().iloc[0, 1],
      "mean abs diff %.4f" % (j.daily_comp - j.monthly).abs().mean())

def stats(df, P):
    out = {}
    for f in F:
        x = df[f]
        out[f] = {"n": len(x), "mean_%": x.mean()*100, "sd_%": x.std()*100, "min_%": x.min()*100, "p25_%": x.quantile(.25)*100,
                  "median_%": x.median()*100, "p75_%": x.quantile(.75)*100, "max_%": x.max()*100, "skew": x.skew(), "ex_kurt": x.kurt(),
                  "ac1": x.autocorr(1), "pos_share": (x > 0).mean(), "ann_mean_%": x.mean()*P*100, "ann_sd_%": x.std()*np.sqrt(P)*100,
                  "t_mean": x.mean()/(x.std()/np.sqrt(len(x)))}
    return pd.DataFrame(out).T

for lab, df, P in (("MONTHLY", mon, 12), ("WEEKLY", wkly, 52)):
    d, o = m.make_observations(df, False, 36); T = len(o); s = int(round(T*0.5))
    print(f"\n==== {lab}: T={T}, train {s} ({d[0].date()} - {d[s-1].date()}), test {T-s} ({d[s].date()} - {d[-1].date()})")
    for win, sl in (("full", slice(None)), ("train", slice(0, s)), ("test", slice(s, None))):
        sub = df.loc[d[sl]]
        print(f"-- {win}"); print(stats(sub, P).round(3).to_string())
        print("corr:\n", sub.corr().round(3).to_string())
        oo = o[sl]; c = np.bincount(oo, minlength=3)
        print("top counts", dict(zip(F, c.tolist())), "shares", dict(zip(F, (c/len(oo)).round(3).tolist())))
        print("avg run length", {k_: round(v, 2) for k_, v in m.run_lengths(oo).items()}, "| switch rate %.3f" % np.mean(oo[1:] != oo[:-1]))
    X = df.loc[d, F].to_numpy(); srt = np.sort(X, axis=1)
    print("exact ties for top:", int((srt[:, 2] == srt[:, 1]).sum()), "| median margin top-second (pp): %.3f" % (np.median(srt[:, 2]-srt[:, 1])*100),
          "| share margin < 0.1pp: %.3f" % np.mean((srt[:, 2]-srt[:, 1]) < 0.001))
    C = np.zeros((3, 3), int); np.add.at(C, (o[:-1], o[1:]), 1); print("transition counts full:\n", C)
    for vs in (True,):
        dv, ov = m.make_observations(df, True, 36 if lab == "MONTHLY" else 52)
        Tv = len(ov); sv = int(round(Tv*0.5))
        print(f"vol-scaled: T={Tv} (first {dv[0].date()}), train {sv}, test {Tv-sv}; top shares train", dict(zip(F, np.round(np.bincount(ov[:sv], minlength=3)/sv, 3).tolist())))
    # decade top shares
    dd = pd.DataFrame({"dec": (d.year//10)*10, "top": [F[i] for i in o]})
    print("top share by decade:\n", pd.crosstab(dd.dec, dd.top, normalize="index").round(2).assign(n=dd.groupby("dec").size()).to_string())
