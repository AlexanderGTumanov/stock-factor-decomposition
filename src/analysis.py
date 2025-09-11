from matplotlib.patches import Patch
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import statsmodels.api as sm
import io, zipfile, re
import yfinance as yf
import pandas as pd
import numpy as np
import requests

def get_returns(tickers, start, end):
    if isinstance(tickers, str):
        df = yf.download(tickers, start = start, end = end, progress = False)
        if df.empty or "Close" not in df.columns:
            return pd.Series(dtype = float)
        prices = df["Close"].squeeze()
        return prices.pct_change().dropna()
    results = {}
    for t in tickers:
        try:
            df = yf.download(t, start = start, end = end, progress = False)
            if df.empty or "Close" not in df.columns:
                continue
            prices = df["Close"].squeeze()
            rets = prices.pct_change().dropna()
            if len(rets) == 0 or np.isnan(rets).any():
                continue
            results[t] = rets
        except Exception:
            continue
    return results

def get_factors(model, start, end):
    if model in ("CAPM", "FF3"):
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_Daily_CSV.zip"
        header = "Date,Mkt-RF,SMB,HML,RF"
        cols = ["MKT_RF", "SMB", "HML", "RF"]
    elif model == "FF5":
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_Daily_CSV.zip"
        header = "Date,Mkt-RF,SMB,HML,RMW,CMA,RF"
        cols = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    else:
        raise ValueError("model must be 'CAPM', 'FF3', or 'FF5'")
    r = requests.get(url, timeout = 30)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        raw = zf.read(csv_name).decode("latin1")
    data_lines = [ln for ln in raw.splitlines() if re.match(r"^\s*\d{8},", ln)]
    csv_text = header + "\n" + "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(csv_text))
    df = df.rename(columns = {"Mkt-RF": "MKT_RF"})
    df["Date"] = pd.to_datetime(df["Date"], format = "%Y%m%d")
    df = df.set_index("Date").sort_index()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors = "coerce") / 100.0
    df = df[cols].dropna()
    if model == "CAPM":
        cols = ["MKT_RF", "RF"]
    return df.loc[start:end, cols]

def factor_regression(returns, factors, precision = 2, window = None, step = None):
    def clean(val):
        return float(round(val, precision))

    cols = [c for c in factors.columns if c != "RF"]

    def regress_one(series):
        idx = series.index.intersection(factors.index)
        y = series.loc[idx] - factors.loc[idx, "RF"]
        X = factors.loc[idx].drop(columns = ["RF"])
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        coeffs = {"alpha": clean(model.params["const"])}
        coeffs.update({c: clean(model.params[c]) for c in cols})
        tstats = {"alpha": clean(model.tvalues["const"])}
        tstats.update({c: clean(model.tvalues[c]) for c in cols})
        pstats = {"alpha": clean(model.pvalues["const"])}
        pstats.update({c: clean(model.pvalues[c]) for c in cols})
        return coeffs, tstats, pstats, clean(model.rsquared)

    def regress_rolling(series, window, step):
        idx = series.index.intersection(factors.index)
        y = series.loc[idx] - factors.loc[idx, "RF"]
        X = factors.loc[idx].drop(columns = ["RF"])
        n = len(idx)
        if step is None:
            step = max(1, int(round(n * 0.01)))
        else:
            step = int(step)
        coeffs_col = {"alpha": []}
        tstats_col = {"alpha": []}
        pstats_col = {"alpha": []}
        for c in cols:
            coeffs_col[c] = []
            tstats_col[c] = []
            pstats_col[c] = []
        centers, r2_vals = [], []
        start = 0
        while start + window <= n:
            end = start + window
            centers.append(idx[start + window // 2])
            Xw = sm.add_constant(X.iloc[start:end])
            yw = y.iloc[start:end]
            model = sm.OLS(yw, Xw).fit()
            coeffs_col["alpha"].append(clean(model.params["const"]))
            tstats_col["alpha"].append(clean(model.tvalues["const"]))
            pstats_col["alpha"].append(clean(model.pvalues["const"]))
            for c in cols:
                coeffs_col[c].append(clean(model.params[c]))
                tstats_col[c].append(clean(model.tvalues[c]))
                pstats_col[c].append(clean(model.pvalues[c]))
            r2_vals.append(clean(model.rsquared))
            start += step
        coeffs = {"alpha": pd.Series(coeffs_col["alpha"], index = centers)}
        tstats = {"alpha": pd.Series(tstats_col["alpha"], index = centers)}
        pstats = {"alpha": pd.Series(pstats_col["alpha"], index = centers)}
        for c in cols:
            coeffs[c] = pd.Series(coeffs_col[c], index = centers)
            tstats[c] = pd.Series(tstats_col[c], index = centers)
            pstats[c] = pd.Series(pstats_col[c], index = centers)
        r2 = pd.Series(r2_vals, index = centers)
        return coeffs, tstats, pstats, r2

    if isinstance(returns, dict):
        coeffs, tstats, pstats, r2 = {}, {}, {}, {}
        if window is None:
            for ticker, s in returns.items():
                c, t, p, r = regress_one(s)
                coeffs[ticker], tstats[ticker], pstats[ticker], r2[ticker] = c, t, p, r
            return coeffs, tstats, pstats, r2
        else:
            for ticker, s in returns.items():
                c, t, p, r = regress_rolling(s, window, step)
                coeffs[ticker], tstats[ticker], pstats[ticker], r2[ticker] = c, t, p, r
            return coeffs, tstats, pstats, r2
    elif isinstance(returns, pd.Series):
        if window is None:
            return regress_one(returns)
        else:
            return regress_rolling(returns, window, step)
    else:
        raise TypeError("returns must be Series or dict of Series")
    
def plot_coeffs(coeffs, r2 = None, title = None):
    vals = list(coeffs.values())
    if all(isinstance(v, pd.Series) for v in vals):
        plot_coeffs_rolling(coeffs, r2 = r2, title = title or "Factor loadings over time")
    else:
        plot_coeffs_static(coeffs, r2 = r2, title = title or "Factor decomposition")
    
def plot_coeffs_static(coeffs, r2 = None, title = "Factor decomposition"):
    color_map = {
        "alpha": "#6baed6",
        "MKT_RF": "#fd8d3c",
        "SMB":   "#74c476",
        "HML":   "#e6550d",
        "RMW":   "#9e9ac8",
        "CMA":   "#a6761d",
    }
    single_ticker = all(isinstance(v, (int, float)) for v in coeffs.values())
    if single_ticker:
        coeffs = {"": coeffs}
        if isinstance(r2, (int, float)):
            r2 = {"": r2}
    tickers = list(coeffs.keys())
    ordered = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]
    other = [f for f in ordered if any(f in d for d in coeffs.values())]
    factors = (["alpha"] if any("alpha" in d for d in coeffs.values()) else []) + other
    x = range(len(tickers))
    if isinstance(r2, dict):
        _, (ax, ax_r2) = plt.subplots(2, 1, figsize = (12, 10), sharex = True)
    else:
        _, ax = plt.subplots(1, 1, figsize = (12, 6))
        ax_r2 = None
    width = 0.3 if single_ticker else 0.8
    pos_base = [0.0] * len(tickers)
    neg_base = [0.0] * len(tickers)
    for f in factors:
        vals = [(coeffs[t].get(f, 0.0) * 252.0 if f == "alpha" else coeffs[t].get(f, 0.0))
                for t in tickers]
        pos_vals = [max(v, 0.0) for v in vals]
        neg_vals = [min(v, 0.0) for v in vals]
        col = color_map.get(f, "gray")
        ax.bar(x, pos_vals, bottom = pos_base, color = col, edgecolor = "none", width = width)
        pos_base = [pb + pv for pb, pv in zip(pos_base, pos_vals)]
        ax.bar(x, neg_vals, bottom = neg_base, color = col, edgecolor = "none", width = width)
        neg_base = [nb + nv for nb, nv in zip(neg_base, neg_vals)]
    ax.axhline(0, color = "black", linewidth = 1)
    ax.set_ylabel("Coefficient value")
    ax.set_title(title)
    ax.grid(axis = "y", alpha = 0.3)
    if single_ticker:
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)
    else:
        ax.set_xticks(list(x))
        ax.set_xticklabels(tickers, rotation = 45, ha = "right")
    legend_patches = [
        Patch(facecolor= color_map.get(f, "gray"), edgecolor = "none", label = "alpha (annualized)" if f == "alpha" else f)
        for f in factors
    ]
    if legend_patches:
        ax.legend(handles = legend_patches, loc = "best")
    if isinstance(r2, dict):
        r2_vals = [r2.get(t, 0.0) for t in tickers]
        ax_r2.bar(x, r2_vals, color = "black", edgecolor = "none", width = width)
        ax_r2.set_ylim(0, 1)
        ax_r2.set_ylabel("R²")
        if single_ticker:
            ax_r2.set_xticks([])
            ax_r2.set_xlim(-0.5, 0.5)
        else:
            ax_r2.set_xlabel("Ticker")
            ax_r2.set_xticks(list(x))
            ax_r2.set_xticklabels(tickers, rotation = 45, ha = "right")
        ax_r2.grid(axis = "y", alpha = 0.3)

    plt.tight_layout()
    plt.show()

def plot_coeffs_rolling(coeffs, r2 = None, title = "Factor loadings over time"):
    _, axes = plt.subplots(2 if r2 is not None else 1, 1, figsize = (10, 8), sharex = True)
    if r2 is None:
        ax = axes if isinstance(axes, plt.Axes) else axes[0]
    else:
        ax = axes[0]
    for name, series in coeffs.items():
        if name == "alpha":
            ax.plot(series.index, series.values * 252, label = "alpha (annualized)")
        else:
            ax.plot(series.index, series.values, label = name)
    ax.set_ylabel("Coefficient value")
    ax.set_title(title)
    ax.legend(loc = "best")
    ax.grid(True)
    if r2 is not None:
        ax2 = axes[1]
        ax2.plot(r2.index, r2.values, color = "black", linewidth = 2, label = "R²")
        ax2.fill_between(r2.index, r2.values, color = "gray", alpha = 0.3)
        ax2.set_ylabel("R²")
        ax2.set_xlabel("Date")
        ax2.grid(True)
        ax2.legend(loc = "best")
    plt.tight_layout()
    plt.show()

def plot_alpha(coeffs, tstats, title = "Alpha and t-statistic over time"):
    alpha = coeffs["alpha"] if isinstance(coeffs, dict) else coeffs
    tstat = tstats["alpha"] if isinstance(tstats, dict) else tstats
    _, ax = plt.subplots(figsize = (10, 6))
    ax.plot(alpha.index, alpha.values * 252, label = "alpha (annualized)", linewidth = 2, color = "tab:blue")
    ax.plot(tstat.index, tstat.values, label = "t-statistic", linestyle = "--", color = "tab:red")
    ax.axhline(0, color = "black", linewidth = 1)
    ax.axhline(2, color = "gray", linestyle = "--", linewidth =1)
    ax.axhline(-2, color = "gray", linestyle = "--", linewidth = 1)
    x = mdates.date2num(tstat.index.to_pydatetime())
    s = np.abs(tstat.values) - 2.0
    intervals = []
    state = s[0] > 0
    start = x[0] if state else None
    for i in range(len(s) - 1):
        if s[i] == 0:
            xc = x[i]
            if state:
                intervals.append((start, xc))
                state = False
                start = None
            else:
                state = True
                start = xc
            continue
        if s[i] * s[i + 1] < 0:
            xc = x[i] + (x[i + 1] - x[i]) * (-s[i]) / (s[i + 1] - s[i])
            if state:
                intervals.append((start, xc))
                state = False
                start = None
            else:
                state = True
                start = xc
    if s[-1] == 0:
        xc = x[-1]
        if state:
            intervals.append((start, xc))
    elif state:
        intervals.append((start, x[-1]))
    for a, b in intervals:
        ax.axvspan(a, b, color = "red", alpha = 0.3)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(loc = "best")
    ax.grid(True)
    plt.tight_layout()
    plt.show()