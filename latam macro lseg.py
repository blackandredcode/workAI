"""
LatAm macro monitor with the LSEG Data Library for Python (lseg-data)

For Brazil, Mexico, Chile, Colombia, Peru, Argentina it pulls:
  1. GDP (real, YoY)        - history + latest actual, prior, Reuters poll, NEXT release date
  2. CPI (headline, YoY)    - same
  3. Central bank decisions - decision history + NEXT MEETING DATE + Reuters poll

Key idea: rate decisions are economic indicators in LSEG, with real-time "=ECI"
RICs (like USFOMC=ECI or EUECBR=ECI). Their real-time record carries the next
scheduled release date, which for a rate decision is the next meeting date.
So one mechanism covers all three data sets.

Setup:
    pip install lseg-data pandas openpyxl
    Workspace running + lseg-data.config.json with your App Key
    (or use ld.open_session("platform.ldp") with RDP credentials).

First run: RICs are discovered via Search and cached to latam_rics.json.
Open that file, fix any wrong picks once, and later runs reuse it (no searching).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import lseg.data as ld

START = "2010-01-01"
RIC_CACHE = Path("latam_rics.json")
OUTPUT = Path("latam_macro.xlsx")

COUNTRIES = ["Brazil", "Mexico", "Chile", "Colombia", "Peru", "Argentina"]

# Search terms per data set, plus history interval
INDICATORS = {
    "GDP":      {"query": "{c} GDP YoY",                  "interval": "quarterly"},
    "CPI":      {"query": "{c} CPI YoY",                  "interval": "monthly"},
    "DECISION": {"query": "{c} central bank interest rate decision", "interval": None},
}

# Real-time fields on =ECI records
SNAPSHOT_FIELDS = [
    "DSPLY_NMLL",  # display name
    "FCAST_PRD",   # reference period of the latest release
    "ECON_ACT",    # latest actual
    "ECON_PRIOR",  # prior
    "RTR_POLL",    # Reuters poll consensus for the next release
    "ECI_ACT_DT",  # confirmed release date (can be stale right after a release)
    "NDOR_1",      # next date of release (forward-looking fallback)
]

SEARCH_SELECT = "RIC,RealtimeRIC,DTSubjectName,RCSCountryOfIndicatorLeaf,SourceName,Unit"


# ---------------------------------------------------------------- discovery
def _search(query: str, country: str) -> pd.DataFrame:
    """Search economic indicators, scoped to the country when the filter works."""
    base = "SearchAllCategoryv2 eq 'Economic Indicators'"
    for flt in (f"{base} and RCSCountryOfIndicatorLeaf eq '{country}'", base):
        try:
            hits = ld.discovery.search(
                view=ld.discovery.Views.INDICATOR_QUOTES,
                query=query, filter=flt, select=SEARCH_SELECT, top=10,
            )
            if hits is not None and not hits.empty:
                return hits
        except Exception as exc:  # bad filter field/value -> retry unscoped
            print(f"  [search] {country}: {exc}")
    return pd.DataFrame()


def _pick(hits: pd.DataFrame) -> dict | None:
    """Prefer a hit with a real-time =ECI RIC (that's what carries release dates)."""
    if hits.empty:
        return None
    hits = hits.copy()
    rt = hits.get("RealtimeRIC", pd.Series(index=hits.index, dtype=object)).fillna("")
    ric = hits["RIC"].fillna("")
    hits["eci"] = rt.where(rt.str.endswith("=ECI"), ric.where(ric.str.endswith("=ECI"), ""))
    best = hits[hits["eci"] != ""].head(1)
    row = (best if not best.empty else hits.head(1)).iloc[0]
    return {
        "ric": row["eci"] or row["RIC"],
        "name": row.get("DTSubjectName"),
        "source": row.get("SourceName"),
        "realtime": bool(row["eci"]),
    }


def load_or_discover_rics() -> dict:
    if RIC_CACHE.exists():
        print(f"Using cached RICs from {RIC_CACHE}")
        return json.loads(RIC_CACHE.read_text())
    print("Discovering RICs via Search (first run only)...")
    rics: dict = {}
    for c in COUNTRIES:
        rics[c] = {}
        for kind, spec in INDICATORS.items():
            pick = _pick(_search(spec["query"].format(c=c), c))
            rics[c][kind] = pick
            label = f"{pick['ric']} ({pick['name']})" if pick else "NOT FOUND"
            print(f"  {c:<10} {kind:<9} {label}")
    RIC_CACHE.write_text(json.dumps(rics, indent=2, default=str))
    print(f"Saved {RIC_CACHE}: review it and fix any wrong picks.\n")
    return rics


# ---------------------------------------------------------------- data pulls
def snapshot(rics: dict) -> pd.DataFrame:
    """One get_data call for every real-time RIC: latest, poll, next date."""
    rows = [(c, k, v["ric"]) for c, d in rics.items() for k, v in d.items()
            if v and v["ric"].endswith("=ECI")]
    if not rows:
        return pd.DataFrame()
    meta = pd.DataFrame(rows, columns=["country", "kind", "Instrument"])
    snap = ld.get_data(universe=meta["Instrument"].tolist(), fields=SNAPSHOT_FIELDS)
    df = meta.merge(snap, on="Instrument", how="left")

    # Next date: confirmed date if it's still ahead, else the forward-looking NDOR_1
    today = pd.Timestamp.today().normalize()
    confirmed = pd.to_datetime(df["ECI_ACT_DT"], errors="coerce")
    fallback = pd.to_datetime(df["NDOR_1"], errors="coerce")
    df["next_date"] = confirmed.where(confirmed >= today, fallback)
    df["next_label"] = df["kind"].map(
        {"DECISION": "next meeting"}).fillna("next release")
    return df


def history(ric: str, interval: str | None) -> pd.Series:
    """VALUE history for one indicator RIC. Failures return an empty series."""
    try:
        kw = {"universe": ric, "fields": ["VALUE"], "start": START}
        if interval:
            kw["interval"] = interval
        df = ld.get_history(**kw)
    except Exception as exc:
        print(f"  [history] {ric}: {exc}")
        return pd.Series(dtype=float, name=ric)
    if df is None or df.empty:
        return pd.Series(dtype=float, name=ric)
    col = "VALUE" if "VALUE" in df.columns else df.columns[0]
    return pd.to_numeric(df[col], errors="coerce").dropna().rename(ric)


def decision_table(country: str, s: pd.Series) -> pd.DataFrame:
    """Each observation of a decision RIC is one meeting: rate, change, action."""
    out = pd.DataFrame({"country": country, "rate": s})
    out["change_bp"] = (s.diff() * 100).round(0)
    out["action"] = out["change_bp"].map(
        lambda x: "hike" if x > 0 else "cut" if x < 0 else "hold" if x == 0 else "")
    out.index.name = "date"
    return out.reset_index()


# ---------------------------------------------------------------- main
def main() -> None:
    ld.open_session()
    try:
        rics = load_or_discover_rics()

        gdp, cpi, decisions = {}, {}, []
        for c, d in rics.items():
            for kind, pick in d.items():
                if not pick:
                    continue
                s = history(pick["ric"], INDICATORS[kind]["interval"])
                if kind == "GDP":
                    gdp[c] = s
                elif kind == "CPI":
                    cpi[c] = s
                elif not s.empty:
                    decisions.append(decision_table(c, s))

        gdp_df = pd.DataFrame({k: v for k, v in gdp.items() if not v.empty})
        cpi_df = pd.DataFrame({k: v for k, v in cpi.items() if not v.empty})
        dec_df = pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame()
        snap = snapshot(rics)

        # Upcoming calendar: next meetings + next GDP/CPI prints, sorted by date
        cal_cols = ["next_date", "country", "kind", "next_label",
                    "RTR_POLL", "ECON_ACT", "FCAST_PRD", "Instrument"]
        calendar = (snap[cal_cols].dropna(subset=["next_date"])
                    .sort_values("next_date") if not snap.empty else pd.DataFrame())

        print("\nUpcoming calendar:\n", calendar.to_string(index=False))
        if not dec_df.empty:
            print("\nLast 3 decisions per country:\n",
                  dec_df.sort_values("date").groupby("country").tail(3).to_string(index=False))

        with pd.ExcelWriter(OUTPUT) as xw:
            calendar.to_excel(xw, sheet_name="Calendar", index=False)
            snap.to_excel(xw, sheet_name="Latest", index=False)
            gdp_df.to_excel(xw, sheet_name="GDP_YoY")
            cpi_df.to_excel(xw, sheet_name="CPI_YoY")
            dec_df.to_excel(xw, sheet_name="CB_decisions", index=False)
            pd.DataFrame([{"country": c, "kind": k, **(v or {})}
                          for c, d in rics.items() for k, v in d.items()]
                         ).to_excel(xw, sheet_name="RICs", index=False)
        print(f"\nSaved {OUTPUT}")
    finally:
        ld.close_session()


if __name__ == "__main__":
    main()
