from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    from sqlalchemy import create_engine
except Exception:  # pragma: no cover
    create_engine = None


st.set_page_config(page_title="Finance Dashboard", page_icon="📈", layout="wide")

SPREADSHEET_ID = "1Ozt_eBUvh6cuJI7njTtMAe0uAtJgV_GmOHdJM1HxxSg"
DEFAULT_TARGET_CASH = 150000.0


@dataclass
class PortfolioData:
    transactions: pd.DataFrame
    tickers: pd.DataFrame
    history: pd.DataFrame
    tax_rate: pd.DataFrame
    target_cash: float


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _parse_dates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.tz_localize(None)
    return out


def _sheet_csv_url(sheet_name: str) -> str:
    return (
        f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}"
        f"/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    )


def _read_sheet_csv(sheet_name: str) -> pd.DataFrame:
    r = requests.get(_sheet_csv_url(sheet_name), timeout=20)
    r.raise_for_status()
    return _standardize_columns(pd.read_csv(io.StringIO(r.text)))


def _load_from_db() -> PortfolioData | None:
    connect_str = st.secrets.get("DB_CONNECT_STR")
    if not connect_str or create_engine is None:
        return None

    try:
        engine = create_engine(connect_str)
        with engine.begin() as conn:
            transactions = pd.read_sql("SELECT * FROM transactions", conn)
            tickers = pd.read_sql("SELECT * FROM tickers", conn)
            history = pd.read_sql("SELECT * FROM history", conn)
            tax_rate = pd.read_sql("SELECT * FROM tax_rate", conn)

            target_cash = DEFAULT_TARGET_CASH
            try:
                cfg = pd.read_sql("SELECT * FROM goal", conn)
                if "target_cash" in cfg.columns and not cfg.empty:
                    target_cash = float(cfg["target_cash"].iloc[0])
            except Exception:
                pass

        return PortfolioData(
            transactions=_parse_dates(_standardize_columns(transactions), ["Date"]),
            tickers=_standardize_columns(tickers),
            history=_parse_dates(_standardize_columns(history), ["Date"]),
            tax_rate=_standardize_columns(tax_rate),
            target_cash=target_cash,
        )
    except Exception:
        return None


@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_portfolio_data() -> PortfolioData:
    db_data = _load_from_db()
    if db_data is not None:
        return db_data

    transactions = _parse_dates(_read_sheet_csv("Transactions"), ["Date"])
    tickers = _read_sheet_csv("Tickers")
    history = _parse_dates(_read_sheet_csv("History"), ["Date"])
    tax_rate = _read_sheet_csv("Tax Rate")

    target_cash = DEFAULT_TARGET_CASH
    try:
        goal_df = _read_sheet_csv("Goal")
        for col in goal_df.columns:
            col_key = col.lower().replace(" ", "")
            if "target" in col_key and "cash" in col_key:
                vals = pd.to_numeric(goal_df[col], errors="coerce").dropna()
                if len(vals):
                    target_cash = float(vals.iloc[0])
                    break
    except Exception:
        pass

    return PortfolioData(
        transactions=transactions,
        tickers=tickers,
        history=history,
        tax_rate=tax_rate,
        target_cash=target_cash,
    )


def _lookup_tax_account(tax_rate: pd.DataFrame, account: str) -> dict[str, Any]:
    if tax_rate.empty:
        return {}
    key_col = tax_rate.columns[0]
    row = tax_rate[tax_rate[key_col].astype(str) == str(account)]
    if row.empty:
        return {}
    r = row.iloc[0].to_dict()
    return {
        "taxable": r.get("Capital taxable componant", r.get("Capital taxable component", "None")),
        "st_rate": float(pd.to_numeric(r.get("Short-term rate", 0), errors="coerce") or 0),
        "lt_rate": float(pd.to_numeric(r.get("Long-term rate", 0), errors="coerce") or 0),
    }


def calculate_portfolio(data: PortfolioData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tx = data.transactions.copy()
    history = data.history.copy()
    tickers = data.tickers.copy()

    required = ["Date", "Type", "Account", "Stock", "Transacted Units", "Transacted Value", "Realized Tax"]
    for col in required:
        if col not in tx.columns:
            tx[col] = 0 if col != "Type" else ""

    tx = tx.dropna(subset=["Date"]).sort_values("Date")
    history = history.dropna(subset=["Date"]).sort_values("Date")

    price_cols = [c for c in history.columns if c != "Date"]
    for c in price_cols:
        history[c] = pd.to_numeric(history[c], errors="coerce").fillna(method="ffill").fillna(0)

    date_index = history["Date"].tolist()

    accounts: dict[str, dict[str, Any]] = {}

    def ensure_account(name: str) -> dict[str, Any]:
        if name not in accounts:
            tax = _lookup_tax_account(data.tax_rate, name)
            n = len(date_index)
            accounts[name] = {
                "Cash": {
                    "Balance": np.zeros(n),
                    "Contribution": np.zeros(n),
                    "Realized gain": np.zeros(n),
                    "Realized tax": np.zeros(n),
                    "Taxable": tax.get("taxable", "None"),
                    "LT tax rate": tax.get("lt_rate", 0.0),
                }
            }
        return accounts[name]

    date_to_idx = {d: i for i, d in enumerate(date_index)}

    for _, row in tx.iterrows():
        if row["Date"] not in date_to_idx:
            continue
        idx = date_to_idx[row["Date"]]
        account = ensure_account(str(row["Account"]))
        ticker = str(row["Stock"])
        txn_type = str(row["Type"])
        units = float(pd.to_numeric(row["Transacted Units"], errors="coerce") or 0)
        value = float(pd.to_numeric(row["Transacted Value"], errors="coerce") or 0)
        realized_tax = float(pd.to_numeric(row.get("Realized Tax", 0), errors="coerce") or 0)

        def bump(arr: np.ndarray, delta: float):
            arr[idx:] = np.round(arr[idx:] + delta, 2)

        if ticker in {"Cash", "Contrib"}:
            cash = account["Cash"]
            sign = -1 if txn_type == "Sell" and ticker == "Contrib" else 1
            bump(cash["Balance"], sign * value)
            if ticker == "Contrib":
                bump(cash["Contribution"], sign * value)
            if txn_type.startswith("Div"):
                bump(cash["Realized gain"], value)
            bump(cash["Realized tax"], realized_tax)
            continue

        if ticker not in account:
            n = len(date_index)
            account[ticker] = {
                "Units": np.zeros(n),
                "Cost": np.zeros(n),
                "Realized gain": np.zeros(n),
                "Realized tax": np.zeros(n),
            }

        sec = account[ticker]
        cash = account["Cash"]
        if txn_type == "Buy":
            bump(cash["Balance"], -value)
            bump(sec["Cost"], value)
            sec["Units"][idx:] = np.round(sec["Units"][idx:] + units, 3)
        elif txn_type == "Sell":
            bump(cash["Balance"], value)
            current_units = max(sec["Units"][idx], 1e-9)
            current_cost = sec["Cost"][idx]
            avg_cost = current_cost / current_units
            sell_cost = avg_cost * units
            bump(sec["Cost"], -sell_cost)
            sec["Units"][idx:] = np.round(sec["Units"][idx:] - units, 3)
            bump(sec["Realized gain"], value - sell_cost)
        elif txn_type in {"Div", "DivDrip"}:
            bump(sec["Realized gain"], value)
            if txn_type == "Div":
                bump(cash["Balance"], value)
            else:
                bump(sec["Cost"], value)
                sec["Units"][idx:] = np.round(sec["Units"][idx:] + units, 3)

        bump(sec["Realized tax"], realized_tax)

    category_map = {}
    if not tickers.empty:
        cat_col = "Category" if "Category" in tickers.columns else None
        symbol_col = tickers.columns[0]
        if cat_col:
            category_map = dict(zip(tickers[symbol_col].astype(str), tickers[cat_col].astype(str)))

    history_rows = []
    summary_rows = []

    for account, rec in accounts.items():
        cash = rec["Cash"]
        lt_rate = float(rec["Cash"].get("LT tax rate", 0.0) or 0.0)
        taxable = str(rec["Cash"].get("Taxable", "None"))
        discount = (1 - lt_rate) if taxable == "Balance" else 1.0

        for i, d in enumerate(date_index):
            total_mv = cash["Balance"][i]
            total_cost = cash["Contribution"][i]
            total_gain = cash["Realized gain"][i]
            realized_tax = cash["Realized tax"][i]
            ta_total = discount * cash["Balance"][i]
            cat_values: dict[str, float] = {}

            for ticker, tdata in rec.items():
                if ticker == "Cash":
                    continue
                price = history.loc[i, ticker] if ticker in history.columns else 1.0
                mv = float(tdata["Units"][i] * price)
                total_mv += mv
                total_cost += float(tdata["Cost"][i])
                total_gain += float(mv - tdata["Cost"][i] + tdata["Realized gain"][i])
                realized_tax += float(tdata["Realized tax"][i])

                unreal_tax = max((mv - tdata["Cost"][i]) * lt_rate, 0.0) if discount == 1 else 0.0
                ta_mv = discount * (mv - unreal_tax)
                ta_total += ta_mv

                cat = category_map.get(ticker, "Other")
                cat_values[cat] = cat_values.get(cat, 0.0) + mv

            history_rows.append(
                {
                    "Date": d,
                    "Account": account,
                    "Total": round(total_mv, 2),
                    "Cost": round(total_cost, 2),
                    "Gain": round(total_gain, 2),
                    "Realized Tax": round(realized_tax, 2),
                    "TA Total": round(ta_total, 2),
                    **{f"Category: {k}": round(v, 2) for k, v in cat_values.items()},
                }
            )

        latest = history_rows[-1]
        summary_rows.append(
            {
                "Account": account,
                "Total": latest["Total"],
                "Cost": latest["Cost"],
                "Gain": latest["Gain"],
                "TA Total": latest["TA Total"],
                "Realized Tax": latest["Realized Tax"],
            }
        )

    hist_df = pd.DataFrame(history_rows)
    if hist_df.empty:
        return hist_df, pd.DataFrame(), pd.DataFrame()

    agg = hist_df.groupby("Date", as_index=False)[["Total", "Cost", "Gain", "TA Total", "Realized Tax"]].sum()

    if category_map:
        latest = hist_df.sort_values("Date").groupby("Account", as_index=False).last()
        cat_cols = [c for c in latest.columns if c.startswith("Category: ")]
        cat_totals = latest[cat_cols].sum().rename_axis("Category").reset_index(name="Amount")
        cat_totals["Category"] = cat_totals["Category"].str.replace("Category: ", "", regex=False)
    else:
        cat_totals = pd.DataFrame(columns=["Category", "Amount"])

    return agg, pd.DataFrame(summary_rows), cat_totals


def render_history(agg: pd.DataFrame):
    st.subheader("History")
    if agg.empty:
        st.warning("No history data available.")
        return
    c1, c2, c3, c4 = st.columns(4)
    latest = agg.iloc[-1]
    c1.metric("Total", f"${latest['Total']:,.0f}")
    c2.metric("Gain", f"${latest['Gain']:,.0f}")
    c3.metric("TA Total", f"${latest['TA Total']:,.0f}")
    c4.metric("Realized Tax", f"${latest['Realized Tax']:,.0f}")
    st.line_chart(agg.set_index("Date")[["Total", "Cost", "Gain", "TA Total"]])
    st.dataframe(agg, use_container_width=True)


def render_gain(agg: pd.DataFrame):
    st.subheader("Gain")
    if agg.empty:
        st.warning("No gain data available.")
        return
    gain_df = agg[["Date", "Gain", "Realized Tax"]].copy()
    gain_df["Net Gain"] = gain_df["Gain"] - gain_df["Realized Tax"]
    st.area_chart(gain_df.set_index("Date")[["Gain", "Net Gain"]])
    st.bar_chart(gain_df.set_index("Date")[["Realized Tax"]])
    st.dataframe(gain_df, use_container_width=True)


def render_summary(summary_df: pd.DataFrame, cat_totals: pd.DataFrame):
    st.subheader("Summary")
    if summary_df.empty:
        st.warning("No summary data available.")
        return
    st.dataframe(summary_df, use_container_width=True)
    if not cat_totals.empty and cat_totals["Amount"].sum() > 0:
        st.caption("Allocation by category")
        pie = cat_totals.set_index("Category")
        st.bar_chart(pie)


def render_goal(agg: pd.DataFrame, cat_totals: pd.DataFrame, target_cash: float):
    st.subheader("Goal")
    if agg.empty:
        st.warning("No goal data available.")
        return
    latest = agg.iloc[-1]
    ta_total = float(latest["TA Total"])
    investable = max(ta_total - target_cash, 0)

    st.write("Target cash is read from **Goal** sheet if available, else a default value.")
    target_cash = st.number_input("Target cash", value=float(target_cash), step=1000.0)
    st.metric("Tax-adjusted total", f"${ta_total:,.0f}")
    st.metric("Investable amount (TA Total - Target Cash)", f"${investable:,.0f}")

    if cat_totals.empty or cat_totals["Amount"].sum() <= 0:
        st.info("No category allocation data found in Tickers sheet.")
        return

    cdf = cat_totals.copy()
    cdf["Current %"] = cdf["Amount"] / cdf["Amount"].sum() * 100

    st.write("Current allocation")
    st.dataframe(cdf[["Category", "Amount", "Current %"]], use_container_width=True)
    st.bar_chart(cdf.set_index("Category")[["Current %"]])


st.title("📈 Veochang Finance Dashboard")
st.caption("Data source priority: DB_CONNECT_STR (Postgres) ➜ Google Sheets.")

with st.spinner("Loading portfolio data..."):
    try:
        pdata = load_portfolio_data()
        agg_history, summary, category_totals = calculate_portfolio(pdata)
    except Exception as exc:
        st.error("Failed to load or calculate dashboard data.")
        st.code(json.dumps({"error": str(exc)}, indent=2))
        st.stop()

page = st.sidebar.radio("Page", ["History", "Gain", "Summary", "Goal"], index=0)

if page == "History":
    render_history(agg_history)
elif page == "Gain":
    render_gain(agg_history)
elif page == "Summary":
    render_summary(summary, category_totals)
else:
    render_goal(agg_history, category_totals, pdata.target_cash)
