# Veochang Finance Dashboard

Streamlit dashboard for portfolio tracking with pages:

- **History**: total, cost, gain, tax-adjusted trend over time
- **Gain**: gain and realized tax views
- **Summary**: latest account-level snapshot + category allocations
- **Goal**: target-cash planning and allocation gap view

## Data sources

The app loads data in this priority order:

1. **Postgres via `st.secrets["DB_CONNECT_STR"]`** (tables: `transactions`, `tickers`, `history`, `tax_rate`, optional `goal`)
2. **Google Sheets CSV export** for spreadsheet ID `1Ozt_eBUvh6cuJI7njTtMAe0uAtJgV_GmOHdJM1HxxSg`
   - Sheets expected: `Transactions`, `Tickers`, `History`, `Tax Rate`, optional `Goal`

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Cloud secrets

Set these in Streamlit Cloud (already provisioned in your environment):

- `DB`
- `DB_HOST`
- `DB_POLLER`
- `DB_ROLE`
- `DB_PASSWORD`
- `DB_CONNECT_STR`

Only `DB_CONNECT_STR` is required by the app directly.
