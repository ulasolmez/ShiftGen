# ShiftGen: Personnel & Shuttle Optimizer

An automated coding agent for optimizing personnel shifts and shuttle logistics.

## Features
- **Integer Linear Programming**: Minimizes headcount and FTE using `PuLP`.
- **Shuttle Alignment**: Groups shifts to minimize shuttle trips.
- **Legal Compliance**: Enforces 12-hour rest periods and max FTE budgets.
- **Auto-Shuttle Generation**: Automatically clusters shuttle windows in 30-min increments.
- **Streamlit Dashboard**: Easy-to-use UI for parameter tuning and CSV data export.

## Deployment Instructions

### Streamlit Community Cloud (Recommended)
1. Push this folder to a GitHub repository.
2. Sign in to [Streamlit Cloud](https://share.streamlit.io/).
3. Click "New App" and select your repository and `app.py`.
4. Deploy!

### Local Development
1. `python -m venv .venv`
2. `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows)
3. `pip install -r requirements.txt`
4. `streamlit run app.py`

## Requirements
- Python 3.8+
- pulp
- pandas
- matplotlib
- numpy
- streamlit
- seaborn
