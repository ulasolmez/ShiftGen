import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

# Occupation color palette (up to 3 occupations)
OCC_COLORS = [
    {"line": "rgba(31, 119, 180, 1)",   "fill": "rgba(31, 119, 180, 0.35)", "name_color": "#1f77b4"},   # Blue
    {"line": "rgba(44, 160, 44, 1)",    "fill": "rgba(44, 160, 44, 0.35)",  "name_color": "#2ca02c"},   # Green
    {"line": "rgba(255, 127, 14, 1)",   "fill": "rgba(255, 127, 14, 0.35)", "name_color": "#ff7f0e"},   # Orange
]

def _detect_occupation_columns(df_cov):
    """Detect coverage_* and required_* columns to find occupation names."""
    occ_names = []
    for col in df_cov.columns:
        if col.startswith("coverage_"):
            occ_name = col.replace("coverage_", "")
            if f"required_{occ_name}" in df_cov.columns:
                occ_names.append(occ_name)
    return occ_names

def plot_weekly_results_interactive(coverage_csv="weekly_coverage_comparison.csv", shifts_csv="assigned_shifts_weekly.csv", shuttle_csv="shuttle_report_weekly.csv"):
    if not os.path.exists(coverage_csv):
        return None, None, None

    df_cov = pd.read_csv(coverage_csv)
    occ_names = _detect_occupation_columns(df_cov)

    # Fallback: single occupation mode (backward compatibility)
    single_mode = len(occ_names) == 0
    if single_mode:
        occ_names = ["Staff"]

    # --- 1. Daily Coverage Charts ---
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_figs = {}

    for day in days:
        day_data = df_cov[df_cov['day_name'] == day].copy()
        if day_data.empty:
            continue

        fig = go.Figure()

        if single_mode:
            # Original single-occupation view
            fig.add_trace(go.Scatter(
                x=day_data['time'],
                y=day_data['required_headcount'],
                mode='lines',
                name='Required Workload',
                line=dict(color='red', width=2, dash='solid')
            ))
            fig.add_trace(go.Scatter(
                x=day_data['time'],
                y=day_data['actual_coverage'],
                fill='tozeroy',
                mode='lines',
                name='Staff Coverage',
                line=dict(color='blue', width=2, shape='hv'),
                fillcolor='rgba(0, 0, 255, 0.2)'
            ))
        else:
            # Multi-occupation view:
            # - One single filled area for TOTAL coverage (all occupations combined)
            # - Per-occupation required lines (dashed, each in its own color)
            # - Total required line (solid red)

            # Total coverage filled area (single color)
            fig.add_trace(go.Scatter(
                x=day_data['time'],
                y=day_data['actual_coverage'],
                fill='tozeroy',
                mode='lines',
                name='Total Coverage',
                line=dict(color='rgba(65, 105, 225, 0.9)', width=2, shape='hv'),
                fillcolor='rgba(65, 105, 225, 0.2)',
            ))

            # Per-occupation required workload lines (dashed, colored)
            for i, occ_name in enumerate(occ_names):
                color = OCC_COLORS[i % len(OCC_COLORS)]
                req_col = f"required_{occ_name}"

                if req_col in day_data.columns:
                    fig.add_trace(go.Scatter(
                        x=day_data['time'],
                        y=day_data[req_col],
                        mode='lines',
                        name=f'{occ_name} Required',
                        line=dict(color=color["line"], width=2, dash='dot'),
                    ))

            # Total required workload line (bold red, solid)
            fig.add_trace(go.Scatter(
                x=day_data['time'],
                y=day_data['required_headcount'],
                mode='lines',
                name='Total Required',
                line=dict(color='red', width=2.5, dash='solid'),
            ))

        fig.update_layout(
            title=f"{day} Staffing Coverage" + (f" ({len(occ_names)} Occupations)" if not single_mode else ""),
            xaxis_title="Time",
            yaxis_title="Headcount",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )
        daily_figs[day] = fig

    # --- 2. Shuttle Hourly Distribution ---
    shuttle_fig_hourly = None
    if os.path.exists(shuttle_csv):
        df_shuttle = pd.read_csv(shuttle_csv)
        if not df_shuttle.empty:
            df_shuttle['datetime_id'] = df_shuttle['Day'] + " " + df_shuttle['Time']
            df_active = df_shuttle[df_shuttle['Total_Shuttles'] > 0].copy()
            if not df_active.empty:
                shuttle_fig_hourly = px.bar(
                    df_active,
                    x="datetime_id",
                    y=["Shuttles_In", "Shuttles_Out"],
                    title="Shuttle Activity by Window (Weekly)",
                    labels={"value": "Shuttles", "datetime_id": "Time Window"},
                    height=400
                )
                shuttle_fig_hourly.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': df_shuttle['datetime_id']})

    # --- 3. Shuttle Daily Totals ---
    shuttle_fig_daily = None
    if os.path.exists(shuttle_csv):
        df_shuttle = pd.read_csv(shuttle_csv)
        if not df_shuttle.empty:
            daily_totals = df_shuttle.groupby("Day")["Total_Shuttles"].sum().reindex(days).reset_index()
            shuttle_fig_daily = px.bar(
                daily_totals,
                x="Day",
                y="Total_Shuttles",
                title="Total Shuttles per Day",
                text="Total_Shuttles",
                color="Total_Shuttles",
                color_continuous_scale="Blues"
            )

    return daily_figs, shuttle_fig_hourly, shuttle_fig_daily
