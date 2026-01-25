import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

def plot_weekly_results_interactive(coverage_csv="weekly_coverage_comparison.csv", shifts_csv="assigned_shifts_weekly.csv", shuttle_csv="shuttle_report_weekly.csv"):
    if not os.path.exists(coverage_csv):
        return None, None, None

    df_cov = pd.read_csv(coverage_csv)
    
    # --- 1. Daily Coverage Charts ---
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_figs = {}

    for day in days:
        day_data = df_cov[df_cov['day_name'] == day].copy()
        if day_data.empty:
            continue
            
        fig = go.Figure()
        
        # Workload Area (Filled)
        fig.add_trace(go.Scatter(
            x=day_data['time'], 
            y=day_data['required_headcount'],
            fill='tozeroy',
            mode='none',
            name='Required Workload',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))
        
        # Coverage Line (Step)
        fig.add_trace(go.Scatter(
            x=day_data['time'], 
            y=day_data['actual_coverage'],
            mode='lines',
            name='Staff Coverage',
            line=dict(color='blue', width=2, shape='hv')
        ))
        
        fig.update_layout(
            title=f"{day} Staffing Coverage",
            xaxis_title="Time",
            yaxis_title="Headcount",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        daily_figs[day] = fig

    # --- 2. Shuttle Hourly Distribution ---
    shuttle_fig_hourly = None
    if os.path.exists(shuttle_csv):
        df_shuttle = pd.read_csv(shuttle_csv)
        if not df_shuttle.empty:
            # Group by Time across all days to show average/total demand structure
            # Or show full weekly timeline
            
            # Let's show full weekly timeline as a bar chart
            # Create a datetime column for continuous x-axis
            # We'll just fake it with Day + Time string for unique categorical axis
            df_shuttle['datetime_id'] = df_shuttle['Day'] + " " + df_shuttle['Time']
            
            # Filter non-zero entries to make chart readable
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
                shuttle_fig_hourly.update_layout(xaxis={'categoryorder':'array', 'categoryarray': df_shuttle['datetime_id']})

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
