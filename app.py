import streamlit as st
import pandas as pd
import os
import subprocess
from generate_data import generate_sample_workload
from optimizer import solve_weekly_shift_optimization
from visualize import plot_weekly_results
from shuttle_visualizer import create_shuttle_report

st.set_page_config(page_title="ShiftGen: Personnel Optimizer", layout="wide")

st.title("🚀 ShiftGen Personnel & Shuttle Optimizer")
st.markdown("""
This tool calculates the minimum personnel required to cover a weekly workload curve while aligning with shuttle service times.
""")

# --- Sidebar: Parameters ---
st.sidebar.header("Parameters & Constraints")
max_fte = st.sidebar.number_input("Maximum allowed FTE", min_value=0.0, value=80.0, help="FTE = Total Hours / 45")
max_headcount = st.sidebar.number_input("Maximum Headcount", min_value=0, value=0, help="Optional: Hard limit on unique personnel. Set to 0 for unlimited.")
max_hours_per_person = st.sidebar.number_input("Max Weekly Hours per Person", min_value=1.0, value=48.0, step=0.5, help="Constraint: An employee cannot be assigned more than these hours.")
max_shuttles = st.sidebar.number_input("Maximum Weekly Shuttles", min_value=0, value=200, help="Warning only: Total sum of shuttle trips allowed.")
shuttle_capacity = st.sidebar.number_input("Shuttle Capacity (Pax)", min_value=1, value=16)

st.sidebar.markdown("---")
st.sidebar.subheader("🚐 Shuttle Timeline Selection")

auto_shuttle = st.sidebar.toggle("Auto-detect Shuttle Windows", value=True, help="Automatically picks shuttle times based on workload changes.")

if not auto_shuttle:
    st.sidebar.write("Manual selection (increments of 30 mins):")
    all_half_hours = [f"{h:02d}:{m:02d}" for h in range(24) for m in [0, 30]]
    selected_windows = {}
    for t in all_half_hours:
        default_val = t.endswith(":00")
        selected_windows[t] = st.sidebar.checkbox(t, value=default_val, key=f"shuttle_{t}")
    shuttle_windows = [t for t, val in selected_windows.items() if val]
else:
    st.sidebar.info("Shuttles will be aligned to workload transitions automatically.")
    shuttle_windows = None

st.sidebar.markdown("---")
if st.sidebar.button("🎲 Generate Random Workload"):
    generate_sample_workload(randomized=True)
    st.sidebar.success("New 'workload_weekly.csv' generated!")

# --- Main Layout: File Operations ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Import Data")
    uploaded_workload = st.file_uploader("Upload Workload CSV (time, required_headcount)", type=["csv"])
    if uploaded_workload:
        df = pd.read_csv(uploaded_workload)
        df.to_csv("workload_weekly.csv", index=False)
        st.success("Workload imported!")
    
    uploaded_templates = st.file_uploader("Upload Shift Templates (Optional)", type=["csv"])
    if uploaded_templates:
        df_t = pd.read_csv(uploaded_templates)
        df_t.to_csv("shift_templates.csv", index=False)
        st.success("Templates imported!")

with col2:
    st.subheader("2. Run Calculation")
    if st.button("🏁 RUN OPTIMIZER"):
        with st.spinner("Calculating optimal shifts..."):
            result = solve_weekly_shift_optimization(
                max_fte=max_fte,
                max_headcount=max_headcount if max_headcount > 0 else None,
                max_weekly_hours=max_hours_per_person,
                auto_shuttle=auto_shuttle,
                custom_shuttle_windows=shuttle_windows,
                shuttle_capacity=shuttle_capacity,
                max_shuttles=max_shuttles
            )
            
            if result is not None:
                st.success("Optimization Successful!")
                # Generate plots
                plot_weekly_results()
                create_shuttle_report()
            else:
                st.error("Optimization failed. Try increasing Max FTE or relaxing constraints.")

# --- Results Display ---
if os.path.exists("weekly_summary.csv"):
    st.markdown("---")
    st.subheader("📊 Weekly Summary Metrics")
    summary = pd.read_csv("weekly_summary.csv")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total FTE", f"{summary.iloc[0]['FTE']:.2f}")
    m2.metric("Total Headcount", f"{int(summary.iloc[0]['Headcount'])}")
    m3.metric("Total Hours", f"{summary.iloc[0]['Total Hours']:.1f}")
    m4.metric("Total Weekly Shuttles", f"{int(summary.iloc[0]['Total Weekly Shuttles'])}")

    # Validation Warning
    if summary.iloc[0]['Total Weekly Shuttles'] > max_shuttles:
        st.warning(f"⚠️ Actual shuttles ({int(summary.iloc[0]['Total Weekly Shuttles'])}) exceed the maximum limit of {max_shuttles} set in sidebar.")
        
    if max_headcount > 0 and int(summary.iloc[0]['Headcount']) > max_headcount:
        st.error(f"⚠️ Actual headcount ({int(summary.iloc[0]['Headcount'])}) exceeds the limit of {max_headcount}.")

    tab1, tab2, tab3 = st.tabs(["📅 Personnel Roster", "🚐 Shuttle Logistics", "📈 Visualization"])

    with tab1:
        st.write("### Individual Personnel Roster")
        if os.path.exists("personnel_roster_weekly.csv"):
            roster = pd.read_csv("personnel_roster_weekly.csv")
            st.dataframe(roster, height=400)
            st.download_button("📥 Download Roster", roster.to_csv(index=False), "personnel_roster_weekly.csv", "text/csv")

    with tab2:
        st.write("### Hourly Shuttle Requirements")
        if os.path.exists("shuttle_report_weekly.csv"):
            shuttle = pd.read_csv("shuttle_report_weekly.csv")
            st.dataframe(shuttle, height=400)
            st.download_button("📥 Download Shuttle Report", shuttle.to_csv(index=False), "shuttle_report_weekly.csv", "text/csv")
            
            c1, c2 = st.columns(2)
            if os.path.exists("shuttle_daily_totals.png"):
                c1.image("shuttle_daily_totals.png", caption="Daily Shuttle Totals")
            if os.path.exists("shuttle_hourly_distribution.png"):
                c2.image("shuttle_hourly_distribution.png", caption="Hourly Shuttle Load")

    with tab3:
        st.write("### Daily Coverage Graphs")
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        selected_day = st.selectbox("Select Day to View", days)
        graph_path = f"graphs/optimization_{selected_day}.png"
        if os.path.exists(graph_path):
            st.image(graph_path, width='stretch')
        else:
            st.info("Run the optimizer to generate graphs.")
