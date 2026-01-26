import streamlit as st
import pandas as pd
import os
import subprocess
from io import BytesIO
from generate_data import generate_sample_workload
from optimizer import solve_weekly_shift_optimization
from visualize_interactive import plot_weekly_results_interactive

st.set_page_config(page_title="ShiftGen: Personnel Optimizer", layout="wide")

st.title("🚀 ShiftGen Personnel & Shuttle Optimizer")
st.markdown("""
This tool calculates the minimum personnel required to cover a weekly workload curve while aligning with shuttle service times.
""")

# --- Sidebar: Parameters ---
st.sidebar.header("Parameters & Constraints")
max_fte = st.sidebar.number_input("Maximum allowed FTE", min_value=0.0, value=80.0, help="FTE = Total Hours / 45")
max_headcount = st.sidebar.number_input("Maximum Headcount", min_value=0, value=0, help="Optional: Hard limit on unique personnel. Set to 0 for unlimited.")
c1, c2 = st.sidebar.columns(2)
min_hours_per_person = c1.number_input("Min Weekly Hours", min_value=0.0, value=35.0, step=0.5, help="Constraint: An employee should be assigned at least these hours if possible.")
max_hours_per_person = c2.number_input("Max Weekly Hours", min_value=1.0, value=48.0, step=0.5, help="Constraint: An employee cannot be assigned more than these hours.")
max_shuttles = st.sidebar.number_input("Maximum Weekly Shuttles", min_value=0, value=200, help="Warning only: Total sum of shuttle trips allowed.")
shuttle_capacity = st.sidebar.number_input("Shuttle Capacity (Pax)", min_value=1, value=16)

st.sidebar.markdown("### ⏱️ Shift Constraints")
c_min, c_max = st.sidebar.columns(2)
min_shift_len = c_min.number_input("Min Shift (Hrs)", 4.0, 12.0, 4.0, 0.5)
max_shift_len = c_max.number_input("Max Shift (Hrs)", 4.0, 12.0, 11.0, 0.5)

add_buffer = st.sidebar.checkbox("Apply 30-min Prep/Handover Buffer", value=False, help="Forces shifts to start 30 mins early or stay 30 mins late around workload peaks to allow for preparation and shift handovers.")
peak_cutting = st.sidebar.checkbox("Ignore Short-Duration Peak Spikes", value=False, help="Smoothes out very short workload spikes (less than 30 mins) to avoid hiring extra staff for momentary fluctuations.")

st.sidebar.markdown("### 📅 Off-Day Policy")
off_days_policy = st.sidebar.selectbox(
    "Minimum Days Off per Week",
    options=[1.0, 1.5, 2.0],
    index=0,
    format_func=lambda x: f"{x} day{'s' if x != 1 else ''} off",
    help="Ensures each worker gets sufficient rest days. 1.5 days = 36 consecutive hours off."
)

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

# --- Main Layout: File Operations ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Workload Management")
    
    # Check if workload exists
    current_workload_exists = os.path.exists("workload_weekly.csv")
    
    tab_import, tab_edit = st.tabs(["📂 Import / Generate", "✏️ Interactive Editor"])
    
    with tab_import:
        # 1. Random Generation
        if st.button("🎲 Generate Random Workload", help="Overwrites current workload with random data"):
            generate_sample_workload(randomized=True)
            st.success("New 'workload_weekly.csv' generated!")
            st.rerun()

        st.markdown("---")

        # 2. Upload
        uploaded_workload = st.file_uploader("Upload Workload CSV", type=["csv"], help="Columns: day_name, time, required_headcount")
        if uploaded_workload:
            df = pd.read_csv(uploaded_workload)
            df.to_csv("workload_weekly.csv", index=False)
            st.success("Workload imported!")
            st.rerun()

    with tab_edit:
        if current_workload_exists:
            try:
                df_edit = pd.read_csv("workload_weekly.csv")
                st.caption("Modify the required headcount for specific times.")
                
                edited_workload = st.data_editor(
                    df_edit, 
                    key="workload_editor", 
                    height=400, 
                    width="stretch",
                    num_rows="dynamic"
                )
                
                if st.button("💾 Save Manual Changes"):
                    edited_workload.to_csv("workload_weekly.csv", index=False)
                    st.success("Changes saved to 'workload_weekly.csv'!")
            except Exception as e:
                st.error(f"Error reading workload file: {e}")
                st.warning("The current workload file appears corrupted or has inconsistent formatting (e.g., extra commas). Please generate a new random workload or upload a valid CSV in the Import tab.")
        else:
            st.warning("No workload data found. Please Generate or Upload first.")

with col2:
    st.subheader("2. Run Calculation")
    if st.button("🏁 RUN OPTIMIZER"):
        with st.spinner("Calculating optimal shifts..."):
            result = solve_weekly_shift_optimization(
                max_fte=max_fte,
                max_headcount=max_headcount if max_headcount > 0 else None,
                min_weekly_hours=min_hours_per_person,
                max_weekly_hours=max_hours_per_person,
                min_shift_length=min_shift_len,
                max_shift_length=max_shift_len,
                min_days_off=off_days_policy,
                auto_shuttle=auto_shuttle,
                custom_shuttle_windows=shuttle_windows,
                shuttle_capacity=shuttle_capacity,
                max_shuttles=max_shuttles,
                add_handover_buffer=add_buffer,
                apply_peak_cutting=peak_cutting
            )
            
            if result is not None:
                st.success("Optimization Successful!")
                # Generate plots (interactive)
                # plot_weekly_results() # Old static plots
                # create_shuttle_report() # Old static plots
            else:
                st.error("Optimization failed. Try increasing Max FTE or relaxing constraints.")

# --- Results Display ---
if os.path.exists("weekly_summary.csv"):
    st.markdown("---")
    st.subheader("📊 Weekly Summary Metrics")
    summary = pd.read_csv("weekly_summary.csv")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total FTE", f"{summary.iloc[0]['FTE']:.2f}")
    
    # Headcount Logic
    actual_headcount = int(summary.iloc[0]['Headcount'])
    target_min = int(summary.iloc[0]['Theoretical Min Headcount']) if 'Theoretical Min Headcount' in summary.columns else 0
    target_max = int(summary.iloc[0]['Theoretical Max Headcount']) if 'Theoretical Max Headcount' in summary.columns else 0
    
    m2.metric("Headcount", actual_headcount, help=f"Theoretical Target: {target_min} to {target_max} people." if target_max > 0 else None)
    m3.metric("Total Hours", f"{summary.iloc[0]['Total Hours']:.1f}")
    m4.metric("Weekly Shuttles", f"{int(summary.iloc[0]['Total Weekly Shuttles'])}")
    
    if "People Below Min Hours" in summary.columns:
        below_min = int(summary.iloc[0]['People Below Min Hours'])
        m5.metric("Below Min Hours", below_min, delta=-below_min if below_min > 0 else 0, delta_color="inverse")
    
    # Validation Warnings
    if target_max > 0:
        st.markdown(f"**Target Headcount Analysis:** Given {summary.iloc[0]['Total Hours']:.1f} total hours and {min_hours_per_person}-{max_hours_per_person}h range, you should ideally have **{target_min} - {target_max}** workers. Current headcount is **{actual_headcount}**.")
        
        if actual_headcount > target_max:
            st.warning(f"⚠️ Headcount ({actual_headcount}) is higher than the theoretical maximum ({target_max}) for a {min_hours_per_person}h minimum. This means you have too many people working too few hours, likely due to shift overlap constraints.")
        elif actual_headcount < target_min:
            st.error(f"⚠️ Headcount ({actual_headcount}) is lower than the theoretical minimum ({target_min}). This shouldn't be possible without violating Max Hours.")
    else:
        st.info("💡 Run the optimizer to see the headcount analysis for your new min/max hour settings.")

    if summary.iloc[0]['Total Weekly Shuttles'] > max_shuttles:
        st.warning(f"⚠️ Actual shuttles ({int(summary.iloc[0]['Total Weekly Shuttles'])}) exceed the maximum limit of {max_shuttles} set in sidebar.")
        
    if max_headcount > 0 and int(summary.iloc[0]['Headcount']) > max_headcount:
        st.error(f"⚠️ Actual headcount ({int(summary.iloc[0]['Headcount'])}) exceeds the limit of {max_headcount}.")
    
    if "People Below Min Hours" in summary.columns and int(summary.iloc[0]['People Below Min Hours']) > 0:
        st.info(f"💡 {int(summary.iloc[0]['People Below Min Hours'])} workers are assigned fewer than {min_hours_per_person} hours. This usually happens when shift timings prevent people from being combined further while respecting the 12-hour rest rule.")

    # --- Excel Report Generation ---
    def generate_excel():
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # 1. Summary Sheet
            summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # 2. Roster Sheet
            if os.path.exists("personnel_roster_weekly.csv"):
                df_roster = pd.read_csv("personnel_roster_weekly.csv")
                df_roster.to_excel(writer, sheet_name='People Roster', index=False)
                
                # 2b. Worker Summary Sheet (Total hours + off days per person)
                if 'Personnel_ID' in df_roster.columns:
                    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    worker_summary = []
                    
                    for worker_id in df_roster['Personnel_ID'].unique():
                        worker_shifts = df_roster[df_roster['Personnel_ID'] == worker_id]
                        
                        # Calculate total hours (assuming we can derive from Start/End or have Duration column)
                        total_hours = 0
                        days_worked = set()
                        
                        for _, shift in worker_shifts.iterrows():
                            if 'Start' in shift and 'End' in shift:
                                start_time = pd.to_datetime(shift['Start'], format='%H:%M')
                                end_time = pd.to_datetime(shift['End'], format='%H:%M')
                                if end_time < start_time:
                                    end_time += pd.Timedelta(days=1)
                                duration = (end_time - start_time).total_seconds() / 3600
                                total_hours += duration
                            
                            if 'Day' in shift:
                                days_worked.add(shift['Day'])
                        
                        # Determine off days
                        off_days = [day for day in days_of_week if day not in days_worked]
                        
                        worker_summary.append({
                            'Personnel_ID': worker_id,
                            'Total_Hours': round(total_hours, 2),
                            'Days_Worked': len(days_worked),
                            'Off_Days': ', '.join(off_days) if off_days else 'None'
                        })
                    
                    pd.DataFrame(worker_summary).to_excel(writer, sheet_name='Worker Summary', index=False)
                
                # 3. Unique Shift Types Sheet
                if 'Start' in df_roster.columns and 'End' in df_roster.columns:
                    # Count frequency of each shift type
                    unique_shifts = df_roster.groupby(['Start', 'End']).size().reset_index(name='Total Assigned')
                    unique_shifts = unique_shifts.sort_values('Start')
                    
                    # Add formatted shift type column
                    def format_shift_type(row):
                        start_hhmm = row['Start'].replace(':', '')
                        end_hhmm = row['End'].replace(':', '')
                        return f"{start_hhmm}-{end_hhmm}"
                    
                    unique_shifts['Shift_Type'] = unique_shifts.apply(format_shift_type, axis=1)
                    # Reorder columns to put Shift_Type first
                    unique_shifts = unique_shifts[['Shift_Type', 'Start', 'End', 'Total Assigned']]
                    unique_shifts.to_excel(writer, sheet_name='Shift Types', index=False)
            
            # 4. Shuttle Report
            if os.path.exists("shuttle_report_weekly.csv"):
                df_shuttle = pd.read_csv("shuttle_report_weekly.csv")
                df_shuttle.to_excel(writer, sheet_name='Shuttles', index=False)

            # 5. Workload
            if os.path.exists("workload_weekly.csv"):
                pd.read_csv("workload_weekly.csv").to_excel(writer, sheet_name='Workload Input', index=False)
                
        return output.getvalue()

    st.download_button(
        label="📥 Download Full Excel Report (XLSX)",
        data=generate_excel(),
        file_name="ShiftGen_Weekly_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Contains Summary, Roster, Shift Types, Shuttles, and Workload data in separate sheets."
    )

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
            
            # Interactive Shuttle Charts
            _, fig_hourly, fig_daily = plot_weekly_results_interactive()
            c1, c2 = st.columns(2)
            if fig_daily:
                c1.plotly_chart(fig_daily, width="stretch")
            if fig_hourly:
                c2.plotly_chart(fig_hourly, width="stretch")

    with tab3:
        st.write("### Daily Coverage Graphs")
        daily_figs, _, _ = plot_weekly_results_interactive()
        
        if daily_figs:
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            selected_day = st.selectbox("Select Day to View", days)
            if selected_day in daily_figs:
                st.plotly_chart(daily_figs[selected_day], width="stretch")
        else:
            st.info("Run the optimizer to generate graphs.")
