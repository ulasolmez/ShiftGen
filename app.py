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
This tool calculates the minimum personnel required to cover weekly workload curves while aligning with shared shuttle service times.  
Upload up to **3 different occupations** — each with its own workload — and the optimizer will generate shifts for all of them jointly.
""")

# --- Sidebar: Parameters ---
st.sidebar.header("Parameters & Constraints")
max_fte = st.sidebar.number_input("Maximum allowed FTE (combined)", min_value=0.0, value=200.0, help="FTE = Total Hours / 45. This is the combined limit across all occupations.")
max_headcount = st.sidebar.number_input("Maximum Headcount", min_value=0, value=0, help="Optional: Hard limit on unique personnel (combined). Set to 0 for unlimited.")
c1, c2 = st.sidebar.columns(2)
min_hours_per_person = c1.number_input("Min Weekly Hours", min_value=0.0, value=35.0, step=0.5, help="Constraint: An employee should be assigned at least these hours if possible.")
max_hours_per_person = c2.number_input("Max Weekly Hours", min_value=1.0, value=48.0, step=0.5, help="Constraint: An employee cannot be assigned more than these hours.")
max_shuttles = st.sidebar.number_input("Maximum Weekly Shuttles", min_value=0, value=200, help="Warning only: Total sum of shuttle trips allowed.")
shuttle_capacity = st.sidebar.number_input("Shuttle Capacity (Pax)", min_value=1, value=16)

st.sidebar.markdown("### ⏱️ Shift Constraints")
c_min, c_max = st.sidebar.columns(2)
min_shift_len = c_min.number_input("Min Shift (Hrs)", 4.0, 12.0, 4.0, 0.5)
max_shift_len = c_max.number_input("Max Shift (Hrs)", 4.0, 12.0, 11.0, 0.5)

add_buffer = st.sidebar.checkbox("Apply 30-min Prep/Handover Buffer", value=False, help="Forces shifts to start 30 mins early or stay 30 mins late around workload peaks.")
peak_cutting = st.sidebar.checkbox("Ignore Short-Duration Peak Spikes", value=False, help="Smoothes out very short workload spikes (less than 30 mins).")

st.sidebar.markdown("### 📅 Off-Day Policy")
off_days_policy = st.sidebar.selectbox(
    "Maximum Days Off per Week",
    options=[1.0, 1.5, 2.0],
    index=0,
    format_func=lambda x: f"{x} day{'s' if x != 1 else ''} off (max)",
    help="Sets the maximum rest days allowed per worker per week."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🚐 Shuttle Timeline Selection")

auto_shuttle = st.sidebar.toggle("Auto-detect Shuttle Windows", value=True, help="Provides half-hourly shuttle windows during work hours for tight demand matching. Shuttle consolidation penalty keeps actual shuttle events reasonable.")

if not auto_shuttle:
    st.sidebar.write("Manual selection (increments of 30 mins):")
    all_half_hours = [f"{h:02d}:{m:02d}" for h in range(24) for m in [0, 30]]
    selected_windows = {}
    for t in all_half_hours:
        default_val = t.endswith(":00")
        selected_windows[t] = st.sidebar.checkbox(t, value=default_val, key=f"shuttle_{t}")
    shuttle_windows = [t for t, val in selected_windows.items() if val]
else:
    st.sidebar.info("Shuttle windows will be auto-detected from workload edges (≥2h apart).")
    shuttle_windows = None

# ============ MAIN LAYOUT: MULTI-OCCUPATION WORKLOAD ============
st.subheader("1. Workload Management (Up to 3 Occupations)")
st.caption("Each occupation has its own workload curve and personnel pool. They share the same shuttle service.")

OCC_COLORS = {"occ1": "🔵", "occ2": "🟢", "occ3": "🟠"}

# Initialize session state for occupation data
if "occ_workloads" not in st.session_state:
    st.session_state.occ_workloads = {}

occ_tabs = st.tabs(["👷 Occupation 1", "👷 Occupation 2", "👷 Occupation 3"])

active_workloads = []

for i, occ_tab in enumerate(occ_tabs):
    occ_key = f"occ{i+1}"
    occ_file = f"workload_{occ_key}.csv"

    with occ_tab:
        col_name, col_actions = st.columns([1, 2])

        with col_name:
            occ_name = st.text_input(
                "Occupation Name",
                value=f"Occupation {i+1}" if i > 0 else "Staff",
                key=f"name_{occ_key}",
                help="Give this occupation a descriptive name (e.g., 'Operators', 'Technicians', 'Logistics')"
            )

        with col_actions:
            sub_tab_import, sub_tab_edit = st.tabs(["📂 Import / Generate", "✏️ Editor"])

            with sub_tab_import:
                col_gen, col_up = st.columns(2)
                with col_gen:
                    if st.button(f"🎲 Generate Random", key=f"gen_{occ_key}", help="Generate random workload for this occupation"):
                        generate_sample_workload(filename=occ_file, randomized=True)
                        st.success(f"Random workload generated for {occ_name}!")
                        st.rerun()

                with col_up:
                    uploaded = st.file_uploader(
                        f"Upload CSV",
                        type=["csv"],
                        key=f"upload_{occ_key}",
                        help="Columns: day_name, time, required_headcount"
                    )
                    if uploaded:
                        df_up = pd.read_csv(uploaded)
                        df_up.to_csv(occ_file, index=False)
                        st.success(f"Workload imported for {occ_name}!")
                        st.rerun()

                if os.path.exists(occ_file):
                    if st.button(f"🗑️ Remove this workload", key=f"remove_{occ_key}"):
                        os.remove(occ_file)
                        st.rerun()

            with sub_tab_edit:
                if os.path.exists(occ_file):
                    try:
                        df_edit = pd.read_csv(occ_file)
                        edited_wl = st.data_editor(
                            df_edit,
                            key=f"editor_{occ_key}",
                            height=300,
                            use_container_width=True,
                            num_rows="dynamic"
                        )
                        if st.button(f"💾 Save Changes", key=f"save_{occ_key}"):
                            edited_wl.to_csv(occ_file, index=False)
                            st.success("Changes saved!")
                    except Exception as e:
                        st.error(f"Error reading workload: {e}")
                else:
                    st.info("No workload loaded. Use Import / Generate tab first.")

        # Track active workloads
        if os.path.exists(occ_file):
            active_workloads.append({"csv_path": occ_file, "name": occ_name})
            st.success(f"✅ {occ_name} workload loaded ({occ_file})")

# Also support legacy single workload file
if not active_workloads and os.path.exists("workload_weekly.csv"):
    active_workloads.append({"csv_path": "workload_weekly.csv", "name": "Staff"})
    st.info("Using legacy workload_weekly.csv as default.")

# ============ RUN OPTIMIZER ============
st.markdown("---")
col_run, col_info = st.columns([1, 2])

with col_run:
    st.subheader("2. Run Calculation")
    num_active = len(active_workloads)
    if num_active > 0:
        st.write(f"**{num_active} occupation(s)** ready: {', '.join(w['name'] for w in active_workloads)}")

        if st.button("🏁 RUN OPTIMIZER", type="primary", use_container_width=True):
            with st.spinner(f"Optimizing shifts for {num_active} occupation(s)..."):
                result = solve_weekly_shift_optimization(
                    workloads=active_workloads,
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
                else:
                    st.error("Optimization failed. Try increasing Max FTE or relaxing constraints.")
    else:
        st.warning("Upload or generate at least one occupation workload to run the optimizer.")

with col_info:
    st.subheader(" ")
    st.info("""
    **How multi-occupation works:**
    - Each occupation has its own workload curve and personnel pool
    - Workers are assigned to only one occupation (different jobs)
    - All occupations share the same shuttle service (entry/exit times)
    - The solver minimizes total headcount across all occupations jointly
    """)

# ============ RESULTS DISPLAY ============
if os.path.exists("weekly_summary.csv"):
    st.markdown("---")
    st.subheader("📊 Results Summary")
    summary = pd.read_csv("weekly_summary.csv")

    # Combined metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total FTE", f"{summary.iloc[0]['FTE']:.2f}")

    actual_headcount = int(summary.iloc[0]['Headcount'])
    target_min = int(summary.iloc[0].get('Theoretical Min Headcount', 0))
    target_max = int(summary.iloc[0].get('Theoretical Max Headcount', 0))

    m2.metric("Total Headcount", actual_headcount, help=f"Theoretical range: {target_min}–{target_max}" if target_max > 0 else None)
    m3.metric("Total Hours", f"{summary.iloc[0]['Total Hours']:.1f}")
    m4.metric("Weekly Shuttles", f"{int(summary.iloc[0]['Total Weekly Shuttles'])}")

    below_min = int(summary.iloc[0].get('People Below Min Hours', 0))
    m5.metric("Below Min Hours", below_min, delta=-below_min if below_min > 0 else 0, delta_color="inverse")

    # Per-occupation breakdown
    num_occ = int(summary.iloc[0].get('Num Occupations', 1))
    if num_occ > 1 and active_workloads:
        st.markdown("#### Per-Occupation Breakdown")
        display_workloads = active_workloads[:num_occ]
        occ_metrics_cols = st.columns(len(display_workloads))
        for idx, wl in enumerate(display_workloads):
            occ_name = wl["name"]
            with occ_metrics_cols[idx]:
                occ_hc = summary.iloc[0].get(f"{occ_name} - Headcount", "–")
                occ_fte = summary.iloc[0].get(f"{occ_name} - FTE", "–")
                occ_hrs = summary.iloc[0].get(f"{occ_name} - Total Hours", "–")
                occ_below = summary.iloc[0].get(f"{occ_name} - People Below Min Hours", "–")
                color_emoji = ["🔵", "🟢", "🟠"][idx % 3]
                st.markdown(f"**{color_emoji} {occ_name}**")
                try:
                    hc_str = str(int(occ_hc)) if occ_hc != "–" else "–"
                    fte_str = f"{float(occ_fte):.2f}" if occ_fte != "–" else "–"
                    hrs_str = f"{float(occ_hrs):.1f}" if occ_hrs != "–" else "–"
                    st.write(f"Headcount: **{hc_str}** | FTE: **{fte_str}** | Hours: **{hrs_str}**")
                except (ValueError, TypeError):
                    st.write(f"Headcount: **{occ_hc}** | FTE: **{occ_fte}** | Hours: **{occ_hrs}**")
                if occ_below != "–":
                    try:
                        if int(occ_below) > 0:
                            st.caption(f"⚠️ {int(occ_below)} below min hours")
                    except (ValueError, TypeError):
                        pass

    # Validation warnings
    if target_max > 0:
        st.markdown(f"**Target Headcount Analysis:** Given {summary.iloc[0]['Total Hours']:.1f} total hours and {min_hours_per_person}–{max_hours_per_person}h range, target is **{target_min}–{target_max}** workers. Actual: **{actual_headcount}**.")
        if actual_headcount > target_max:
            st.warning(f"⚠️ Headcount ({actual_headcount}) exceeds theoretical maximum ({target_max}). This is common due to rest and off-day constraints.")

    if summary.iloc[0]['Total Weekly Shuttles'] > max_shuttles:
        st.warning(f"⚠️ Shuttles ({int(summary.iloc[0]['Total Weekly Shuttles'])}) exceed the limit of {max_shuttles}.")

    if below_min > 0:
        st.info(f"💡 {below_min} workers are below {min_hours_per_person}h. This usually happens due to Sunday/Monday rest conflicts.")

    # ============ EXCEL REPORT ============
    def generate_excel():
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            summary.to_excel(writer, sheet_name='Summary', index=False)

            if os.path.exists("personnel_roster_weekly.csv"):
                df_roster = pd.read_csv("personnel_roster_weekly.csv")
                df_roster.to_excel(writer, sheet_name='People Roster', index=False)

                if 'Personnel_ID' in df_roster.columns:
                    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    worker_summary = []

                    for worker_id in df_roster['Personnel_ID'].unique():
                        worker_shifts = df_roster[df_roster['Personnel_ID'] == worker_id]
                        total_hours = 0
                        days_worked = set()
                        occupation = worker_shifts.iloc[0].get('Occupation', 'Staff') if 'Occupation' in worker_shifts.columns else 'Staff'

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

                        off_days = [day for day in days_of_week if day not in days_worked]

                        worker_summary.append({
                            'Personnel_ID': worker_id,
                            'Occupation': occupation,
                            'Total_Hours': round(total_hours, 2),
                            'Days_Worked': len(days_worked),
                            'Off_Days': ', '.join(off_days) if off_days else 'None'
                        })

                    pd.DataFrame(worker_summary).to_excel(writer, sheet_name='Worker Summary', index=False)

                if 'Start' in df_roster.columns and 'End' in df_roster.columns:
                    group_cols = ['Start', 'End']
                    if 'Occupation' in df_roster.columns:
                        group_cols = ['Occupation'] + group_cols

                    # Build per-shift-type day usage: which days each shift is used on
                    day_names_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
                    def get_days_used(sub_df):
                        day_vals = sub_df['Day'].unique()
                        day_labels = sorted(day_vals)
                        return ', '.join(day_names_map.get(d, str(d)) for d in day_labels)

                    unique_shifts = df_roster.groupby(group_cols).agg(
                        **{'Total Assigned': ('Start', 'size')}
                    ).reset_index()
                    days_used = df_roster.groupby(group_cols)['Day'].apply(
                        lambda x: ', '.join(day_names_map.get(d, str(d)) for d in sorted(x.unique()))
                    ).reset_index(name='Days Used')
                    unique_shifts = unique_shifts.merge(days_used, on=group_cols)
                    unique_shifts = unique_shifts.sort_values(group_cols)

                    def format_shift_type(row):
                        return f"{row['Start'].replace(':', '')}-{row['End'].replace(':', '')}"
                    unique_shifts['Shift_Type'] = unique_shifts.apply(format_shift_type, axis=1)
                    cols_order = ['Shift_Type'] + group_cols + ['Total Assigned', 'Days Used']
                    unique_shifts = unique_shifts[cols_order]
                    unique_shifts.to_excel(writer, sheet_name='Shift Types', index=False)

            if os.path.exists("shuttle_report_weekly.csv"):
                pd.read_csv("shuttle_report_weekly.csv").to_excel(writer, sheet_name='Shuttles', index=False)

            # Write each occupation workload input
            for wl in active_workloads:
                if os.path.exists(wl["csv_path"]):
                    sheet_name = f"Workload - {wl['name']}"[:31]  # Excel sheet name limit
                    pd.read_csv(wl["csv_path"]).to_excel(writer, sheet_name=sheet_name, index=False)

        return output.getvalue()

    st.download_button(
        label="📥 Download Full Excel Report (XLSX)",
        data=generate_excel(),
        file_name="ShiftGen_Weekly_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Contains Summary, Roster, Worker Summary, Shift Types, Shuttles, and Workload inputs."
    )

    # ============ RESULTS TABS ============
    tab1, tab2, tab3 = st.tabs(["📅 Personnel Roster", "🚐 Shuttle Logistics", "📈 Visualization"])

    with tab1:
        st.write("### Personnel Roster")
        if os.path.exists("personnel_roster_weekly.csv"):
            roster = pd.read_csv("personnel_roster_weekly.csv")

            # Filter by occupation if multi-occupation
            if 'Occupation' in roster.columns and roster['Occupation'].nunique() > 1:
                occ_filter = st.multiselect(
                    "Filter by Occupation",
                    options=roster['Occupation'].unique().tolist(),
                    default=roster['Occupation'].unique().tolist(),
                    key="roster_filter"
                )
                roster = roster[roster['Occupation'].isin(occ_filter)]

            st.dataframe(roster, height=400, use_container_width=True)
            st.download_button("📥 Download Roster", roster.to_csv(index=False), "personnel_roster_weekly.csv", "text/csv")

    with tab2:
        st.write("### Hourly Shuttle Requirements")
        st.caption("Shuttles are shared across all occupations — entry and exit times are combined.")
        if os.path.exists("shuttle_report_weekly.csv"):
            shuttle = pd.read_csv("shuttle_report_weekly.csv")
            st.dataframe(shuttle, height=400, use_container_width=True)
            st.download_button("📥 Download Shuttle Report", shuttle.to_csv(index=False), "shuttle_report_weekly.csv", "text/csv")

            _, fig_hourly, fig_daily = plot_weekly_results_interactive()
            c1, c2 = st.columns(2)
            if fig_daily:
                c1.plotly_chart(fig_daily, use_container_width=True)
            if fig_hourly:
                c2.plotly_chart(fig_hourly, use_container_width=True)

    with tab3:
        st.write("### Daily Coverage Graphs")
        if num_occ > 1:
            st.caption("Each occupation is shown as a stacked colored area. Dashed lines show per-occupation required workload. The bold red line is total required.")
        daily_figs, _, _ = plot_weekly_results_interactive()

        if daily_figs:
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            selected_day = st.selectbox("Select Day to View", days)
            if selected_day in daily_figs:
                st.plotly_chart(daily_figs[selected_day], use_container_width=True)
        else:
            st.info("Run the optimizer to generate graphs.")
