import pandas as pd
import pulp
import numpy as np
from datetime import datetime, timedelta
import math
import os

def format_as_hhmm(time_str):
    return time_str.replace(":", "")

def get_end_time_str(start_time_str, duration_hours):
    start_dt = datetime.strptime(start_time_str, "%H:%M")
    end_dt = start_dt + timedelta(hours=duration_hours)
    return end_dt.strftime("%H:%M")

def solve_weekly_shift_optimization(csv_path="workload_weekly.csv", max_fte=None, shuttle_interval=60, sparse_mode=False, shuttle_capacity=16, templates_path="shift_templates.csv", custom_shuttle_windows=None, max_shuttles=None, max_headcount=None, max_weekly_hours=48.0, min_shift_length=4.0, max_shift_length=11.0, auto_shuttle=False, add_handover_buffer=False, apply_peak_cutting=False):
    # 1. Load data
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run generate_data.py first.")
        return
    df = pd.read_csv(csv_path)

    # Robust Column Handling:
    # If standard columns are missing, assume the first 3 columns are [Day, Time, Headcount]
    expected_cols = ["day_name", "time", "required_headcount"]
    if not all(col in df.columns for col in expected_cols):
        if len(df.columns) >= 3:
            # Create a map of old names to new names for the first 3 columns
            rename_map = {df.columns[0]: "day_name", df.columns[1]: "time", df.columns[2]: "required_headcount"}
            df = df.rename(columns=rename_map)
        else:
            print("Error: Input CSV must have at least 3 columns (Day, Time, Headcount).")
            return

    # Internal Normalization: Ensure 'time' column is HH:MM (08:30 not 8:30)
    # This prevents aggregation errors later if the input CSV is loosely formatted
    if 'time' in df.columns:
        df['time'] = df['time'].astype(str).apply(lambda x: x.split(" ")[-1]) # Remove date if present
        df['time'] = df['time'].apply(lambda x: f"{int(x.split(':')[0]):02d}:{int(x.split(':')[1]):02d}" if ':' in x and len(x) < 5 else x)

    required = df["required_headcount"].tolist()
    num_intervals = len(df) # 7 * 288 = 2016
    
    # 1.4 Apply Peak Cutting (Short-duration spike removal)
    # We do this BEFORE buffering so we don't accidentally widen a noise spike
    if apply_peak_cutting:
        print("Applying peak cutting to ignore short-duration spikes...")
        # Window size of 6 intervals = 30 minutes. 
        # Any peak narrower than ~15-20 mins will be flattened.
        window = 6 
        # Pad for wrapping: add last 6 to start, first 6 to end to handle boundaries
        padded = required[-window:] + required + required[:window]
        s_padded = pd.Series(padded)
        
        # Morphological Opening: Erosion (Min) followed by Dilation (Max)
        # This removes small positive objects (peaks) without affecting bulk
        eroded = s_padded.rolling(window=window, center=True, min_periods=1).min()
        opened = eroded.rolling(window=window, center=True, min_periods=1).max()
        
        # Crop back to original size
        # Start index is 'window', length is 'num_intervals'
        cut_required = opened.tolist()[window : window + num_intervals]
        
        # Ensure we only cut down, never increase
        required = [min(r, c) for r, c in zip(required, cut_required)]

    # 1.5 Apply Handover/Prep Buffer
    final_required = required.copy()
    if add_handover_buffer:
        print("Applying 30-min prep/handover buffer...")
        # Dilate the workload curve by +/- 30 mins (6 intervals)
        buffer_steps = 6 
        buffered = [0] * num_intervals
        for i in range(num_intervals):
            # Look ahead and behind to find the maximum requirement in the window
            # This forces shifts to start early for ramp-ups and stay late for ramp-downs
            window_vals = []
            for offset in range(-buffer_steps, buffer_steps + 1):
                idx = (i + offset) % num_intervals
                window_vals.append(required[idx])
            buffered[i] = max(window_vals)
        final_required = buffered
    
    # 2. Define possible shifts & Shuttle Times
    if auto_shuttle:
        print("Providing 30-min grid for auto-shuttle optimization...")
        shuttle_windows = [f"{h:02d}:{m:02d}" for h in range(24) for m in [0, 30]]
    elif custom_shuttle_windows:
        shuttle_windows = custom_shuttle_windows
    else:
        all_times = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 5)]
        shuttle_windows = [t for i, t in enumerate(all_times) if (i * 5) % shuttle_interval == 0]
        
        if sparse_mode:
            sparse_windows = ["07:00", "08:00", "15:00", "16:00", "22:00", "23:00", "00:00"]
            shuttle_windows = [t for t in shuttle_windows if t in sparse_windows]
    
    print(f"Using {len(shuttle_windows)} shuttle windows per day.")
    use_templates = False
    if os.path.exists(templates_path):
        templates_df = pd.read_csv(templates_path)
        print(f"Loading {len(templates_df)} pre-assigned shift templates from {templates_path}")
        use_templates = True

    shifts = []
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    for day_idx in range(7):
        if use_templates:
            for _, row in templates_df.iterrows():
                start_time_str = row['start']
                duration_hours = float(row['duration'])
                h, m = map(int, start_time_str.split(":"))
                global_start_idx = day_idx * 288 + (h * 12 + m // 5)
                duration_intervals = int(duration_hours * 12)
                coverage = np.zeros(num_intervals, dtype=int)
                for i in range(duration_intervals):
                    coverage[(global_start_idx + i) % num_intervals] = 1
                end_t = get_end_time_str(start_time_str, duration_hours)
                shift_display = f"{format_as_hhmm(start_time_str)}-{format_as_hhmm(end_t)}"
                shift_id = f"Day{day_idx}_{format_as_hhmm(start_time_str)}_{duration_hours}h"
                shifts.append({
                    "id": shift_id, "day_idx": day_idx, "day_name": days[day_idx],
                    "start_time": start_time_str, "end_time": end_t,
                    "duration": duration_hours, "coverage": coverage, "display": shift_display,
                    "global_start_idx": global_start_idx,
                    "global_end_idx": (global_start_idx + duration_intervals) % num_intervals
                })
        else:
            # Generate possible shift durations based on min/max constraints
            # We step by 0.5 hours (30 mins)
            possible_durations = [d * 0.5 for d in range(int(min_shift_length * 2), int(max_shift_length * 2) + 1)]
            
            for start_time_str in shuttle_windows:
                h, m = map(int, start_time_str.split(":"))
                global_start_idx = day_idx * 288 + (h * 12 + m // 5)
                for duration_hours in possible_durations:
                    end_time_str = get_end_time_str(start_time_str, duration_hours)
                    if end_time_str not in shuttle_windows:
                        continue
                    duration_intervals = int(duration_hours * 12)
                    coverage = np.zeros(num_intervals, dtype=int)
                    for i in range(duration_intervals):
                        coverage[(global_start_idx + i) % num_intervals] = 1
                    shift_display = f"{format_as_hhmm(start_time_str)}-{format_as_hhmm(end_time_str)}"
                    shift_id = f"Day{day_idx}_{format_as_hhmm(start_time_str)}_{duration_hours}h"
                    shifts.append({
                        "id": shift_id, "day_idx": day_idx, "day_name": days[day_idx],
                        "start_time": start_time_str, "end_time": end_time_str,
                        "duration": duration_hours, "coverage": coverage, "display": shift_display,
                        "global_start_idx": global_start_idx,
                        "global_end_idx": (global_start_idx + duration_intervals) % num_intervals
                    })

    # 3. Create the Optimization Problem
    prob = pulp.LpProblem("Weekly_Shuttle_Shift_Optimization", pulp.LpMinimize)
    shift_vars = pulp.LpVariable.dicts("Shifts", [s["id"] for s in shifts], lowBound=0, cat='Integer')
    
    # Pre-calculate shift indices for faster constraint addition
    starts_at = {t: [] for t in range(num_intervals)}
    ends_at = {t: [] for t in range(num_intervals)}
    covers = {t: [] for t in range(num_intervals)}
    
    for s in shifts:
        starts_at[s["global_start_idx"]].append(s["id"])
        ends_at[s["global_end_idx"]].append(s["id"])
        # Find which indices this shift covers
        indices = np.where(s["coverage"] == 1)[0]
        for idx in indices:
            covers[idx].append(s["id"])

    # Shuttle variables only for shuttle window intervals
    shuttle_windows_indices = []
    times_list = df["time"].tolist()
    for t in range(num_intervals):
        if times_list[t] in shuttle_windows:
            shuttle_windows_indices.append(t)
            
    shuttle_in_vars = pulp.LpVariable.dicts("ShuttleIn", shuttle_windows_indices, lowBound=0, cat='Integer')
    shuttle_out_vars = pulp.LpVariable.dicts("ShuttleOut", shuttle_windows_indices, lowBound=0, cat='Integer')

    # Objective: Minimize headcount + small penalty for shuttles
    prob += pulp.lpSum([shift_vars[s["id"]] for s in shifts]) + 0.1 * pulp.lpSum([shuttle_in_vars[t] + shuttle_out_vars[t] for t in shuttle_windows_indices])
    
    print("Adding workload and shuttle constraints...")
    for t in range(num_intervals):
        # Workload constraint
        if covers[t]:
            prob += pulp.lpSum([shift_vars[sid] for sid in covers[t]]) >= final_required[t]
        elif final_required[t] > 0:
            print(f"Warning: No possible shifts cover interval {t} ({times_list[t]})")
        
        # Shuttle logic only if it's a shuttle window
        if t in shuttle_windows_indices:
            if starts_at[t]:
                prob += shuttle_in_vars[t] >= pulp.lpSum([shift_vars[sid] for sid in starts_at[t]]) / shuttle_capacity
            if ends_at[t]:
                prob += shuttle_out_vars[t] >= pulp.lpSum([shift_vars[sid] for sid in ends_at[t]]) / shuttle_capacity

    total_hours = pulp.lpSum([shift_vars[s["id"]] * s["duration"] for s in shifts])
    if max_fte is not None:
        prob += (total_hours / 45.0) <= max_fte
        
    # 4. Solve (with a 30s time limit to ensure responsiveness)
    print(f"Solving optimization (Shuttle-aligned, Max FTE: {max_fte if max_fte else 'Unlimited'})...")
    # Added timeLimit and gap tolerance to speed up the solver
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30, gapRel=0.01))
    
    if pulp.LpStatus[prob.status] != 'Optimal':
        print("Warning: No optimal solution found.")
        return None

    # 5. Extract Results and Assign to Personnel with 12h Rest
    assigned_shifts_types = []
    total_coverage = np.zeros(num_intervals)
    total_hours_val = 0
    
    # List of all specific shifts to be assigned (Day, StartIdx, EndIdx, Display, Duration)
    all_assigned_shifts = []
    
    for s in shifts:
        val = pulp.value(shift_vars[s["id"]])
        if val > 0:
            count = int(val)
            assigned_shifts_types.append({
                "Day": s["day_name"], "Shift": s["display"], "Duration": s["duration"], "Count": count
            })
            total_coverage += val * s["coverage"]
            total_hours_val += val * s["duration"]
            
            h_start, m_start = map(int, s['start_time'].split(":"))
            start_idx = s['day_idx'] * 288 + (h_start * 12 + m_start // 5)
            # Use absolute end index for simpler transition math
            # No modulo here yet to track progression across days
            duration_intervals = int(s['duration'] * 12)
            end_idx_absolute = start_idx + duration_intervals
            
            for _ in range(count):
                all_assigned_shifts.append({
                    "day_idx": s["day_name"],
                    "start": start_idx,
                    "end": end_idx_absolute,
                    "display": s["display"],
                    "duration": s["duration"],
                    "raw_start": s["start_time"],
                    "raw_end": s["end_time"]
                })

    # Sort shifts by start time to assign greedily
    all_assigned_shifts.sort(key=lambda x: x["start"])
    
    # 12-hour rest requirement in intervals (12h * 12 intervals/h = 144)
    REST_INTERVALS = 144
    
    # Personnel is a list of dicts: {'end_time': int, 'weekly_hours': float, 'id': int}
    personnel_pool = [] 
    roster_rows = []

    for shift in all_assigned_shifts:
        assigned = False
        # Try to find a person who has rested enough AND has not exceeded weekly hours
        # Sort people by their end_time to reuse those available earliest, minimizing headcount
        personnel_pool.sort(key=lambda x: x['end_time'])
        
        for p in personnel_pool:
            # Check 1: Rest Constraint
            if shift["start"] >= p['end_time'] + REST_INTERVALS:
                # Check 2: Max Hours Constraint
                if p['weekly_hours'] + shift['duration'] <= max_weekly_hours:
                    p['end_time'] = shift["end"]
                    p['weekly_hours'] += shift["duration"]
                    p_id = p['id']
                    assigned = True
                    break
        
        if not assigned:
            # Need a new person
            new_id = len(personnel_pool) + 1
            personnel_pool.append({
                'id': new_id,
                'end_time': shift["end"],
                'weekly_hours': shift["duration"]
            })
            p_id = new_id
            
        roster_rows.append({
            "Personnel_ID": f"EMP_{p_id:03d}",
            "Day": shift["day_idx"],
            "Shift": shift["display"],
            "Start": shift["raw_start"],
            "End": shift["raw_end"]
        })

    # Summary Stats
    total_unique_personnel = len(personnel_pool)
    
    # 6. Save Outputs
    pd.DataFrame(roster_rows).to_csv("personnel_roster_weekly.csv", index=False)
    
    # Calculate Arrivals/Departures from roster for shuttle report
    arrivals = np.zeros(num_intervals)
    departures = np.zeros(num_intervals)
    for r in roster_rows:
        day_df = df[df['day_name'] == r['Day']]
        s_idx = day_df[day_df['time'] == r['Start']].index[0]
        # Duration is needed to find end_idx
        h_s, m_s = map(int, r['Start'].split(":"))
        h_e, m_e = map(int, r['End'].split(":"))
        # Using a simple duration calculation for end_idx
        # Find duration from shift display "HHMM-HHMM"
        start_hhmm = int(r['Shift'].split("-")[0])
        end_hhmm = int(r['Shift'].split("-")[1])
        # It's better to just re-match end time in df
        e_idx = day_df[day_df['time'] == r['End']].index[0] if r['End'] in day_df['time'].values else (s_idx + 96) % num_intervals
        
        arrivals[s_idx] += 1
        departures[e_idx] += 1

    shuttle_data = []
    for t in range(num_intervals):
        if arrivals[t] > 0 or departures[t] > 0:
            arr_sh = math.ceil(arrivals[t] / shuttle_capacity)
            dep_sh = math.ceil(departures[t] / shuttle_capacity)
            shuttle_data.append({
                "Day": df.iloc[t]["day_name"], "Time": df.iloc[t]["time"],
                "Arrivals": int(arrivals[t]), "Departures": int(departures[t]),
                "Shuttles_In": arr_sh, "Shuttles_Out": dep_sh, "Total_Shuttles": arr_sh + dep_sh
            })
    pd.DataFrame(shuttle_data).to_csv("shuttle_report_weekly.csv", index=False)
    
    fte_val = total_hours_val / 45.0
    pd.DataFrame([{
        "Total Hours": total_hours_val, 
        "FTE": round(fte_val, 2),
        "Headcount": total_unique_personnel,
        "Unique Personnel Needed": total_unique_personnel,
        "Total Weekly Shuttles": sum(x['Total_Shuttles'] for x in shuttle_data), 
        "Max FTE Allowed": max_fte,
        "Max Headcount Allowed": max_headcount if max_headcount else "Unlimited"
    }]).to_csv("weekly_summary.csv", index=False)

    day_order = {day: i for i, day in enumerate(days)}
    results_df = pd.DataFrame(assigned_shifts_types)
    results_df['day_id'] = results_df['Day'].map(day_order)
    results_df.sort_values(['day_id', 'Shift']).drop('day_id', axis=1).to_csv("assigned_shifts_weekly.csv", index=False)
    df["actual_coverage"] = total_coverage
    df.to_csv("weekly_coverage_comparison.csv", index=False)
    print(f"Optimization complete. FTE: {fte_val:.2f}, Shuttles: {sum(x['Total_Shuttles'] for x in shuttle_data)}")
    print("Individual roster saved to personnel_roster_weekly.csv")
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Shift Optimizer')
    parser.add_argument('max_fte', type=float, nargs='?', default=None)
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--capacity', type=int, default=16)
    args = parser.parse_args()
    solve_weekly_shift_optimization(max_fte=args.max_fte, shuttle_interval=args.interval, sparse_mode=args.sparse, shuttle_capacity=args.capacity)
