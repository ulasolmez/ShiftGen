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

def solve_weekly_shift_optimization(csv_path="workload_weekly.csv", max_fte=None, shuttle_interval=60, sparse_mode=False, shuttle_capacity=16, templates_path="shift_templates.csv", custom_shuttle_windows=None, max_shuttles=None, max_headcount=None, min_weekly_hours=35.0, max_weekly_hours=48.0, min_shift_length=4.0, max_shift_length=11.0, min_days_off=1.0, auto_shuttle=False, add_handover_buffer=False, apply_peak_cutting=False):
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
        # Window size of 2 intervals = 10 minutes. 
        # Any peak narrower than ~5 mins will be flattened.
        window = 2 
        # Pad for wrapping: add last 2 to start, first 2 to end to handle boundaries
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
            # ============ COLUMN GENERATION: PHASE 1 - Strategic Shift Generation ============
            # Instead of generating ALL possible shifts (which creates huge problems),
            # we generate shifts in phases based on workload patterns
            
            # Identify high-demand periods for this day
            day_workload = final_required[day_idx * 288 : (day_idx + 1) * 288]
            avg_demand = np.mean([d for d in day_workload if d > 0]) if any(d > 0 for d in day_workload) else 0
            
            # Phase 1: Generate "core" shifts covering peak periods
            # We focus on shifts that start/end near high-demand times
            possible_durations = [d * 0.5 for d in range(int(min_shift_length * 2), int(max_shift_length * 2) + 1)]
            
            # Prioritize shift start times based on workload changes
            prioritized_starts = []
            for start_time_str in shuttle_windows:
                h, m = map(int, start_time_str.split(":"))
                global_start_idx = day_idx * 288 + (h * 12 + m // 5)
                local_idx = global_start_idx % 288
                
                # Calculate workload intensity around this start time
                window_demand = 0
                for offset in range(-6, 6):  # ±30 min window
                    check_idx = (local_idx + offset) % 288
                    window_demand += day_workload[check_idx]
                
                # Prioritize starts during or near high-demand periods
                if window_demand > 0:
                    prioritized_starts.append((start_time_str, window_demand, global_start_idx))
            
            # Sort by demand (highest first) and take top candidates + some random coverage
            prioritized_starts.sort(key=lambda x: -x[1])
            
            # Strategy: Generate shifts for top 60% of demand points + every 4th low-demand point
            num_high_priority = max(3, int(len(prioritized_starts) * 0.6))
            selected_starts = prioritized_starts[:num_high_priority]
            
            # Add some low-demand starts for coverage (every 4th)
            for i, start_info in enumerate(prioritized_starts[num_high_priority:]):
                if i % 4 == 0:
                    selected_starts.append(start_info)
            
            for start_time_str, _, global_start_idx in selected_starts:
                for duration_hours in possible_durations:
                    end_time_str = get_end_time_str(start_time_str, duration_hours)
                    if end_time_str not in shuttle_windows:
                        continue
                    duration_intervals = int(duration_hours * 12)
                    
                    # Validate shift doesn't span excessive zero-workload periods
                    zero_count = 0
                    workload_covered = 0
                    for i in range(duration_intervals):
                        check_idx = (global_start_idx + i) % num_intervals
                        if final_required[check_idx] == 0:
                            zero_count += 1
                        else:
                            workload_covered += final_required[check_idx]
                    
                    # Skip shifts that are mostly during zero-demand periods
                    if zero_count > (duration_intervals * 0.5):
                        continue
                    
                    # Skip shifts that cover almost no workload (efficiency filter)
                    if workload_covered < (duration_hours * avg_demand * 0.1) and avg_demand > 0:
                        continue
                    
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
                        "global_end_idx": (global_start_idx + duration_intervals) % num_intervals,
                        "workload_score": workload_covered  # Track shift value
                    })

    print(f"Phase 1: Generated {len(shifts)} strategic shifts (filtered from full set)")
    
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
    under_covered_intervals = []  # Track intervals that need additional shift options
    
    for t in range(num_intervals):
        # Workload constraint
        if covers[t]:
            prob += pulp.lpSum([shift_vars[sid] for sid in covers[t]]) >= final_required[t]
        elif final_required[t] > 0:
            print(f"Warning: Interval {t} ({times_list[t]}) needs coverage but has no shifts - marking for Phase 2")
            under_covered_intervals.append(t)
        
        # Shuttle logic only if it's a shuttle window
        if t in shuttle_windows_indices:
            if starts_at[t]:
                prob += shuttle_in_vars[t] >= pulp.lpSum([shift_vars[sid] for sid in starts_at[t]]) / shuttle_capacity
            if ends_at[t]:
                prob += shuttle_out_vars[t] >= pulp.lpSum([shift_vars[sid] for sid in ends_at[t]]) / shuttle_capacity

    total_hours = pulp.lpSum([shift_vars[s["id"]] * s["duration"] for s in shifts])
    if max_fte is not None:
        prob += (total_hours / 45.0) <= max_fte
    
    # ============ PHASE 2: Generate Additional Shifts if Needed ============
    if under_covered_intervals:
        print(f"Phase 2: Generating targeted shifts for {len(under_covered_intervals)} under-covered intervals...")
        phase2_shifts = []
        
        for problem_idx in under_covered_intervals:
            day_idx = problem_idx // 288
            local_idx = problem_idx % 288
            
            # Find shuttle windows near this interval
            for offset in range(-12, 13, 3):  # Search ±1 hour in 15-min steps
                potential_start_idx = problem_idx + offset
                if potential_start_idx < 0 or potential_start_idx >= num_intervals:
                    continue
                
                start_local = potential_start_idx % 288
                start_hour = start_local // 12
                start_min = (start_local % 12) * 5
                start_time_str = f"{start_hour:02d}:{start_min:02d}"
                
                if start_time_str not in shuttle_windows:
                    continue
                
                # Try various durations that would cover the problem interval
                for duration_hours in [4.0, 6.0, 8.0, 10.0]:
                    end_time_str = get_end_time_str(start_time_str, duration_hours)
                    if end_time_str not in shuttle_windows:
                        continue
                    
                    duration_intervals = int(duration_hours * 12)
                    coverage = np.zeros(num_intervals, dtype=int)
                    for i in range(duration_intervals):
                        coverage[(potential_start_idx + i) % num_intervals] = 1
                    
                    # Check if this shift actually covers the problem interval
                    if coverage[problem_idx] == 0:
                        continue
                    
                    shift_id = f"Phase2_Day{day_idx}_{format_as_hhmm(start_time_str)}_{duration_hours}h"
                    if shift_id not in shift_vars:  # Don't duplicate
                        phase2_shifts.append({
                            "id": shift_id, "day_idx": day_idx, "day_name": days[day_idx],
                            "start_time": start_time_str, "end_time": end_time_str,
                            "duration": duration_hours, "coverage": coverage,
                            "display": f"{format_as_hhmm(start_time_str)}-{format_as_hhmm(end_time_str)}",
                            "global_start_idx": potential_start_idx,
                            "global_end_idx": (potential_start_idx + duration_intervals) % num_intervals
                        })
        
        if phase2_shifts:
            print(f"Adding {len(phase2_shifts)} Phase 2 shifts to cover gaps...")
            # Add new shifts to the problem
            for s in phase2_shifts:
                shifts.append(s)
                shift_vars[s["id"]] = pulp.LpVariable(s["id"], lowBound=0, cat='Integer')
                starts_at[s["global_start_idx"]].append(s["id"])
                ends_at[s["global_end_idx"]].append(s["id"])
                indices = np.where(s["coverage"] == 1)[0]
                for idx in indices:
                    if idx not in covers:
                        covers[idx] = []
                    covers[idx].append(s["id"])
            
            # Rebuild objective and constraints with new shifts
            prob.objective = pulp.lpSum([shift_vars[s["id"]] for s in shifts]) + 0.1 * pulp.lpSum([shuttle_in_vars[t] + shuttle_out_vars[t] for t in shuttle_windows_indices])
            
            # Update coverage constraints
            for t in under_covered_intervals:
                if covers[t]:
                    prob += pulp.lpSum([shift_vars[sid] for sid in covers[t]]) >= final_required[t]
            
            # Update shuttle constraints for new shifts
            for s in phase2_shifts:
                if s["global_start_idx"] in shuttle_windows_indices:
                    prob += shuttle_in_vars[s["global_start_idx"]] >= shift_vars[s["id"]] / shuttle_capacity
                if s["global_end_idx"] in shuttle_windows_indices:
                    prob += shuttle_out_vars[s["global_end_idx"]] >= shift_vars[s["id"]] / shuttle_capacity
            
            # Update total hours constraint
            prob.constraints["_C1"] = (pulp.lpSum([shift_vars[s["id"]] * s["duration"] for s in shifts]) / 45.0 <= max_fte) if max_fte else None
        
    # 4. Solve (with a 30s time limit to ensure responsiveness)
    print(f"Solving optimization with {len(shifts)} total shifts (Phase 1 + Phase 2)...")
    print(f"Parameters: Max FTE: {max_fte if max_fte else 'Unlimited'}, Min Hours: {min_weekly_hours}, Max Hours: {max_weekly_hours}")
    # Added timeLimit and gap tolerance to speed up the solver
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30, gapRel=0.01))
    
    if pulp.LpStatus[prob.status] != 'Optimal':
        print(f"Warning: Solution status = {pulp.LpStatus[prob.status]}")
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

    # Sort shifts: earliest start first; for same start, longest duration first
    # Longest duration first helps in packing hours more efficiently into existing workers
    all_assigned_shifts.sort(key=lambda x: (x["start"], -x["duration"]))
    
    # 12-hour rest requirement in intervals (12h * 12 intervals/h = 144)
    REST_INTERVALS = 144
    
    # Calculate theoretical headcount target
    total_shift_hours = sum(s['duration'] for s in all_assigned_shifts)
    theoretical_min_people = max(1, math.ceil(total_shift_hours / max_weekly_hours)) if max_weekly_hours > 0 else 1
    theoretical_max_people = max(theoretical_min_people, math.floor(total_shift_hours / min_weekly_hours)) if min_weekly_hours > 0 else theoretical_min_people * 2
    
    # Pre-initialize personnel slots targeting the theoretical minimum
    # This encourages the algorithm to pack shifts into fewer people
    personnel_pool = []
    for i in range(theoretical_min_people):
        personnel_pool.append({
            'id': i + 1,
            'end_time': -REST_INTERVALS - 1,  # Available from the start
            'weekly_hours': 0.0,
            'days_worked': set(),  # Track which days (0-6) this person has worked
            'shift_times': []  # Track (start_idx, end_idx) for consecutive rest calculation
        })
    
    roster_rows = []
    
    # Helper function to check if assigning a shift violates the off-day policy
    def violates_off_day_policy(person, shift_day_idx, shift_start_idx, shift_end_idx):
        if min_days_off == 1.0:
            # Maximum 1 day off allowed (must work at least 6 days)
            if len(person['days_worked']) >= 6 and shift_day_idx not in person['days_worked']:
                return True
        elif min_days_off == 1.5:
            # Maximum 1.5 days off: Must work enough to leave max 36 consecutive hours off
            # Check if adding this shift would prevent having 36 hours off
            test_shifts = person['shift_times'] + [(shift_start_idx, shift_end_idx)]
            test_shifts.sort()
            
            # Find longest gap between shifts
            max_gap = 0
            if test_shifts:
                # Gap before first shift (from start of week)
                max_gap = max(max_gap, test_shifts[0][0])
                # Gaps between shifts
                for i in range(len(test_shifts) - 1):
                    gap = test_shifts[i+1][0] - test_shifts[i][1]
                    max_gap = max(max_gap, gap)
                # Gap after last shift (to end of week)
                max_gap = max(max_gap, 2016 - test_shifts[-1][1])
            
            if max_gap < 72:  # 36 hours = 72 intervals
                return True
        elif min_days_off == 2.0:
            # Maximum 2 days off allowed (must work at least 5 days)
            if len(person['days_worked']) >= 5 and shift_day_idx not in person['days_worked']:
                return True
        return False

    for shift in all_assigned_shifts:
        assigned = False
        shift_day_idx = shift['start'] // 288  # Which day (0-6) this shift starts on
        
        # Calculate max working days based on off-day policy
        max_working_days = 7 - math.ceil(min_days_off)
        
        # Sorting Strategy for efficient packing:
        # 1. Prioritize workers who haven't maxed out their working days yet
        # 2. Among those, prioritize workers furthest below min hours
        # 3. Then those available earliest (by rest constraint)
        # 4. Finally those with most remaining hour capacity
        personnel_pool.sort(key=lambda x: (
            len(x['days_worked']) >= max_working_days,  # Haven't maxed days comes first
            x['weekly_hours'] >= min_weekly_hours,  # Below min hours comes first
            -1 * (min_weekly_hours - x['weekly_hours']) if x['weekly_hours'] < min_weekly_hours else 0,  # Furthest below min
            x['end_time'],  # Available earliest
            -1 * (max_weekly_hours - x['weekly_hours'])  # Most remaining capacity
        ))
        
        for p in personnel_pool:
            # Check 1: Rest Constraint (12h since their last shift ended)
            if shift["start"] >= p['end_time'] + REST_INTERVALS:
                # Check 2: Max Hours Constraint
                if p['weekly_hours'] + shift['duration'] <= max_weekly_hours:
                    # Check 3: Off-Day Policy Constraint
                    if not violates_off_day_policy(p, shift_day_idx, shift["start"], shift["end"]):
                        p['end_time'] = shift["end"]
                        p['weekly_hours'] += shift["duration"]
                        p['days_worked'].add(shift_day_idx)
                        p['shift_times'].append((shift["start"], shift["end"]))
                        p_id = p['id']
                        assigned = True
                        break
        
        if not assigned:
            # Before creating a new worker, verify that NONE of the existing workers can take this shift
            # due to constraints (not just sorting order)
            can_assign_to_existing = False
            for p in personnel_pool:
                if shift["start"] >= p['end_time'] + REST_INTERVALS:
                    if p['weekly_hours'] + shift['duration'] <= max_weekly_hours:
                        if not violates_off_day_policy(p, shift_day_idx, shift["start"], shift["end"]):
                            can_assign_to_existing = True
                            break
            
            # Only create new worker if absolutely no existing worker is eligible
            if not can_assign_to_existing:
                new_id = len(personnel_pool) + 1
                personnel_pool.append({
                    'id': new_id,
                    'end_time': shift["end"],
                    'weekly_hours': shift["duration"],
                    'days_worked': {shift_day_idx},
                    'shift_times': [(shift["start"], shift["end"])]
                })
                p_id = new_id
            else:
                # Should have been assigned in the loop above, this is a logic error
                # Re-try assignment with the first eligible worker found
                for p in personnel_pool:
                    if shift["start"] >= p['end_time'] + REST_INTERVALS:
                        if p['weekly_hours'] + shift['duration'] <= max_weekly_hours:
                            if not violates_off_day_policy(p, shift_day_idx, shift["start"], shift["end"]):
                                p['end_time'] = shift["end"]
                                p['weekly_hours'] += shift["duration"]
                                p['days_worked'].add(shift_day_idx)
                                p['shift_times'].append((shift["start"], shift["end"]))
                                p_id = p['id']
                                assigned = True
                                break
            
        roster_rows.append({
            "Personnel_ID": f"EMP_{p_id:03d}",
            "Day": shift["day_idx"],
            "Shift": shift["display"],
            "Start": shift["raw_start"],
            "End": shift["raw_end"],
            "duration_intervals": int(shift["duration"] * 12),
            "start_idx": shift["start"]
        })

    # Summary Stats
    total_unique_personnel = len(personnel_pool)
    people_below_min = sum(1 for p in personnel_pool if p['weekly_hours'] < min_weekly_hours)
    
    # 6. Save Outputs
    pd.DataFrame(roster_rows).to_csv("personnel_roster_weekly.csv", index=False)
    
    # Calculate Arrivals/Departures from roster for shuttle report
    arrivals = np.zeros(num_intervals)
    departures = np.zeros(num_intervals)
    for r in roster_rows:
        s_idx = r['start_idx']
        d_ints = r['duration_intervals']
        e_idx = (s_idx + d_ints) % num_intervals
        
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
    # Theoretical range based on min/max hours
    min_headcount_theoretical = math.ceil(total_hours_val / max_weekly_hours) if max_weekly_hours > 0 else 0
    max_headcount_theoretical = math.floor(total_hours_val / min_weekly_hours) if min_weekly_hours > 0 else total_unique_personnel
    
    pd.DataFrame([{
        "Total Hours": total_hours_val, 
        "FTE": round(fte_val, 2),
        "Headcount": total_unique_personnel,
        "Theoretical Min Headcount": min_headcount_theoretical,
        "Theoretical Max Headcount": max_headcount_theoretical,
        "People Below Min Hours": people_below_min,
        "Min Weekly Hours": min_weekly_hours,
        "Max Weekly Hours": max_weekly_hours,
        "Min Days Off": min_days_off,
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
