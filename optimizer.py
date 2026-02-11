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

def _preprocess_workload(df, apply_peak_cutting, add_handover_buffer):
    """Preprocess a single workload dataframe (peak cutting, buffer)."""
    expected_cols = ["day_name", "time", "required_headcount"]
    if not all(col in df.columns for col in expected_cols):
        if len(df.columns) >= 3:
            rename_map = {df.columns[0]: "day_name", df.columns[1]: "time", df.columns[2]: "required_headcount"}
            df = df.rename(columns=rename_map)
        else:
            raise ValueError("Input CSV must have at least 3 columns (Day, Time, Headcount).")

    if 'time' in df.columns:
        df['time'] = df['time'].astype(str).apply(lambda x: x.split(" ")[-1])
        df['time'] = df['time'].apply(lambda x: f"{int(x.split(':')[0]):02d}:{int(x.split(':')[1]):02d}" if ':' in x and len(x) < 5 else x)

    required = df["required_headcount"].tolist()
    num_intervals = len(df)

    if apply_peak_cutting:
        window = 2
        padded = required[-window:] + required + required[:window]
        s_padded = pd.Series(padded)
        eroded = s_padded.rolling(window=window, center=True, min_periods=1).min()
        opened = eroded.rolling(window=window, center=True, min_periods=1).max()
        cut_required = opened.tolist()[window : window + num_intervals]
        required = [min(r, c) for r, c in zip(required, cut_required)]

    final_required = required.copy()
    if add_handover_buffer:
        buffer_steps = 6
        buffered = [0] * num_intervals
        for i in range(num_intervals):
            window_vals = []
            for offset in range(-buffer_steps, buffer_steps + 1):
                idx = (i + offset) % num_intervals
                window_vals.append(required[idx])
            buffered[i] = max(window_vals)
        final_required = buffered

    return df, final_required


def _detect_smart_shuttle_windows(all_occupation_data):
    """
    Detect shuttle windows from workload shape across ALL occupations.
    
    Provides hourly windows during working hours so the solver has enough
    granularity to match the workload curve tightly. The shuttle consolidation
    penalty in the objective will discourage using too many of them.
    """
    num_intervals = len(all_occupation_data[0]["final_required"])
    combined = np.zeros(num_intervals)
    for occ in all_occupation_data:
        combined += np.array(occ["final_required"])

    # Find the work envelope across all days
    earliest_start = 24
    latest_end = 0
    for day_idx in range(7):
        day_start = day_idx * 288
        day_demand = combined[day_start:day_start + 288]
        nonzero = np.nonzero(day_demand)[0]
        if len(nonzero) == 0:
            continue
        start_h = nonzero[0] // 12
        end_h = min((nonzero[-1] // 12) + 1, 23)
        earliest_start = min(earliest_start, start_h)
        latest_end = max(latest_end, end_h)

    if earliest_start >= latest_end:
        return [f"{h:02d}:00" for h in range(0, 24, 2)]

    # Provide half-hourly windows from 1h before work start to 1h after work end
    window_start = max(0, earliest_start - 1)
    window_end = min(23, latest_end + 1)
    
    result = []
    for h in range(window_start, window_end + 1):
        result.append(f"{h:02d}:00")
        result.append(f"{h:02d}:30")
    
    # Always include 00:00 for overnight boundary
    if "00:00" not in result:
        result = ["00:00"] + result
    
    return sorted(set(result))


def _generate_shifts_for_occupation(day_idx, shuttle_windows, final_required, num_intervals,
                                     min_shift_length, max_shift_length, occ_prefix,
                                     use_templates=False, templates_df=None):
    """Generate shift candidates for one day of one occupation."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    shifts = []

    if use_templates and templates_df is not None:
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
            shift_id = f"{occ_prefix}_Day{day_idx}_{format_as_hhmm(start_time_str)}_{duration_hours}h"
            shifts.append({
                "id": shift_id, "day_idx": day_idx, "day_name": days[day_idx],
                "start_time": start_time_str, "end_time": end_t,
                "duration": duration_hours, "coverage": coverage, "display": shift_display,
                "global_start_idx": global_start_idx,
                "global_end_idx": (global_start_idx + duration_intervals) % num_intervals
            })
    else:
        possible_durations = [d * 0.5 for d in range(int(min_shift_length * 2), int(max_shift_length * 2) + 1)]
        for start_time_str in shuttle_windows:
            h, m = map(int, start_time_str.split(":"))
            global_start_idx = day_idx * 288 + (h * 12 + m // 5)
            for duration_hours in possible_durations:
                end_time_str = get_end_time_str(start_time_str, duration_hours)
                if end_time_str not in shuttle_windows:
                    continue
                duration_intervals = int(duration_hours * 12)

                zero_count = 0
                useful_intervals = 0
                for i in range(duration_intervals):
                    check_idx = (global_start_idx + i) % num_intervals
                    if final_required[check_idx] == 0:
                        zero_count += 1
                    else:
                        useful_intervals += 1
                if zero_count > (duration_intervals * 0.5):
                    continue

                # Store useful coverage ratio for objective weighting
                useful_ratio = useful_intervals / duration_intervals if duration_intervals > 0 else 0

                coverage = np.zeros(num_intervals, dtype=int)
                for i in range(duration_intervals):
                    coverage[(global_start_idx + i) % num_intervals] = 1
                shift_display = f"{format_as_hhmm(start_time_str)}-{format_as_hhmm(end_time_str)}"
                shift_id = f"{occ_prefix}_Day{day_idx}_{format_as_hhmm(start_time_str)}_{duration_hours}h"
                # waste_penalty: how much idle time this shift has (covering zero-demand intervals)
                waste_penalty = 1.0 - useful_ratio  # 0 = perfect, 1 = all waste
                shifts.append({
                    "id": shift_id, "day_idx": day_idx, "day_name": days[day_idx],
                    "start_time": start_time_str, "end_time": end_time_str,
                    "duration": duration_hours, "coverage": coverage, "display": shift_display,
                    "global_start_idx": global_start_idx,
                    "global_end_idx": (global_start_idx + duration_intervals) % num_intervals,
                    "waste_penalty": waste_penalty
                })
    return shifts


def _assign_personnel(all_assigned_shifts, max_weekly_hours, min_weekly_hours, min_days_off, occ_prefix):
    """
    Two-pass personnel assignment to prevent Sunday starvation.
    
    Problem: Chronological assignment (Mon→Sun) exhausts workers by Sunday,
    forcing creation of new workers who only get ~10-20h of Sunday shifts.
    
    Solution: Interleaved day-balanced assignment.
    Pass 1: Assign Sunday shifts first to seed the worker pool.
    Pass 2: Fill remaining days (Sat→Mon) into Sunday-seeded workers.
    This ensures Sunday workers accumulate 35-48h across the full week.
    """
    REST_INTERVALS = 144  # 12 hours in 5-min intervals
    total_shift_hours = sum(s['duration'] for s in all_assigned_shifts)
    theoretical_min_people = max(1, math.ceil(total_shift_hours / max_weekly_hours)) if max_weekly_hours > 0 else 1
    max_working_days = 7 - math.ceil(min_days_off)

    personnel_pool = []
    for i in range(theoretical_min_people):
        personnel_pool.append({
            'id': i + 1,
            'weekly_hours': 0.0,
            'days_worked': set(),
            'shift_times': []  # list of (start_idx, end_idx) tuples
        })

    roster_rows = []
    shift_to_person = {}  # shift index -> person id

    def violates_off_day_policy(person, shift_day_idx, shift_start_idx, shift_end_idx):
        if min_days_off == 1.0:
            if len(person['days_worked']) >= 6 and shift_day_idx not in person['days_worked']:
                return True
        elif min_days_off == 1.5:
            test_shifts = person['shift_times'] + [(shift_start_idx, shift_end_idx)]
            test_shifts.sort()
            max_gap = 0
            if test_shifts:
                max_gap = max(max_gap, test_shifts[0][0])
                for i in range(len(test_shifts) - 1):
                    gap = test_shifts[i+1][0] - test_shifts[i][1]
                    max_gap = max(max_gap, gap)
                max_gap = max(max_gap, 2016 - test_shifts[-1][1])
            if max_gap < 72:
                return True
        elif min_days_off == 2.0:
            if len(person['days_worked']) >= 5 and shift_day_idx not in person['days_worked']:
                return True
        return False

    def respects_rest(person, shift_start, shift_end):
        """Check 12h rest between all existing shifts and the new shift."""
        for (s, e) in person['shift_times']:
            # New shift must not overlap and must have 12h gap in both directions
            if not (shift_end + REST_INTERVALS <= s or shift_start >= e + REST_INTERVALS):
                return False
        return True

    def try_assign(shift, prefer_underfilled=True):
        """Try to assign a shift to an existing worker. Returns person_id or None."""
        shift_day_idx = shift['start'] // 288

        # Sort: prioritize workers who need more hours (below min), then by capacity
        if prefer_underfilled:
            personnel_pool.sort(key=lambda x: (
                len(x['days_worked']) >= max_working_days and shift_day_idx not in x['days_worked'],
                x['weekly_hours'] >= min_weekly_hours,
                -1 * (min_weekly_hours - x['weekly_hours']) if x['weekly_hours'] < min_weekly_hours else 0,
                len(x['shift_times']),
                -1 * (max_weekly_hours - x['weekly_hours'])
            ))
        else:
            personnel_pool.sort(key=lambda x: (
                len(x['days_worked']) >= max_working_days and shift_day_idx not in x['days_worked'],
                -1 * x['weekly_hours'],  # prefer fuller workers (pack them tight)
            ))

        for p in personnel_pool:
            if p['weekly_hours'] + shift['duration'] > max_weekly_hours:
                continue
            if not respects_rest(p, shift['start'], shift['end']):
                continue
            if violates_off_day_policy(p, shift_day_idx, shift['start'], shift['end']):
                continue
            # Assign
            p['weekly_hours'] += shift['duration']
            p['days_worked'].add(shift_day_idx)
            p['shift_times'].append((shift['start'], shift['end']))
            return p['id']
        return None

    def create_new_worker(shift):
        """Create a new worker and assign the shift."""
        shift_day_idx = shift['start'] // 288
        new_id = len(personnel_pool) + 1
        personnel_pool.append({
            'id': new_id,
            'weekly_hours': shift['duration'],
            'days_worked': {shift_day_idx},
            'shift_times': [(shift['start'], shift['end'])]
        })
        return new_id

    # ========= SPLIT SHIFTS BY DAY =========
    # Group shifts and remember original indices for roster output
    indexed_shifts = list(enumerate(all_assigned_shifts))
    
    sunday_shifts = [(i, s) for i, s in indexed_shifts if s['start'] // 288 == 6]
    saturday_shifts = [(i, s) for i, s in indexed_shifts if s['start'] // 288 == 5]
    weekday_shifts = [(i, s) for i, s in indexed_shifts if s['start'] // 288 < 5]

    # Sort within each group by start time, prefer longer shifts first
    sunday_shifts.sort(key=lambda x: (x[1]['start'], -x[1]['duration']))
    saturday_shifts.sort(key=lambda x: (x[1]['start'], -x[1]['duration']))
    weekday_shifts.sort(key=lambda x: (x[1]['start'], -x[1]['duration']))

    # ========= PASS 1: SEED WITH SUNDAY SHIFTS =========
    # Assign Sunday shifts first — these workers will get filled with earlier-week shifts
    for idx, shift in sunday_shifts:
        p_id = try_assign(shift, prefer_underfilled=True)
        if p_id is None:
            p_id = create_new_worker(shift)
        shift_to_person[idx] = p_id

    sunday_workers = len(personnel_pool)
    print(f"    Pass 1 (Sunday seed): {len(sunday_shifts)} shifts → {sunday_workers} workers")

    # ========= PASS 2: FILL SATURDAY (BACKWARD) =========
    for idx, shift in saturday_shifts:
        p_id = try_assign(shift, prefer_underfilled=True)
        if p_id is None:
            p_id = create_new_worker(shift)
        shift_to_person[idx] = p_id

    sat_workers = len(personnel_pool)
    print(f"    Pass 2 (Saturday fill): {len(saturday_shifts)} shifts → {sat_workers - sunday_workers} new workers")

    # ========= PASS 3: FILL WEEKDAYS (MON→FRI) =========
    for idx, shift in weekday_shifts:
        p_id = try_assign(shift, prefer_underfilled=True)
        if p_id is None:
            p_id = create_new_worker(shift)
        shift_to_person[idx] = p_id

    total_workers = len(personnel_pool)
    print(f"    Pass 3 (Weekday fill): {len(weekday_shifts)} shifts → {total_workers - sat_workers} new workers")

    # ========= PASS 4: REDISTRIBUTION =========
    # Try to move shifts from overloaded workers to underutilized ones
    underfilled = [p for p in personnel_pool if p['weekly_hours'] < min_weekly_hours]
    if underfilled:
        # For each underfilled worker, see if we can steal compatible shifts from other workers
        # who have shifts on days the underfilled worker doesn't work yet
        redistributed = 0
        for uf in underfilled:
            if uf['weekly_hours'] >= min_weekly_hours:
                continue
            needed = min_weekly_hours - uf['weekly_hours']
            # Look for shifts we can move from workers above min hours
            for donor in personnel_pool:
                if donor['id'] == uf['id']:
                    continue
                if donor['weekly_hours'] <= min_weekly_hours:
                    continue  # Don't steal from other underfilled workers
                
                # Try each of the donor's shifts
                for shift_time in list(donor['shift_times']):
                    s_start, s_end = shift_time
                    s_day = s_start // 288
                    s_duration = (s_end - s_start) / 12.0
                    
                    # Would donor still be above min after losing this shift?
                    if donor['weekly_hours'] - s_duration < min_weekly_hours:
                        continue
                    # Can underfilled worker take it?
                    if uf['weekly_hours'] + s_duration > max_weekly_hours:
                        continue
                    if not respects_rest(uf, s_start, s_end):
                        continue
                    if violates_off_day_policy(uf, s_day, s_start, s_end):
                        continue
                    
                    # Move the shift
                    donor['shift_times'].remove(shift_time)
                    donor['weekly_hours'] -= s_duration
                    donor['days_worked'] = set(t[0] // 288 for t in donor['shift_times'])
                    
                    uf['shift_times'].append(shift_time)
                    uf['weekly_hours'] += s_duration
                    uf['days_worked'].add(s_day)
                    
                    # Update roster mapping
                    for orig_idx, pid in shift_to_person.items():
                        if pid == donor['id']:
                            orig_shift = all_assigned_shifts[orig_idx]
                            if orig_shift['start'] == s_start and orig_shift['end'] == s_end:
                                shift_to_person[orig_idx] = uf['id']
                                break
                    
                    redistributed += 1
                    if uf['weekly_hours'] >= min_weekly_hours:
                        break
                if uf['weekly_hours'] >= min_weekly_hours:
                    break
        
        if redistributed > 0:
            print(f"    Redistribution: moved {redistributed} shifts to fill underfilled workers")

    # ========= PASS 5: REMOVE EMPTY WORKERS & COMPACT IDs =========
    # Remove workers with 0 shifts (possible after redistribution)
    active_workers = [p for p in personnel_pool if p['shift_times']]
    
    # Re-number workers by total hours descending (cosmetic)
    active_workers.sort(key=lambda x: -x['weekly_hours'])
    id_remap = {}
    for new_idx, p in enumerate(active_workers):
        id_remap[p['id']] = new_idx + 1
        p['id'] = new_idx + 1

    # ========= BUILD ROSTER =========
    for orig_idx in range(len(all_assigned_shifts)):
        shift = all_assigned_shifts[orig_idx]
        old_pid = shift_to_person[orig_idx]
        new_pid = id_remap.get(old_pid, old_pid)
        roster_rows.append({
            "Personnel_ID": f"{occ_prefix}_{new_pid:03d}",
            "Day": shift["day_idx"],
            "Shift": shift["display"],
            "Start": shift["raw_start"],
            "End": shift["raw_end"],
            "duration_intervals": int(shift["duration"] * 12),
            "start_idx": shift["start"]
        })

    people_below_min = sum(1 for p in active_workers if p['weekly_hours'] < min_weekly_hours)
    
    # Print hours distribution
    hours = [p['weekly_hours'] for p in active_workers]
    if hours:
        print(f"    Hours distribution: min={min(hours):.1f}, avg={sum(hours)/len(hours):.1f}, max={max(hours):.1f}")
        below_20 = sum(1 for h in hours if h < 20)
        below_min = sum(1 for h in hours if h < min_weekly_hours)
        print(f"    Workers: {len(active_workers)} total, {below_20} below 20h, {below_min} below {min_weekly_hours}h min")

    return roster_rows, active_workers, people_below_min


def solve_weekly_shift_optimization(
    workloads=None,
    csv_path="workload_weekly.csv",
    max_fte=None,
    shuttle_interval=60,
    sparse_mode=False,
    shuttle_capacity=16,
    templates_path="shift_templates.csv",
    custom_shuttle_windows=None,
    max_shuttles=None,
    max_headcount=None,
    min_weekly_hours=35.0,
    max_weekly_hours=48.0,
    min_shift_length=4.0,
    max_shift_length=11.0,
    min_days_off=1.0,
    auto_shuttle=False,
    add_handover_buffer=False,
    apply_peak_cutting=False
):
    """
    Solve shift optimization for one or multiple occupations.

    workloads: list of dicts [{"csv_path": ..., "name": ...}, ...]
               If None, falls back to single csv_path for backward compatibility.
    """
    # ============ BACKWARD COMPATIBILITY ============
    if workloads is None:
        workloads = [{"csv_path": csv_path, "name": "Staff"}]

    workloads = [w for w in workloads if w.get("csv_path") and os.path.exists(w["csv_path"])]
    if not workloads:
        print("Error: No valid workload files found.")
        return None

    num_occupations = len(workloads)
    print(f"Optimizing for {num_occupations} occupation(s): {', '.join(w['name'] for w in workloads)}")

    # ============ LOAD & PREPROCESS ALL WORKLOADS ============
    occupation_data = []
    base_df = None

    for wl in workloads:
        df = pd.read_csv(wl["csv_path"])
        df, final_required = _preprocess_workload(df, apply_peak_cutting, add_handover_buffer)
        occ_name = wl["name"]
        if base_df is None:
            base_df = df.copy()
        occupation_data.append({
            "name": occ_name,
            "prefix": occ_name.replace(" ", ""),
            "df": df,
            "final_required": final_required,
        })

    num_intervals = len(base_df)
    times_list = base_df["time"].tolist()

    # ============ SHUTTLE WINDOWS (SHARED) ============
    if auto_shuttle:
        shuttle_windows = _detect_smart_shuttle_windows(occupation_data)
        print(f"Smart auto-detected {len(shuttle_windows)} shuttle windows: {shuttle_windows}")
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
    templates_df = None
    if os.path.exists(templates_path):
        templates_df = pd.read_csv(templates_path)
        print(f"Loading {len(templates_df)} pre-assigned shift templates from {templates_path}")
        use_templates = True

    # ============ GENERATE SHIFTS PER OCCUPATION ============
    all_occ_shifts = {}
    for occ in occupation_data:
        occ_shifts = []
        for day_idx in range(7):
            day_shifts = _generate_shifts_for_occupation(
                day_idx, shuttle_windows, occ["final_required"], num_intervals,
                min_shift_length, max_shift_length, occ["prefix"],
                use_templates, templates_df
            )
            occ_shifts.extend(day_shifts)
        all_occ_shifts[occ["name"]] = occ_shifts
        print(f"  {occ['name']}: Generated {len(occ_shifts)} candidate shifts")

    # ============ BUILD JOINT MIP ============
    prob = pulp.LpProblem("Multi_Occupation_Shift_Optimization", pulp.LpMinimize)

    occ_shift_vars = {}
    for occ_name, shifts in all_occ_shifts.items():
        occ_shift_vars[occ_name] = pulp.LpVariable.dicts(
            f"Shifts_{occ_name}", [s["id"] for s in shifts], lowBound=0, cat='Integer'
        )

    occ_covers = {}
    occ_starts_at = {}
    occ_ends_at = {}
    for occ_name, shifts in all_occ_shifts.items():
        covers = {t: [] for t in range(num_intervals)}
        starts_at = {t: [] for t in range(num_intervals)}
        ends_at = {t: [] for t in range(num_intervals)}
        for s in shifts:
            starts_at[s["global_start_idx"]].append(s["id"])
            ends_at[s["global_end_idx"]].append(s["id"])
            indices = np.where(s["coverage"] == 1)[0]
            for idx in indices:
                covers[idx].append(s["id"])
        occ_covers[occ_name] = covers
        occ_starts_at[occ_name] = starts_at
        occ_ends_at[occ_name] = ends_at

    # Shared shuttle variables
    shuttle_windows_indices = []
    for t in range(num_intervals):
        if times_list[t] in shuttle_windows:
            shuttle_windows_indices.append(t)

    shuttle_in_vars = pulp.LpVariable.dicts("ShuttleIn", shuttle_windows_indices, lowBound=0, cat='Integer')
    shuttle_out_vars = pulp.LpVariable.dicts("ShuttleOut", shuttle_windows_indices, lowBound=0, cat='Integer')

    # ============ OVER-COVERAGE PENALTY VARIABLES ============
    # For each interval with demand, add slack variables measuring excess.
    # Use tiered penalties (piecewise-linear approximation of quadratic)
    # so small excess is tolerated but large excess is punished severely.
    overcov_vars = {}       # tier 1: any excess
    overcov_vars_t2 = {}    # tier 2: excess > 2
    overcov_vars_t3 = {}    # tier 3: excess > 4
    max_demand = max(max(occ["final_required"]) for occ in occupation_data)

    for occ in occupation_data:
        occ_name = occ["name"]
        covers = occ_covers[occ_name]
        final_req = occ["final_required"]
        sv = occ_shift_vars[occ_name]
        for t in range(num_intervals):
            if final_req[t] > 0 and covers[t]:
                cov_expr = pulp.lpSum([sv[sid] for sid in covers[t]])
                # Tier 1: any excess above required
                ov1 = pulp.LpVariable(f"overcov1_{occ_name}_{t}", lowBound=0, cat='Continuous')
                overcov_vars[(occ_name, t)] = ov1
                prob += ov1 >= cov_expr - final_req[t]
                # Tier 2: excess above required + 2
                ov2 = pulp.LpVariable(f"overcov2_{occ_name}_{t}", lowBound=0, cat='Continuous')
                overcov_vars_t2[(occ_name, t)] = ov2
                prob += ov2 >= cov_expr - final_req[t] - 2
                # Tier 3: excess above required + 4
                ov3 = pulp.LpVariable(f"overcov3_{occ_name}_{t}", lowBound=0, cat='Continuous')
                overcov_vars_t3[(occ_name, t)] = ov3
                prob += ov3 >= cov_expr - final_req[t] - 4

    print(f"  Tiered over-coverage penalty: {len(overcov_vars)} intervals tracked")

    # ============ OBJECTIVE ============
    # 1. Primary: minimize total shifts (headcount)
    # 2. Over-coverage penalty: strongly penalize excess staff at each interval
    # 3. Waste penalty: prefer shifts that cover actual demand, not dead time
    # 4. Shuttle penalty: consolidate shuttle events
    obj_terms = []

    # (1) Total shifts — main objective
    for occ_name, shifts in all_occ_shifts.items():
        sv = occ_shift_vars[occ_name]
        obj_terms.append(pulp.lpSum([sv[s["id"]] for s in shifts]))

    # (2) Over-coverage penalty — tiered (piecewise-linear quadratic approx)
    # Tier 1: moderate penalty for any excess (0..2 above required)
    # Tier 2: heavier penalty for excess > 2
    # Tier 3: very heavy penalty for excess > 4
    # Weight per interval-unit must be high enough to make the solver
    # prefer splitting one long shift into two shorter ones when it reduces
    # total excess across all intervals.
    num_active_intervals = len(overcov_vars)
    if num_active_intervals > 0:
        # Each unit of overcov across all intervals should cost MORE than
        # the 1 shift it would take to split a long shift into short ones.
        # With ~1200 active intervals, weight=1.0 per unit means avg excess of 1
        # costs 1200 — much more than adding a shift (cost=1). This strongly
        # pushes the solver to minimize excess even at the cost of more shifts.
        base_weight = 1.0
        obj_terms.append(base_weight * pulp.lpSum(list(overcov_vars.values())))
        obj_terms.append(base_weight * 3.0 * pulp.lpSum(list(overcov_vars_t2.values())))
        obj_terms.append(base_weight * 6.0 * pulp.lpSum(list(overcov_vars_t3.values())))

    # (3) Waste penalty — prefer shifts whose coverage matches workload shape
    for occ_name, shifts in all_occ_shifts.items():
        sv = occ_shift_vars[occ_name]
        waste_terms = []
        for s in shifts:
            wp = s.get("waste_penalty", 0)
            if wp > 0:
                waste_terms.append(wp * 0.05 * sv[s["id"]])
        if waste_terms:
            obj_terms.append(pulp.lpSum(waste_terms))

    # (4) Shuttle events penalty — discourage many distinct shuttle windows
    # Use binary indicators: is there ANY shuttle activity at this window?
    shuttle_active_vars = {}
    for t in shuttle_windows_indices:
        sa = pulp.LpVariable(f"shuttle_active_{t}", cat='Binary')
        shuttle_active_vars[t] = sa
        # Big-M: if any shuttle at t, sa=1
        big_M = 500
        prob += shuttle_in_vars[t] + shuttle_out_vars[t] <= big_M * sa

    # Penalize number of active shuttle windows (consolidation)
    # Plus smaller penalty on total shuttle count
    obj_terms.append(1.0 * pulp.lpSum(list(shuttle_active_vars.values())))
    obj_terms.append(0.05 * pulp.lpSum([shuttle_in_vars[t] + shuttle_out_vars[t] for t in shuttle_windows_indices]))

    prob += pulp.lpSum(obj_terms)

    # ============ PER-OCCUPATION WORKLOAD CONSTRAINTS ============
    print("Adding workload constraints per occupation...")
    for occ in occupation_data:
        occ_name = occ["name"]
        sv = occ_shift_vars[occ_name]
        covers = occ_covers[occ_name]
        final_req = occ["final_required"]

        for t in range(num_intervals):
            if covers[t]:
                prob += pulp.lpSum([sv[sid] for sid in covers[t]]) >= final_req[t]
            elif final_req[t] > 0:
                print(f"  Warning: {occ_name} interval {t} ({times_list[t]}) needs coverage but has no shifts")

    # ============ SHARED SHUTTLE CONSTRAINTS ============
    print("Adding shared shuttle constraints across all occupations...")
    for t in shuttle_windows_indices:
        total_starts = []
        total_ends = []
        for occ_name in all_occ_shifts:
            sv = occ_shift_vars[occ_name]
            if occ_starts_at[occ_name][t]:
                total_starts.extend([sv[sid] for sid in occ_starts_at[occ_name][t]])
            if occ_ends_at[occ_name][t]:
                total_ends.extend([sv[sid] for sid in occ_ends_at[occ_name][t]])

        if total_starts:
            prob += shuttle_in_vars[t] >= pulp.lpSum(total_starts) / shuttle_capacity
        if total_ends:
            prob += shuttle_out_vars[t] >= pulp.lpSum(total_ends) / shuttle_capacity

    # FTE constraint (combined)
    total_hours_expr = pulp.lpSum([
        occ_shift_vars[occ_name][s["id"]] * s["duration"]
        for occ_name, shifts in all_occ_shifts.items()
        for s in shifts
    ])
    if max_fte is not None:
        prob += (total_hours_expr / 45.0) <= max_fte

    # ============ SOLVE ============
    timeout = 60 * num_occupations
    print(f"Solving joint optimization for {num_occupations} occupation(s) (timeout: {timeout}s)...")
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=timeout, gapRel=0.01))

    if pulp.LpStatus[prob.status] != 'Optimal':
        print(f"Warning: Solution status = {pulp.LpStatus[prob.status]}")
        return None

    # ============ EXTRACT RESULTS PER OCCUPATION ============
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    all_roster_rows = []
    all_assigned_shifts_types = []
    occ_coverages = {}
    occ_stats = {}
    grand_total_hours = 0
    grand_total_headcount = 0
    grand_people_below_min = 0

    for occ in occupation_data:
        occ_name = occ["name"]
        occ_prefix = occ["prefix"]
        sv = occ_shift_vars[occ_name]
        shifts = all_occ_shifts[occ_name]

        total_coverage = np.zeros(num_intervals)
        total_hours_val = 0
        occ_assigned_shifts = []

        for s in shifts:
            val = pulp.value(sv[s["id"]])
            if val and val > 0:
                count = int(val)
                all_assigned_shifts_types.append({
                    "Occupation": occ_name,
                    "Day": s["day_name"], "Shift": s["display"],
                    "Duration": s["duration"], "Count": count
                })
                total_coverage += val * s["coverage"]
                total_hours_val += val * s["duration"]

                h_start, m_start = map(int, s['start_time'].split(":"))
                start_idx = s['day_idx'] * 288 + (h_start * 12 + m_start // 5)
                duration_intervals = int(s['duration'] * 12)
                end_idx_absolute = start_idx + duration_intervals

                for _ in range(count):
                    occ_assigned_shifts.append({
                        "day_idx": s["day_name"],
                        "start": start_idx,
                        "end": end_idx_absolute,
                        "display": s["display"],
                        "duration": s["duration"],
                        "raw_start": s["start_time"],
                        "raw_end": s["end_time"]
                    })

        occ_assigned_shifts.sort(key=lambda x: (x["start"], -x["duration"]))

        roster_rows, personnel_pool, people_below_min = _assign_personnel(
            occ_assigned_shifts, max_weekly_hours, min_weekly_hours, min_days_off, occ_prefix
        )

        for row in roster_rows:
            row["Occupation"] = occ_name
        all_roster_rows.extend(roster_rows)

        occ_coverages[occ_name] = total_coverage
        occ_headcount = len(personnel_pool)
        grand_total_hours += total_hours_val
        grand_total_headcount += occ_headcount
        grand_people_below_min += people_below_min

        occ_stats[occ_name] = {
            "Total Hours": total_hours_val,
            "FTE": round(total_hours_val / 45.0, 2),
            "Headcount": occ_headcount,
            "People Below Min Hours": people_below_min,
        }
        print(f"  {occ_name}: {occ_headcount} workers, FTE {total_hours_val / 45.0:.2f}, {people_below_min} below min hours")

    # ============ SAVE OUTPUTS ============
    pd.DataFrame(all_roster_rows).to_csv("personnel_roster_weekly.csv", index=False)

    coverage_df = base_df.copy()
    total_cov = np.zeros(num_intervals)
    for occ_name, cov in occ_coverages.items():
        coverage_df[f"coverage_{occ_name}"] = cov
        total_cov += cov
    coverage_df["actual_coverage"] = total_cov

    total_required = np.zeros(num_intervals)
    for occ in occupation_data:
        coverage_df[f"required_{occ['name']}"] = occ["final_required"]
        total_required += np.array(occ["final_required"])
    coverage_df["required_headcount"] = total_required.astype(int)
    coverage_df.to_csv("weekly_coverage_comparison.csv", index=False)

    arrivals = np.zeros(num_intervals)
    departures = np.zeros(num_intervals)
    for r in all_roster_rows:
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
                "Day": base_df.iloc[t]["day_name"], "Time": base_df.iloc[t]["time"],
                "Arrivals": int(arrivals[t]), "Departures": int(departures[t]),
                "Shuttles_In": arr_sh, "Shuttles_Out": dep_sh, "Total_Shuttles": arr_sh + dep_sh
            })
    pd.DataFrame(shuttle_data).to_csv("shuttle_report_weekly.csv", index=False)

    total_shuttles = sum(x['Total_Shuttles'] for x in shuttle_data)
    fte_val = grand_total_hours / 45.0
    min_hc_theoretical = math.ceil(grand_total_hours / max_weekly_hours) if max_weekly_hours > 0 else 0
    max_hc_theoretical = math.floor(grand_total_hours / min_weekly_hours) if min_weekly_hours > 0 else grand_total_headcount

    summary_row = {
        "Total Hours": grand_total_hours,
        "FTE": round(fte_val, 2),
        "Headcount": grand_total_headcount,
        "Theoretical Min Headcount": min_hc_theoretical,
        "Theoretical Max Headcount": max_hc_theoretical,
        "People Below Min Hours": grand_people_below_min,
        "Min Weekly Hours": min_weekly_hours,
        "Max Weekly Hours": max_weekly_hours,
        "Min Days Off": min_days_off,
        "Total Weekly Shuttles": total_shuttles,
        "Max FTE Allowed": max_fte,
        "Max Headcount Allowed": max_headcount if max_headcount else "Unlimited",
        "Num Occupations": num_occupations,
    }
    for occ_name, stats in occ_stats.items():
        for k, v in stats.items():
            summary_row[f"{occ_name} - {k}"] = v
    pd.DataFrame([summary_row]).to_csv("weekly_summary.csv", index=False)

    day_order = {day: i for i, day in enumerate(days)}
    results_df = pd.DataFrame(all_assigned_shifts_types)
    if not results_df.empty:
        results_df['day_id'] = results_df['Day'].map(day_order)
        results_df.sort_values(['Occupation', 'day_id', 'Shift']).drop('day_id', axis=1).to_csv("assigned_shifts_weekly.csv", index=False)

    print(f"\nOptimization complete. Total FTE: {fte_val:.2f}, Total Headcount: {grand_total_headcount}, Shuttles: {total_shuttles}")
    print("Individual roster saved to personnel_roster_weekly.csv")
    return coverage_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Shift Optimizer')
    parser.add_argument('max_fte', type=float, nargs='?', default=None)
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--capacity', type=int, default=16)
    args = parser.parse_args()
    solve_weekly_shift_optimization(max_fte=args.max_fte, shuttle_interval=args.interval, sparse_mode=args.sparse, shuttle_capacity=args.capacity)
