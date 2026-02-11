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
                for i in range(duration_intervals):
                    check_idx = (global_start_idx + i) % num_intervals
                    if final_required[check_idx] == 0:
                        zero_count += 1
                if zero_count > (duration_intervals * 0.5):
                    continue

                coverage = np.zeros(num_intervals, dtype=int)
                for i in range(duration_intervals):
                    coverage[(global_start_idx + i) % num_intervals] = 1
                shift_display = f"{format_as_hhmm(start_time_str)}-{format_as_hhmm(end_time_str)}"
                shift_id = f"{occ_prefix}_Day{day_idx}_{format_as_hhmm(start_time_str)}_{duration_hours}h"
                shifts.append({
                    "id": shift_id, "day_idx": day_idx, "day_name": days[day_idx],
                    "start_time": start_time_str, "end_time": end_time_str,
                    "duration": duration_hours, "coverage": coverage, "display": shift_display,
                    "global_start_idx": global_start_idx,
                    "global_end_idx": (global_start_idx + duration_intervals) % num_intervals
                })
    return shifts


def _assign_personnel(all_assigned_shifts, max_weekly_hours, min_weekly_hours, min_days_off, occ_prefix):
    """Greedy bin-packing assignment for one occupation."""
    REST_INTERVALS = 144
    total_shift_hours = sum(s['duration'] for s in all_assigned_shifts)
    theoretical_min_people = max(1, math.ceil(total_shift_hours / max_weekly_hours)) if max_weekly_hours > 0 else 1

    personnel_pool = []
    for i in range(theoretical_min_people):
        personnel_pool.append({
            'id': i + 1,
            'end_time': -REST_INTERVALS - 1,
            'weekly_hours': 0.0,
            'days_worked': set(),
            'shift_times': []
        })

    roster_rows = []

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

    for shift in all_assigned_shifts:
        assigned = False
        shift_day_idx = shift['start'] // 288
        max_working_days = 7 - math.ceil(min_days_off)

        personnel_pool.sort(key=lambda x: (
            len(x['days_worked']) >= max_working_days,
            x['weekly_hours'] >= min_weekly_hours,
            -1 * (min_weekly_hours - x['weekly_hours']) if x['weekly_hours'] < min_weekly_hours else 0,
            x['end_time'],
            -1 * (max_weekly_hours - x['weekly_hours'])
        ))

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

        if not assigned:
            can_assign_to_existing = False
            for p in personnel_pool:
                if shift["start"] >= p['end_time'] + REST_INTERVALS:
                    if p['weekly_hours'] + shift['duration'] <= max_weekly_hours:
                        if not violates_off_day_policy(p, shift_day_idx, shift["start"], shift["end"]):
                            can_assign_to_existing = True
                            break

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
            "Personnel_ID": f"{occ_prefix}_{p_id:03d}",
            "Day": shift["day_idx"],
            "Shift": shift["display"],
            "Start": shift["raw_start"],
            "End": shift["raw_end"],
            "duration_intervals": int(shift["duration"] * 12),
            "start_idx": shift["start"]
        })

    people_below_min = sum(1 for p in personnel_pool if p['weekly_hours'] < min_weekly_hours)
    return roster_rows, personnel_pool, people_below_min


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

    # ============ OBJECTIVE ============
    obj_terms = []
    for occ_name, shifts in all_occ_shifts.items():
        sv = occ_shift_vars[occ_name]
        obj_terms.append(pulp.lpSum([sv[s["id"]] for s in shifts]))
    obj_terms.append(0.1 * pulp.lpSum([shuttle_in_vars[t] + shuttle_out_vars[t] for t in shuttle_windows_indices]))
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
    timeout = 30 * num_occupations
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
