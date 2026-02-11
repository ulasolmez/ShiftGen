from optimizer import solve_weekly_shift_optimization
import numpy as np
import pandas as pd

# Create test CSVs with realistic smooth workload (dual-peak)
days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
times = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 5)]
np.random.seed(42)

for idx in range(2):
    rows = []
    for day in days:
        for t_str in times:
            h, m = map(int, t_str.split(":"))
            hour_frac = h + m / 60.0
            if 6 <= hour_frac <= 20:
                morning = 15 * np.exp(-0.5 * ((hour_frac - 9) / 2) ** 2)
                afternoon = 12 * np.exp(-0.5 * ((hour_frac - 15) / 2) ** 2)
                base = morning + afternoon
                if idx == 0:
                    val = int(base * 1.2) + np.random.randint(-1, 2)
                else:
                    val = int(base * 0.5) + np.random.randint(-1, 2)
                val = max(0, val)
            else:
                val = 0
            rows.append({"day": day, "time": t_str, "required_headcount": val})
    df = pd.DataFrame(rows)
    df.to_csv(f"test_occ{idx+1}.csv", index=False)

workloads = [
    {"csv_path": "test_occ1.csv", "name": "Occ1"},
    {"csv_path": "test_occ2.csv", "name": "Occ2"},
]

result = solve_weekly_shift_optimization(
    workloads=workloads,
    shuttle_capacity=20,
    min_shift_length=3,
    max_shift_length=12,
    min_weekly_hours=35,
    max_weekly_hours=48,
    min_days_off=1,
    auto_shuttle=True,
)

if result is not None and not result.empty:
    print("\n=== OVERCOVERAGE ANALYSIS ===")
    for occ_idx in range(2):
        occ_name = f"Occ{occ_idx+1}"
        cov_col = f"coverage_{occ_name}"
        req_col = f"required_{occ_name}"
        if cov_col in result.columns and req_col in result.columns:
            cov = result[cov_col].values
            req = result[req_col].values
            active = req > 0
            excess = cov[active] - req[active]
            print(f"  {occ_name}: mean_excess={excess.mean():.2f}, max_excess={excess.max():.0f}, exact={np.sum(excess==0)/len(excess)*100:.1f}%, within_2={np.sum(excess<=2)/len(excess)*100:.1f}%")
    cov_total = result["actual_coverage"].values
    req_total = result["required_headcount"].values
    active = req_total > 0
    excess = cov_total[active] - req_total[active]
    print(f"  TOTAL: mean_excess={excess.mean():.2f}, max_excess={excess.max():.0f}, exact={np.sum(excess==0)/len(excess)*100:.1f}%, within_2={np.sum(excess<=2)/len(excess)*100:.1f}%")
    
    # Check shuttle count
    shuttle_df = pd.read_csv("shuttle_report_weekly.csv")
    total_shuttles = shuttle_df["Total_Shuttles"].sum()
    unique_times = shuttle_df.shape[0]
    print(f"\n=== SHUTTLE ANALYSIS ===")
    print(f"  Total shuttle trips: {total_shuttles}")
    print(f"  Unique shuttle time-slots used: {unique_times}")
    
    # Check Sunday workers
    roster = pd.read_csv("personnel_roster_weekly.csv")
    print(f"\n=== SUNDAY CHECK ===")
    all_personnel = roster["Personnel_ID"].unique()
    print(f"  Total unique workers: {len(all_personnel)}")
else:
    print('INFEASIBLE or no result')

# Clean up
import os
for f in ["test_occ1.csv", "test_occ2.csv"]:
    if os.path.exists(f):
        os.remove(f)
