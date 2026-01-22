import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_weekly_results(coverage_csv="weekly_coverage_comparison.csv", shifts_csv="assigned_shifts_weekly.csv", shuttle_csv="shuttle_report_weekly.csv"):
    df_cov = pd.read_csv(coverage_csv)
    df_shifts = pd.read_csv(shifts_csv)
    
    # Try to load shuttle data
    try:
        df_shuttle = pd.read_csv(shuttle_csv)
    except:
        df_shuttle = pd.DataFrame()
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    if not os.path.exists("graphs"):
        os.makedirs("graphs")
        
    for day_name in days:
        day_data = df_cov[df_cov['day_name'] == day_name]
        day_shifts = df_shifts[df_shifts['Day'] == day_name]
        day_shuttle = df_shuttle[df_shuttle['Day'] == day_name] if not df_shuttle.empty else pd.DataFrame()
        
        fig, (ax, ax_text) = plt.subplots(2, 1, figsize=(15, 14), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot 1: Curves
        ax.plot(day_data['time'], day_data['required_headcount'], label='Required', color='blue', linewidth=2)
        ax.plot(day_data['time'], day_data['actual_coverage'], label='Optimized (Shuttle Aligned)', color='red', linestyle='--', linewidth=2)
        ax.fill_between(day_data['time'], day_data['actual_coverage'], alpha=0.2, color='red')
        
        ax.set_title(f'Personnel & Shuttle Optimization: {day_name}')
        ax.set_xlabel('Time of Day')
        ax.set_ylabel('Headcount')
        ax.legend()
        ax.set_xticks(day_data['time'][::12])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Plot 2: Text summaries
        ax_text.axis('off')
        
        # Header Info
        try:
            summary = pd.read_csv("weekly_summary.csv")
            fte_total = summary.iloc[0]["FTE"]
            shuttle_total = summary.iloc[0]["Total Weekly Shuttles"]
            header = f"Weekly Stats: {fte_total:.2f} FTE | {int(shuttle_total)} Total Shuttles\n"
        except:
            header = ""

        # Shift List
        shift_header = f"\n{day_name} SHIFTS (Personnel Entering/Leaving at Shuttle Hours):\n"
        shift_lines = []
        for _, row in day_shifts.iterrows():
            shift_lines.append(f"{row['Count']} pax: {row['Shift']}")
            
        # Shuttle List for Day
        shuttle_header = f"\n\n{day_name} SHUTTLE REQUIREMENTS (16 pax capacity):\n"
        shuttle_lines = []
        if not day_shuttle.empty:
            for _, row in day_shuttle.iterrows():
                shuttle_lines.append(f"{row['Time']}: {row['Total_Shuttles']} Shuttles ({row['Arrivals']} in, {row['Departures']} out)")

        # Combine and split into columns
        full_text = header + shift_header + "\n".join(shift_lines) + shuttle_header + "\n".join(shuttle_lines)
        
        # Simple column logic
        all_lines = (shift_header + "\n".join(shift_lines) + shuttle_header + "\n".join(shuttle_lines)).split("\n")
        chunk_size = 14
        columns = [all_lines[i:i + chunk_size] for i in range(0, len(all_lines), chunk_size)]
        
        ax_text.text(0, 1.1, header, transform=ax_text.transAxes, fontsize=12, fontweight='bold')
        
        curr_x = 0
        for col in columns:
            ax_text.text(curr_x, 0.95, "\n".join(col), transform=ax_text.transAxes, 
                        verticalalignment='top', fontsize=9, fontfamily='monospace')
            curr_x += 0.33
        
        plt.tight_layout()
        plt.savefig(f"graphs/optimization_{day_name}.png")
        plt.close()
        print(f"Graph for {day_name} saved with Shuttle details.")

if __name__ == "__main__":
    plot_weekly_results()
