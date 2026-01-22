import pandas as pd
import matplotlib.pyplot as plt
import os

def create_shuttle_report(shuttle_csv="shuttle_report_weekly.csv"):
    if not os.path.exists(shuttle_csv):
        print(f"Error: {shuttle_csv} not found. Run optimizer first.")
        return

    df = pd.read_csv(shuttle_csv)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # 1. Daily Total Shuttles Bar Chart
    daily_totals = df.groupby('Day')['Total_Shuttles'].sum().reindex(days)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(daily_totals.index, daily_totals.values, color='teal', alpha=0.7)
    plt.title('Total Shuttle Trips Required Per Day', fontsize=14)
    plt.ylabel('Number of Shuttle Trips')
    plt.xlabel('Day of the Week')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom', fontweight='bold')
    
    plt.savefig('shuttle_daily_totals.png')
    print("Daily totals bar graph saved to shuttle_daily_totals.png")
    
    # 2. Hourly Breakdown (Aggregated across the week for a general view)
    # Or maybe just a multi-bar chart per day if it's not too crowded
    plt.figure(figsize=(15, 8))
    
    # Pivot data for hourly view
    # Some times might be missing if no shuttles, so we fill with 0
    hourly_data = df.pivot_table(index='Time', columns='Day', values='Total_Shuttles', aggfunc='sum').fillna(0)
    hourly_data = hourly_data.reindex(columns=days)
    
    hourly_data.plot(kind='bar', stacked=True, figsize=(15, 7), colormap='viridis')
    plt.title('Hourly Shuttle Distribution Throughout the Week', fontsize=14)
    plt.ylabel('Total Shuttles (In + Out)')
    plt.xlabel('Time of Day')
    plt.xticks(rotation=45)
    plt.legend(title='Day')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('shuttle_hourly_distribution.png')
    print("Hourly distribution bar graph saved to shuttle_hourly_distribution.png")

if __name__ == "__main__":
    create_shuttle_report()
