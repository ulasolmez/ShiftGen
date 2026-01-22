import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_workload(filename="workload_weekly.csv", randomized=True):
    all_data = []
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # Base parameters for peaks - randomized if requested
    morning_peak_time = np.random.uniform(8, 11) if randomized else 10
    afternoon_peak_time = np.random.uniform(14, 18) if randomized else 16
    morning_intensity = np.random.uniform(10, 25) if randomized else 15
    afternoon_intensity = np.random.uniform(15, 35) if randomized else 20
    base_floor = np.random.uniform(2, 8) if randomized else 5

    for day_idx, day_name in enumerate(days):
        start_time = datetime(2026, 1, 19 + day_idx, 0, 0)
        times = [start_time + timedelta(minutes=5 * i) for i in range(288)]
        
        x = np.linspace(0, 24, 288)
        
        # Create a randomized workload curve for this specific day
        d_morning = morning_peak_time + np.random.uniform(-0.5, 0.5)
        d_afternoon = afternoon_peak_time + np.random.uniform(-0.5, 0.5)
        
        # Smooth curve based on Gaussian peaks
        workload = (
            base_floor + 
            5 * np.sin(np.pi * x / 12)**2 + 
            morning_intensity * np.exp(-(x - d_morning)**2 / 6) + 
            afternoon_intensity * np.exp(-(x - d_afternoon)**2 / 8)
        )
        
        # Add a very small amount of smooth noise (low frequency)
        smooth_noise = np.convolve(np.random.normal(0, 1, 288), np.ones(12)/12, mode='same')
        workload = np.maximum(workload + smooth_noise, 0).astype(int)
        
        for i in range(288):
            all_data.append({
                "day_idx": day_idx,
                "day_name": day_name,
                "time": times[i].strftime("%H:%M"),
                "required_headcount": workload[i]
            })
    
    df = pd.DataFrame(all_data)
    df.to_csv(filename, index=False)
    print(f"Randomized weekly workload saved to {filename}")

if __name__ == "__main__":
    generate_sample_workload(randomized=True)

if __name__ == "__main__":
    generate_sample_workload()
