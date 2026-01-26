import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_workload(filename="workload_weekly.csv", randomized=True):
    all_data = []
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    for day_idx, day_name in enumerate(days):
        # Day-specific base intensity
        is_weekend = day_name in ["Saturday", "Sunday"]
        day_mult = 0.6 if is_weekend else 1.0
        
        # Base parameters for peaks
        morning_peak_time = np.random.uniform(7, 10)
        afternoon_peak_time = np.random.uniform(15, 18)
        morning_intensity = np.random.uniform(15, 30) * day_mult
        afternoon_intensity = np.random.uniform(20, 45) * day_mult
        base_floor = np.random.uniform(2, 6) * day_mult

        start_time = datetime(2026, 1, 19 + day_idx, 0, 0)
        times = [start_time + timedelta(minutes=5 * i) for i in range(288)]
        x = np.linspace(0, 24, 288)
        
        # 1. Smooth foundational peaks
        workload = (
            base_floor + 
            morning_intensity * np.exp(-(x - morning_peak_time)**2 / 4) + 
            afternoon_intensity * np.exp(-(x - afternoon_peak_time)**2 / 6)
        )
        
        # 2. Add Sudden "Event" Spikes (Very narrow peaks)
        num_spikes = np.random.randint(2, 6)
        for _ in range(num_spikes):
            spike_time = np.random.uniform(6, 22)
            spike_height = np.random.uniform(10, 25)
            workload += spike_height * np.exp(-(x - spike_time)**2 / 0.05)
            
        # 3. Add Sharp Dips (Negative spikes)
        num_dips = np.random.randint(1, 4)
        for _ in range(num_dips):
            dip_time = np.random.uniform(9, 17)
            dip_depth = np.random.uniform(5, 15)
            workload -= dip_depth * np.exp(-(x - dip_time)**2 / 0.1)

        # 4. Mandatory Zero/Low periods (Night time: 00:00 - 05:30)
        # We apply a transition curve to zero out the night
        night_mask = np.ones(288)
        for i, t in enumerate(x):
            if t < 5.0 or t > 23.5:
                night_mask[i] = 0
            elif 5.0 <= t <= 6.5: # Ramp up
                night_mask[i] = (t - 5.0) / 1.5
            elif 22.5 <= t <= 23.5: # Ramp down
                night_mask[i] = (23.5 - t) / 1.0
        
        workload *= night_mask

        # 5. Add "Chunky" Randomness (Simulating groups of passengers/tasks)
        # Instead of smooth noise, we add random integers in 30-min blocks
        for i in range(0, 288, 6): # Every 30 mins
            chunk_noise = np.random.choice([-3, -2, 0, 2, 5, 8], p=[0.1, 0.1, 0.4, 0.2, 0.15, 0.05])
            workload[i:i+6] += chunk_noise

        # Final cleanup: Ensure non-negative and round to integers
        workload = np.maximum(workload, 0).astype(int)
        
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
