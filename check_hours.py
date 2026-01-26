import pandas as pd
from datetime import datetime

roster = pd.read_csv('personnel_roster_weekly.csv')

worker_hours = {}
for _, row in roster.iterrows():
    worker_id = row['Personnel_ID']
    if worker_id not in worker_hours:
        worker_hours[worker_id] = 0
    
    start = datetime.strptime(row['Start'], '%H:%M')
    end = datetime.strptime(row['End'], '%H:%M')
    
    if end < start:
        from datetime import timedelta
        end = end + timedelta(days=1)
    
    duration = (end - start).total_seconds() / 3600
    worker_hours[worker_id] += duration

violations = [(w, h) for w, h in worker_hours.items() if h > 48.1]
print(f'Workers with >48h: {len(violations)}')
if violations:
    print('\nViolations:')
    for w, h in sorted(violations, key=lambda x: -x[1]):
        print(f'  {w}: {h:.1f}h')

print('\nTop 10 highest hours:')
for w, h in sorted(worker_hours.items(), key=lambda x: -x[1])[:10]:
    print(f'  {w}: {h:.1f}h')

print(f'\nTotal workers: {len(worker_hours)}')
print(f'Average hours: {sum(worker_hours.values()) / len(worker_hours):.1f}h')
print(f'Min hours: {min(worker_hours.values()):.1f}h')
print(f'Max hours: {max(worker_hours.values()):.1f}h')
