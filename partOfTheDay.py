import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("PRICE_AND_DEMAND_2010_2024_NSW1.csv")  # Change the filename if needed

# Check if 'Part_of_Day' column exists
if 'Part_of_Day' not in df.columns:
    raise ValueError("The 'Part_of_Day' column is missing in the dataset!")

# Group by 'Part_of_Day' and calculate mean consumption
part_of_day_consumption = df.groupby('Part_of_Day')['TOTALDEMAND'].mean().sort_values(ascending=False)

# Print the sorted energy consumption
print("Average energy consumption by part of the day:")
print(part_of_day_consumption)

# Plotting
plt.figure(figsize=(10,6))
bars = plt.bar(part_of_day_consumption.index, part_of_day_consumption.values, color='royalblue', width=0.5)

# Add values on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f'{yval:.0f}', ha='center', va='bottom', fontsize=10)

plt.title('Average Energy Consumption by Part of the Day', fontsize=16)
plt.xlabel('Part of the Day', fontsize=14)
plt.ylabel('Average Energy Consumption (MW)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
