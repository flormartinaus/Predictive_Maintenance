#Import Dependecies 

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sys


#read csv file

df = pd.read_csv('C:\\Users\\FloMartin-ResolveMin\\Desktop\\git\\Predictive_maintenance\\predictive_maintenance_dataset.csv')

print(df)

# check for missing values

missing_data = df.isnull().sum()

# remove duplicates columns

df = df.loc[:, ~df.columns.duplicated()]


# check the numbers of entries and unique devices

num_entries = df.shape[0]
num_unique_devices = df['device'].nunique()

# print summary statistics

print(df.describe())

#check the class imbalance

class_counts = df['failure'].value_counts()

# data visualisation
# visualise the total number of failures per month

df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month

monthly_fail = df.groupby('month')['failure'].sum().reset_index()

plt.figure(figsize=(20, 12))
sns.barplot(x='month', y='failure', data=monthly_fail, color='midnightblue')
plt.title('Aggregated device failures by month')
plt.xlabel('Month')
plt.ylabel('Total number of failures')
plt.ylim(0, 100)
plt.xticks(rotation=0)
plt.savefig('device_failures_by_month.png')
plt.show()

# Check devices with multiple failures
device_failure_counts = df.groupby('device')['failure'].sum()
devices_with_multiple_failures = device_failure_counts[device_failure_counts >= 2]
num_devices_with_multiple_failures = len(devices_with_multiple_failures)

print(f"Number of devices with multiple failures: {num_devices_with_multiple_failures}")






