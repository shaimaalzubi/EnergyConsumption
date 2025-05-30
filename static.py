
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# قراءة البيانات
df = pd.read_csv("PRICE_AND_DEMAND_2010_2024_NSW1.csv")

# تحويل التاريخ
df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
df['Month'] = df['SETTLEMENTDATE'].dt.month
df['Year'] = df['SETTLEMENTDATE'].dt.year
df['Hour'] = df['SETTLEMENTDATE'].dt.hour

# 1. نظرة سريعة
print(df.info())
print(df.describe())

# 2. مصفوفة الارتباط
plt.figure(figsize=(8, 6))
sns.heatmap(df[['TOTALDEMAND', 'TEMP', 'RH2M', 'HEAT_INDEX']].corr(), annot=True, cmap='coolwarm')
plt.title("Heatmap - Correlation")
plt.show()

# 3. Scatter plot بين HEAT_INDEX و TOTALDEMAND
plt.figure(figsize=(8, 5))
sns.scatterplot(x='HEAT_INDEX', y='TOTALDEMAND', data=df, alpha=0.3)
plt.title("HEAT_INDEX vs TOTALDEMAND")
plt.xlabel("HEAT INDEX (°C)")
plt.ylabel("Total Electricity Demand (MW)")
plt.show()

# 4. Boxplot شهري للاستهلاك
plt.figure(figsize=(10, 6))
sns.boxplot(x='Month', y='TOTALDEMAND', data=df)
plt.title("Monthly Electricity Demand Distribution")
plt.xlabel("Month")
plt.ylabel("Total Demand (MW)")
plt.show()
# تأكد أن التاريخ بصيغة datetime
df['DATE'] = df['SETTLEMENTDATE'].dt.date

# تجميع حسب كل يوم (متوسط)
daily_df = df.groupby('SETTLEMENTDATE').agg({
    'TOTALDEMAND': 'mean',
    'TEMP': 'mean',
    'RH2M': 'mean',
    'HEAT_INDEX': 'mean'
}).reset_index()