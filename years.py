import pandas as pd
import matplotlib.pyplot as plt

# تحميل البيانات
df = pd.read_csv('PRICE_AND_DEMAND_2010_2024_NSW1.csv')  # غيّر اسم الملف إذا لازم

# تحويل العمود لتاريخ
df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])

# استخراج السنة
df['Year'] = df['SETTLEMENTDATE'].dt.year

# حساب مجموع الاستهلاك لكل سنة
yearly_consumption = df.groupby('Year')['TOTALDEMAND'].sum()

# عرض النتائج
print(yearly_consumption)

# رسم بياني
plt.figure(figsize=(10,6))
yearly_consumption.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Total Energy Consumption per Year')
plt.xlabel('Year')
plt.ylabel('Total Demand (MW)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
