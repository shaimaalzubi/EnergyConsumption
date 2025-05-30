import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# تحميل البيانات
df = pd.read_csv("PRICE_AND_DEMAND_2010_2024_NSW1.csv")

# تأكد من أن العمودين موجودين
assert 'HEAT_INDEX' in df.columns, "عمود HEAT_INDEX غير موجود!"
assert 'TOTALDEMAND' in df.columns, "عمود TOTALDEMAND غير موجود!"

# حذف أي صف فيه قيم مفقودة
df = df[['HEAT_INDEX', 'TOTALDEMAND']].dropna()

# حساب معامل الارتباط
correlation = df['HEAT_INDEX'].corr(df['TOTALDEMAND'])
print(f"📊 معامل الارتباط بين HEAT_INDEX و TOTALDEMAND هو: {correlation:.3f}")

# رسم Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='HEAT_INDEX', y='TOTALDEMAND', data=df, alpha=0.3)
plt.title('العلاقة بين مؤشر الحرارة واستهلاك الكهرباء')
plt.xlabel('HEAT_INDEX (°C)')
plt.ylabel('TOTALDEMAND (MW)')
plt.grid(True)
plt.tight_layout()
plt.show()
