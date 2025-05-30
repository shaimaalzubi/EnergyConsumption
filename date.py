import pandas as pd

# قراءة الملف
df = pd.read_csv("PRICE_AND_DEMAND_2010_2024_NSW1.csv")

# تحويل عمود التاريخ لتاريخ فعلي
df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])

# استخراج خصائص من التاريخ
df['Year'] = df['SETTLEMENTDATE'].dt.year
df['Month'] = df['SETTLEMENTDATE'].dt.month
df['Day'] = df['SETTLEMENTDATE'].dt.day
df['Hour'] = df['SETTLEMENTDATE'].dt.hour
df['DayOfWeek'] = df['SETTLEMENTDATE'].dt.dayofweek  # Monday=0, Sunday=6
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# حفظ التحديث
df.to_csv("PRICE_AND_DEMAND_2010_2024_NSW1.csv", index=False)

print("✅ تم استخراج معلومات التاريخ والوقت بنجاح!")
