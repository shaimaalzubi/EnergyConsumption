import pandas as pd

# 1. تحميل بيانات الطاقة الشمسية اليومية
solar_df = pd.read_csv("daily_solar_output_nsw.csv")
solar_df['Date'] = pd.to_datetime(solar_df['Date'])

# 2. تحميل بيانات استهلاك الكهرباء (كل ساعة)
consumption_df = pd.read_csv("PRICE_AND_DEMAND_2010_2024_NSW1.csv")
consumption_df['SETTLEMENTDATE'] = pd.to_datetime(consumption_df['SETTLEMENTDATE'])  # التاريخ مع الوقت

# 3. استخراج تاريخ اليوم فقط من بيانات الاستهلاك
consumption_df['Day'] = consumption_df['SETTLEMENTDATE'].dt.date
solar_df['Day'] = solar_df['Date'].dt.date

# 4. دمج على أساس اليوم فقط (تجاهل الساعة)
merged_df = pd.merge(consumption_df, solar_df[['Day', 'Daily_PV_Output_NSW']], on='Day', how='left')

# 5. حذف العمود المساعد
merged_df.drop(columns=['Day'], inplace=True)

# 6. حفظ الملف بعد الدمج
merged_df.to_csv("PRICE_AND_DEMAND_2010_2024_NSW1.csv", index=False)

print("✅ تم دمج بيانات الطاقة الشمسية اليومية مع بيانات الاستهلاك بالساعة.")
