import pandas as pd

# قراءة ملف الكهرباء
power_df = pd.read_csv("PRICE_AND_DEMAND_2010_2024_NSW1.csv")
power_df['SETTLEMENTDATE'] = pd.to_datetime(power_df['SETTLEMENTDATE'])
power_df['SETTLEMENT_HOUR'] = power_df['SETTLEMENTDATE'].dt.floor('H')  # تقريب للساعة

# قراءة ملف الحرارة وتجاهل أول 9 أسطر
weather_df = pd.read_csv(
    "DATA/HUMIDITY/POWER_Point_Hourly_20100101_20241231_033d87S_151d20E_LST (2).csv",
    skiprows=9
)

# تحويل التاريخ والوقت في ملف الحرارة إلى datetime
weather_df['DATETIME'] = pd.to_datetime(dict(
    year=weather_df['YEAR'],
    month=weather_df['MO'],
    day=weather_df['DY'],
    hour=weather_df['HR']
))

# اختيار الأعمدة المهمة فقط
weather_df = weather_df[['DATETIME', 'RH2M']]

# دمج البيانات في نفس الـ DataFrame الأصلي
power_df = pd.merge(
    power_df,
    weather_df,
    left_on='SETTLEMENT_HOUR',
    right_on='DATETIME',
    how='left'
)

# حذف الأعمدة المؤقتة
power_df.drop(['SETTLEMENT_HOUR', 'DATETIME'], axis=1, inplace=True)

# حفظ الملف المعدل بنفس الاسم
power_df.to_csv("PRICE_AND_DEMAND_2010_2024_NSW1.csv", index=False)

print("✅ تم تحديث الملف وإضافة بيانات الحرارة بنجاح.")
