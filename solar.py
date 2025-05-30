import pandas as pd

# === 1. تحميل ملف بيانات الإنتاج الشمسي الشهري ===
file_path = "DATA/SOLAR/states-output-monthly.csv"  # غيّر المسار إذا الملف في مكان آخر
nsw_df = pd.read_csv(file_path)[['Date', 'Output NSW']].copy()

# === 2. تحويل التاريخ من نص إلى datetime حسب تنسيق YYYY-MM مثل "2015-04"
nsw_df['Date'] = pd.to_datetime(nsw_df['Date'], format="%Y-%m")

# === 3. إنشاء تواريخ يومية من 2010 إلى اليوم الحالي
daily_dates = pd.date_range(start="2010-01-01", end=pd.Timestamp.today(), freq='D')
daily_df = pd.DataFrame({'Date': daily_dates})

# === 4. إنشاء عمود YearMonth لتسهيل الدمج
nsw_df['YearMonth'] = nsw_df['Date'].dt.to_period('M')
daily_df['YearMonth'] = daily_df['Date'].dt.to_period('M')

# === 5. دمج البيانات اليومية مع الشهرية بناءً على YearMonth
merged_df = pd.merge(daily_df, nsw_df[['YearMonth', 'Output NSW']], on='YearMonth', how='left')

# === 6. توزيع الإنتاج الشهري على الأيام بالتساوي
days_per_month = daily_df['YearMonth'].value_counts().sort_index()
merged_df['DaysInMonth'] = merged_df['YearMonth'].map(days_per_month)
merged_df['Daily_PV_Output_NSW'] = merged_df['Output NSW'] / merged_df['DaysInMonth']

# === 7. ملء القيم المفقودة (قبل 2015) بصفر
merged_df['Daily_PV_Output_NSW'] = merged_df['Daily_PV_Output_NSW'].fillna(0)

# === 8. الإخراج النهائي: التاريخ + الإنتاج الشمسي اليومي
final_df = merged_df[['Date', 'Daily_PV_Output_NSW']]

# === 9. (اختياري) حفظ البيانات في ملف CSV
final_df.to_csv("daily_solar_output_nsw.csv", index=False)

print("✅ تم إنشاء ملف 'daily_solar_output_nsw.csv' بنجاح.")
