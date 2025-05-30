import pandas as pd

# تحميل البيانات (تأكدي إن المسار صحيح)
df = pd.read_csv("PRICE_AND_DEMAND_2010_2024_NSW1.csv")

# تحويل التاريخ إلى نوع datetime إذا مش محول مسبقاً
df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])

# استخراج الشهر
df['Month'] = df['SETTLEMENTDATE'].dt.month

# إنشاء عمود الفصل (Season)
def get_season(month):
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    elif month in [9, 10, 11]:
        return 'Spring'

df['Season'] = df['Month'].apply(get_season)

# حفظ الملف بنفس الاسم أو اسم جديد
df.to_csv("PRICE_AND_DEMAND_2010_2024_NSW1.csv", index=False)

print("✅ تم إضافة عمودي الشهر والفصل.")
