import pandas as pd

# قراءة ملف الكهرباء اللي فيه T2M و RH
df = pd.read_csv("PRICE_AND_DEMAND_2010_2024_NSW1.csv")

# التأكد من أن القيم رقمية
df['TEMP'] = pd.to_numeric(df['TEMP'], errors='coerce')
df['RH2M'] = pd.to_numeric(df['RH2M'], errors='coerce')

# معادلة مؤشر الحرارة (Heat Index) بوحدة فهرنهايت
# أولاً نحول درجة الحرارة من مئوية إلى فهرنهايت
T_f = df['TEMP'] * 9/5 + 32
R = df['RH2M']

# معادلة Heat Index من NOAA
HI_f = (
    -42.379 + 2.04901523 * T_f + 10.14333127 * R
    - 0.22475541 * T_f * R - 0.00683783 * T_f**2
    - 0.05481717 * R**2 + 0.00122874 * T_f**2 * R
    + 0.00085282 * T_f * R**2 - 0.00000199 * T_f**2 * R**2
)

# تحويل HI إلى مئوي
df['HEAT_INDEX'] = (HI_f - 32) * 5/9

# حفظ الملف مع التعديل على نفس الاسم
df.to_csv("PRICE_AND_DEMAND_2010_2024_NSW1.csv", index=False)

print("✅ تم حساب مؤشر الحرارة (Heat Index) وحفظه في نفس الملف.")

