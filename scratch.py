import pandas as pd

# قراءة الملف
df = pd.read_csv("PRICE_AND_DEMAND_2010_2024_NSW1.csv")

# حذف العمود (استبدلي 'اسم_العمود' بالعمود اللي بدك تحذفيه)
df.drop(columns=['D'], inplace=True)

# حفظ الملف المعدل بنفس الاسم
df.to_csv("PRICE_AND_DEMAND_2010_2024_NSW1.csv", index=False)

print("✅ تم حذف العمود وحفظ الملف.")
