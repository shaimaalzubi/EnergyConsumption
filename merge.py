import pandas as pd
import os

# 🟢 تحديد مسار المجلد اللي فيه ملفات الكهرباء
folder_path = "DATA/PRICE_AND_DEMAND"  # غيّر الاسم إذا مجلدك مختلف

# 🟢 جمع كل الملفات التي تنتهي بـ .csv وتبدأ بـ PRICE_AND_DEMAND
file_paths = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if f.startswith("PRICE_AND_DEMAND") and f.endswith(".csv")
]

# 🟢 دمج جميع الملفات
all_data = []
for file in sorted(file_paths):
    print(f"📂 جاري دمج الملف: {file}")
    df = pd.read_csv(file)
    df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
    all_data.append(df)

# 🟢 دمج البيانات في DataFrame واحد
merged_df = pd.concat(all_data).sort_values('SETTLEMENTDATE')

# 🟢 حفظ الناتج في ملف واحد
merged_df.to_csv("PRICE_AND_DEMAND_2010_2024_NSW1.csv", index=False)

print("✅ تم دمج الملفات بنجاح وحفظها في PRICE_AND_DEMAND_2010_2024_NSW1.csv")
