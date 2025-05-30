import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# قراءة البيانات
df = pd.read_csv("PRICE_AND_DEMAND_2010_2024_NSW1.csv")

# (اختياري) تقليل عدد البيانات للتجريب السريع
df = df.sample(n=50000, random_state=42)  # عدّلي العدد إذا حبيتي

# حذف القيم المفقودة إن وُجدت
available_cols = [col for col in ['TOTALDEMAND', 'TEMP', 'RH2M','Season', 'HEAT_INDEX','Year','Month','Hour','IsWeekend','DayOfWeek'] if col in df.columns]
df.dropna(subset=available_cols, inplace=True)

# تحديد الميزات والهدف
features = ['TEMP', 'RH2M', 'Daily_PV_Output_NSW', 'HEAT_INDEX', 'Season', 'Year', 'Month',
            'Hour', 'IsWeekend', 'DayOfWeek', 'Weekday', 'Part_of_Day', 'Is_Holiday']
X = df[features].copy()
y = df['TOTALDEMAND']

# تحويل الأعمدة النصية أو المنطقية إلى رقمية
le = LabelEncoder()
X.loc[:, 'Season'] = le.fit_transform(X['Season'])

X.loc[:, 'Part_of_Day'] = LabelEncoder().fit_transform(X['Part_of_Day'])
X.loc[:, 'Is_Holiday'] = X['Is_Holiday'].astype(int)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج باستخدام Random Forest مع توازي
model = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)  # عدلي n_estimators إذا حبيتي
model.fit(X_train, y_train)

# التنبؤ
y_pred = model.predict(X_test)

# التقييم
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("📊 تقييم النموذج:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")
# استخراج أهمية الميزات
feature_importances = model.feature_importances_

# إنشاء DataFrame مرتب
features = features
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# ترتيب الترتيب تنازلي
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# عرض النتائج
print(importance_df)

# رسم رسم بياني للميزات
plt.figure(figsize=(10,6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('أهمية الميزة')
plt.title('Feature Importance in Random Forest')
plt.gca().invert_yaxis()  # عشان الأهم تكون فوق
plt.grid(True)
plt.show()