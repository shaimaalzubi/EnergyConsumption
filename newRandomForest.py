import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# --- قراءة البيانات ---
df = pd.read_csv("PRICE_AND_DEMAND_2010_2024_NSW1.csv")

# (اختياري) تقليل البيانات مؤقتاً للتجريب السريع
df = df.sample(n=50000, random_state=42)

# حذف القيم المفقودة
available_cols = [col for col in ['TOTALDEMAND','RRP', 'TEMP', 'RH2M', 'Season', 'HEAT_INDEX',
                                  'Year', 'Month', 'Hour', 'IsWeekend', 'DayOfWeek'] if col in df.columns]
df.dropna(subset=available_cols, inplace=True)

# --- تجهيز الميزات ---
features = ['TEMP','RRP', 'RH2M', 'Daily_PV_Output_NSW', 'HEAT_INDEX', 'Season',
            'Year', 'Month', 'Hour', 'IsWeekend', 'DayOfWeek', 'Weekday',
            'Part_of_Day', 'Is_Holiday']

X = df[features].copy()
y = df['TOTALDEMAND']

# --- ترميز الميزات النصية ---
le = LabelEncoder()
X.loc[:, 'Season'] = le.fit_transform(X['Season'])
X.loc[:, 'Part_of_Day'] = LabelEncoder().fit_transform(X['Part_of_Day'])
X.loc[:, 'Is_Holiday'] = X['Is_Holiday'].astype(int)

# --- تقسيم البيانات ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- إنشاء نموذج Random Forest محسّن ---
model = RandomForestRegressor(
    n_estimators=300,          # عدد الأشجار (زيادة يعطي استقرار أكثر)
    max_depth=20,              # تحديد عمق الشجرة يمنع الـ overfitting
    min_samples_split=5,       # تقسيم فقط لما يكون فيه 5 عينات على الأقل
    min_samples_leaf=3,        # أقل عدد للأوراق = 3
    max_features='sqrt',       # كل شجرة تاخذ جذر عدد الميزات بشكل عشوائي
    bootstrap=True,            # أخذ عينات مع إرجاع (عشان تنوع الأشجار)
    random_state=42,
    n_jobs=-1                  # استخدام كل المعالجات المتاحة (تسريع)
)

# --- تدريب النموذج ---
model.fit(X_train, y_train)

# --- التنبؤ ---
y_pred = model.predict(X_test)

# --- التقييم ---
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 تقييم النموذج المحسن بالـ Random Forest:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# --- أهمية الميزات ---
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# --- رسم ---
plt.figure(figsize=(10,6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='green')
plt.xlabel('أهمية الميزة')
plt.title('Feature Importance (Random Forest)')
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()
