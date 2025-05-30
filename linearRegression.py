import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# قراءة البيانات
df = pd.read_csv("PRICE_AND_DEMAND_2010_2024_NSW1.csv")

# حذف القيم المفقودة إن وُجدت
available_cols = [col for col in ['TOTALDEMAND', 'TEMP', 'RH2M', 'Season', 'HEAT_INDEX', 'Year', 'Month', 'Hour', 'IsWeekend', 'DayOfWeek'] if col in df.columns]
df.dropna(subset=available_cols, inplace=True)

# تحديد الميزات (features) والهدف (target)
features = ['TEMP', 'RH2M', 'Daily_PV_Output_NSW', 'HEAT_INDEX', 'Season', 'Year', 'Month',
            'Hour', 'IsWeekend', 'DayOfWeek', 'Weekday', 'Part_of_Day', 'Is_Holiday']

# نسخة مستقلة من الميزات
X = df[features].copy()
y = df['TOTALDEMAND']

# ترميز الأعمدة النصية
le_season = LabelEncoder()
X['Season'] = le_season.fit_transform(X['Season'])

le_part = LabelEncoder()
X['Part_of_Day'] = le_part.fit_transform(X['Part_of_Day'])

# تحويل العطلة إلى int
X['Is_Holiday'] = X['Is_Holiday'].astype(int)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج
model = LinearRegression()
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
