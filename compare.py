import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# قراءة البيانات
df = pd.read_csv("PRICE_AND_DEMAND_2010_2024_NSW1.csv")

# تقليل عدد البيانات لتجريب أسرع
df = df.sample(n=50000, random_state=42)

# حذف القيم المفقودة
available_cols = ['TOTALDEMAND', 'TEMP', 'RH2M','Season', 'HEAT_INDEX','Year','Month','Hour','IsWeekend','DayOfWeek']
df.dropna(subset=available_cols, inplace=True)

# الميزات والهدف
features = ['TEMP', 'RH2M', 'Daily_PV_Output_NSW', 'HEAT_INDEX', 'Season', 'Year', 'Month',
            'Hour', 'IsWeekend', 'DayOfWeek', 'Weekday', 'Part_of_Day', 'Is_Holiday']
X = df[features].copy()
y = df['TOTALDEMAND']

# تحويل النصوص إلى أرقام
le = LabelEncoder()
X.loc[:, 'Season'] = le.fit_transform(X['Season'])
X.loc[:, 'Part_of_Day'] = LabelEncoder().fit_transform(X['Part_of_Day'])
X.loc[:, 'Is_Holiday'] = X['Is_Holiday'].astype(int)

# مقياس البيانات (مهم للشبكات العصبية)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# بناء موديل ANN قوي
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# كولباكات
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# تدريب النموذج
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# التقييم
y_pred = model.predict(X_test).flatten()
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 تقييم النموذج المحترف:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# رسم منحنى التدريب
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
