import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Ma'lumotlar manbasidan ma'lumotlar olish
data = {
    'Sana': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05'],
    'Sotuv': [100, 120, 110, 130, 140],
    'Kirim': [50, 60, 70, 80, 90]
}

df = pd.DataFrame(data)

# Ma'lumotlarni tayyorlash
df['Sana'] = pd.to_datetime(df['Sana'])
df['Sotuv'] = pd.to_numeric(df['Sotuv'])
df['Kirim'] = pd.to_numeric(df['Kirim'])

# Sotuv va kirimlar o'rtasidagi munosabatni tekshirish
plt.scatter(df['Sotuv'], df['Kirim'])
plt.xlabel('Sotuv')
plt.ylabel('Kirim')
plt.show()

# Sotuv va kirimlar o'rtasidagi munosabatni modellovchi
X = df[['Sotuv']]
y = df['Kirim']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

# Modeldan ma'lumotlar olish
sotuv = 150
kirimga_tahmin = model.predict([[sotuv]])
print(f'{sotuv} sotuvdan {kirimga_tahmin[0]} kirim olish mumkin')
