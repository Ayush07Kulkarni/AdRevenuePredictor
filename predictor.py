import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('Advertising.csv')  

features = data[['TV', 'radio', 'newspaper']]
target = data['sales']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

tv_spend = np.linspace(0, 300, 10)
radio_spend = np.linspace(0, 50, 10)
newspaper_spend = np.linspace(0, 100, 10)

strategy_combinations = pd.DataFrame({
    'TV': np.repeat(tv_spend, len(radio_spend) * len(newspaper_spend)),
    'radio': np.tile(np.repeat(radio_spend, len(newspaper_spend)), len(tv_spend)),
    'newspaper': np.tile(newspaper_spend, len(tv_spend) * len(radio_spend))
})

strategy_predictions = model.predict(strategy_combinations)

best_strategy_index = strategy_predictions.argmax()
best_strategy = strategy_combinations.iloc[best_strategy_index]
best_sales_prediction = strategy_predictions[best_strategy_index]

print(f'Best advertisement strategy: TV: {best_strategy["TV"]}, Radio: {best_strategy["radio"]}, Newspaper: {best_strategy["newspaper"]}')
print(f'Predicted sales for the best strategy: {best_sales_prediction}')


plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
sns.scatterplot(data=data, x='TV', y='sales', label='TV Spend', color='blue')
sns.scatterplot(data=data, x='radio', y='sales', label='Radio Spend', color='orange')
sns.scatterplot(data=data, x='newspaper', y='sales', label='Newspaper Spend', color='green')
plt.title('Advertisement Spend vs Sales')
plt.xlabel('Advertisement Spend')
plt.ylabel('Sales')
plt.legend()


fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(strategy_combinations['TV'], strategy_combinations['radio'], strategy_predictions, c=strategy_predictions, cmap='viridis')
ax.set_title('Predicted Sales based on TV and Radio Spend')
ax.set_xlabel('TV Spend')
ax.set_ylabel('Radio Spend')
ax.set_zlabel('Predicted Sales')

plt.show()