import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, r2_score as r2e
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from statsmodels.tsa.arima.model import ARIMA

# region constants
MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

CLIMATE_DATA_DIRECTORY = "NOAA Climate Data/"
# endregion

house_inflation = pd.read_csv("house_inflation.csv")
house_inflation.drop(columns=["Year"], axis=1, inplace=True)
print(house_inflation)

# region forecast
house_inflation["Lag_1"] = house_inflation["Inflation Rate"].shift(1)

X = house_inflation.loc[:, ["Lag_1"]]
X.dropna(inplace=True)  # drop missing values in the feature set
y = house_inflation.loc[:, "Inflation Rate"]  # create the target
y, X = y.align(X, join="inner")  # drop corresponding values in target

# X = StandardScaler().fit_transform(X)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# model = Ridge(alpha=1)
model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
model.fit(X_train, y_train)
y_pred = pd.Series(model.predict(X), index=y.index - 1)

# print(f"R2 score: {r2e(y_test, y_pred)}")
# print(f"MSE: {mse(y_test, y_pred)}")

# # Plot the actual and predicted values
plt.plot(y_train)
plt.plot(y_test)
plt.plot(y_pred, color="red")

result_data = {"Inflation Rate": []}
cur_val = pd.DataFrame(
    {
        "Inflation Rate": [
            house_inflation["Inflation Rate"].iloc[-3],
            house_inflation["Inflation Rate"].iloc[-2],
        ]
    }
)
result_data["Inflation Rate"].append(cur_val.iloc[0][0])
for _ in range(2060 - 2021):
    # keep last two values
    cur_val = pd.DataFrame(
        {
            "Inflation Rate": [
                cur_val.iloc[1][0],
                model.predict(cur_val.iloc[0].values.reshape(1, -1))[0],
            ]
        }
    )
    result_data["Inflation Rate"].append(cur_val.iloc[0][0])

result_data = pd.DataFrame(result_data)

plt.plot(result_data)
plt.show()

print(result_data.to_string())

# endregion
