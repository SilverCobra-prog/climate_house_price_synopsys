import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, r2_score as r2e
from sklearn.linear_model import LinearRegression, Ridge
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

# region read NOAA data
cdd_data = pd.read_csv(
    os.path.join(CLIMATE_DATA_DIRECTORY, "cdd.tsv"),
    index_col=False,
    sep="\t",
)
hdd_data = pd.read_csv(
    os.path.join(CLIMATE_DATA_DIRECTORY, "hdd.tsv"),
    index_col=False,
    sep="\t",
)
precipitation_data = pd.read_csv(
    os.path.join(CLIMATE_DATA_DIRECTORY, "pcpn.tsv"),
    index_col=False,
    sep="\t",
)
tavg_data = pd.read_csv(
    os.path.join(CLIMATE_DATA_DIRECTORY, "tavg.tsv"),
    index_col=False,
    sep="\t",
)
tmin_data = pd.read_csv(
    os.path.join(CLIMATE_DATA_DIRECTORY, "tmin.tsv"),
    index_col=False,
    sep="\t",
)
tmax_data = pd.read_csv(
    os.path.join(CLIMATE_DATA_DIRECTORY, "tmax.tsv"),
    index_col=False,
    sep="\t",
)
# endregion

for df in (
    cdd_data,
    hdd_data,
    precipitation_data,
    tavg_data,
    tmin_data,
    tmax_data,
):
    df["FIPS"] = df["Code"].apply(lambda x: int(str(x)[:-6]))
    df["Year"] = df["Code"].apply(lambda x: int(str(x)[-4:]))

tmin_data_autauga = tmin_data[tmin_data["FIPS"] == 1001]
with open("log.txt", "w") as f:
    f.write(tmin_data_autauga.to_string())

tmin_data_autauga_jan = tmin_data_autauga.drop(
    MONTHS[1:] + ["Code", "FIPS", "Year"], axis=1
)
with open("log.txt", "w") as f:
    f.write(tmin_data_autauga_jan.to_string())

# region forecast
tmin_data_autauga_jan["Lag_1"] = tmin_data_autauga_jan["Jan"].shift(1)

X = tmin_data_autauga_jan.loc[:, ["Lag_1"]]
X.dropna(inplace=True)  # drop missing values in the feature set
y = tmin_data_autauga_jan.loc[:, "Jan"]  # create the target
y, X = y.align(X, join="inner")  # drop corresponding values in target

# X = StandardScaler().fit_transform(X)
# X = PolynomialFeatures(degree=3).fit_transform(X)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# model = Ridge(alpha=1)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = pd.Series(model.predict(X_test), index=y_test.index)

print(f"R2 score: {r2e(y_test, y_pred)}")
print(f"RMSE: {mse(y_test, y_pred)**0.5}")

# Plot the actual and predicted values
plt.plot(y_train)
plt.plot(y_test)
plt.plot(y_pred, color="red")
plt.show()

# endregion
