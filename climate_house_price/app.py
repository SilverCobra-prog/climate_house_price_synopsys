from flask import Flask, flash, redirect, request, render_template, url_for
from tkinter import *
import os
import pickle
import warnings
import math
import tkinter.messagebox

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, r2_score as r2e
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

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

house_pricing = pd.read_csv("house_pricing.csv", index_col=False, sep="\t")
house_inflation = pd.read_csv("house_inflation.csv")

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
avg_temp_data = pd.read_csv(
    os.path.join(CLIMATE_DATA_DIRECTORY, "tavg.tsv"),
    index_col=False,
    sep="\t",
)
min_temp_data = pd.read_csv(
    os.path.join(CLIMATE_DATA_DIRECTORY, "tmin.tsv"),
    index_col=False,
    sep="\t",
)
max_temp_data = pd.read_csv(
    os.path.join(CLIMATE_DATA_DIRECTORY, "tmax.tsv"),
    index_col=False,
    sep="\t",
)
# endregion

for df in (
    cdd_data,
    hdd_data,
    precipitation_data,
    avg_temp_data,
    min_temp_data,
    max_temp_data,
):
    df["FIPS"] = df["Code"].apply(lambda x: int(str(x)[:-6]))
    df["Year"] = df["Code"].apply(lambda x: int(str(x)[-4:]))

fips = pd.read_csv(
    os.path.join(CLIMATE_DATA_DIRECTORY, "fips.csv"), index_col=False
)

# the NOAA uses different state codes at the beginning of the fips code
with open("noaa_state_codes.txt", "r") as f:
    noaa_state_codes = {
        line.split(",")[0].strip(): line.split(",")[1].strip()
        for line in f.readlines()
    }

# convert state name at end of county name to state abbreviation in
# normalized_house_pricing
state_abbreviations = pd.read_csv("state_abbreviations.csv", index_col=False)

# Load your climate data and create your Ridge regression model
# X, y = load_climate_data()
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# ridge = Ridge(alpha=1.0)
# ridge.fit(X_scaled, y)

# load model from pickle file
model = pickle.load(open("climate_to_growth_model.pkl", "rb"))
with np.printoptions(threshold=np.inf) as _:
        print(model.named_steps["polynomialfeatures"].powers_, file=open("log.txt", "a"))
        print(model.named_steps["ridge"].coef_, file=open("log.txt", "a"))

full_climate_data = cdd_data.rename(
    {month: f"{month} cooling degree days" for month in MONTHS},
    axis=1,
)
for (df, colname) in [
    (hdd_data, "heating degree days"),
    (precipitation_data, "precipitation"),
    (avg_temp_data, "average temperature"),
    (min_temp_data, "minimum temperature"),
    (max_temp_data, "maximum temperature"),
]:
    # print(df)
    full_climate_data = full_climate_data.merge(
        df.rename(
            {month: f"{month} {colname}" for month in MONTHS},
            axis=1,
        ).drop(
            columns=["Code"],
            axis=1,
        ),
        on=["FIPS", "Year"],
    )

# Create a Flask application
app = Flask(__name__)

# # load climate data
# def load_climate_data():
#     # Load the climate data
#     data = np.loadtxt('climate_data.csv', delimiter=',', skiprows=1)

#     # Extract the input data and the target
#     X = data[:, :-1]
#     y = data[:, -1]

#     return X, y

# Define a route for the home page
# @app.route('/')
# def main():
#     return (render_template('index.html'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/results',methods=['POST'])
def results():
    form = request.form
    model = pickle.load(open("climate_to_growth_model.pkl", "rb"))
    if request.method == 'POST':
        FIPS_TO_FORECAST = int(request.form['county'])
        YEAR_TO_FORECAST = int(request.form['year'])
        HOUSE_PRICE = float(request.form['price'])

        # if YEAR_TO_FORECAST <= 2021:
        #     flash('Year must be 2021 or later')
        #     return render_template('index.html')

        COLNAMES = [
            "cooling degree days",
            "heating degree days",
            "precipitation",
            "average temperature",
            "minimum temperature",
            "maximum temperature",
        ]

        full_data_county = full_climate_data[
            full_climate_data["FIPS"] == FIPS_TO_FORECAST
        ]
        full_data_county = full_data_county[full_data_county["Year"] != 2023]

        result_data = defaultdict(list)
        if YEAR_TO_FORECAST > 2021:
            for colname in COLNAMES:
                for month in MONTHS:
                    index = f"{month} {colname}"

                    full_data_month = full_data_county.drop(
                        [f"{m} {colname}" for m in MONTHS if m != month]
                        + ["Code", "FIPS", "Year"],
                        axis=1,
                    )
                    for c in COLNAMES:
                        if c != colname:
                            full_data_month = full_data_month.drop(
                                [f"{m} {c}" for m in MONTHS],
                                axis=1,
                            )

                    # print(full_data_month)

                    full_data_month["Lag_1"] = full_data_month[index].shift(1)

                    X = full_data_month.loc[:, ["Lag_1"]]
                    X.dropna(inplace=True)  # drop missing values in the feature set
                    y = full_data_month.loc[:, index]  # create the target
                    y, X = y.align(
                        X, join="inner"
                    )  # drop corresponding values in target

                    forecast_model = make_pipeline(
                        # PolynomialFeatures(degree=3),
                        LinearRegression(),
                    )
                    forecast_model.fit(X, y)

                    # print(X.columns)

                    cur_val = pd.DataFrame(
                        {
                            index: [
                                full_data_county[index].iloc[-3],
                                full_data_county[index].iloc[-2],
                            ]
                        }
                    )
                    result_data[index].append(cur_val.iloc[0][0])
                    for _ in range(YEAR_TO_FORECAST - 2021):
                        # keep last two values
                        # print(cur_val)
                        cur_val = pd.DataFrame(
                            {
                                index: [
                                    cur_val.iloc[1][0],
                                    forecast_model.predict(
                                        cur_val.iloc[0].values.reshape(1, -1)
                                    )[0],
                                ]
                            }
                        )
                        result_data[index].append(cur_val.iloc[0][0])
                    # for _ in range(YEAR_TO_FORECAST - 2021):
                    #     cur_val = [np.clip(forecast_model.predict(cur_val), -100, 1000)]
                    #     result_data[index].append(cur_val[0][0])
                    # plt.plot(result_data[index])
                    # plt.show()
        else:
            for month in MONTHS:
                for colname in COLNAMES:
                    result_data[f"{month} {colname}"].append(
                        full_data_county[f"{month} {colname}"].iloc[-1]
                    )

        # print(dict(result_data))

        result_data = pd.DataFrame(result_data)

        # print("\n\n"+result_data.to_string(), file=open("log.txt", "a"))

        # print(result_data.columns)

        yoy_growth = model.predict(result_data)

        print(yoy_growth, file=open("log.txt", "w"))

        # add 1 and multiply all values in yoy_growth together to get the total growth
        total_growth = 1
        for val in yoy_growth:
            total_growth *= val + 1

        from_climate_change = total_growth
        # print("From climate change:", total_growth)


        # region inflation forecast
        house_inflation_forecast = house_inflation.copy()

        house_inflation_forecast["Inflation Rate"] /= 100
        house_inflation_forecast["Inflation Rate"] += 1
        house_inflation_forecast["Inflation Rate"] = house_inflation_forecast[
            "Inflation Rate"
        ].cumprod()
        house_inflation_forecast.drop(columns=["Year"], axis=1, inplace=True)
        # print(house_inflation_forecast)

        house_inflation_forecast["Lag_1"] = house_inflation_forecast[
            "Inflation Rate"
        ].shift(1)

        X = house_inflation_forecast.loc[:, ["Lag_1"]]
        X.dropna(inplace=True)  # drop missing values in the feature set
        y = house_inflation_forecast.loc[:, "Inflation Rate"]  # create the target
        y, X = y.align(X, join="inner")  # drop corresponding values in target

        # X = StandardScaler().fit_transform(X)

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        # model = Ridge(alpha=1)
        model = make_pipeline(LinearRegression())
        model.fit(X_train, y_train)
        y_pred = pd.Series(model.predict(X), index=y.index - 1)

        # print(f"R2 score: {r2e(y_test, y_pred)}")
        # print(f"MSE: {mse(y_test, y_pred)}")

        # # Plot the actual and predicted values
        # plt.plot(y_train)
        # plt.plot(y_test)
        # plt.plot(y_pred, color="red")

        inflation_2021 = house_inflation_forecast["Inflation Rate"].iloc[-2]

        result_data = {"Inflation Rate": []}
        cur_val = pd.DataFrame(
            {
                "Inflation Rate": [
                    house_inflation_forecast["Inflation Rate"].iloc[-3],
                    house_inflation_forecast["Inflation Rate"].iloc[-2],
                ]
            }
        )
        result_data["Inflation Rate"].append(cur_val.iloc[0][0])
        for _ in range(YEAR_TO_FORECAST - 2021):
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

        # plt.plot(result_data)
        # plt.show()

        # print(result_data.to_string())

        national_house_inflation = (
            result_data["Inflation Rate"].iloc[-1] / inflation_2021
        )

        total_growth *= national_house_inflation

        inflation_total_growth = total_growth
        print("In total:", total_growth)

        calculate_house_price = round(total_growth * HOUSE_PRICE, 2)

        price_difference = round(calculate_house_price - HOUSE_PRICE, 2)

        # endregion
        # return climate_growth and inflation_growth to results.html
        return render_template(
            "results.html", 
            climate_prediction=from_climate_change, 
            inflation_prediction=inflation_total_growth, 
            price_prediction=calculate_house_price,
            year=YEAR_TO_FORECAST,
            price_difference=price_difference)
        
if __name__ == '__main__':
    # Start the Flask application on localhost:5000
    app.run(debug=True)


