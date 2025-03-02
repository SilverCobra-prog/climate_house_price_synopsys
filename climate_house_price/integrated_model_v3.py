import os
import pickle
import warnings

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


if not os.path.exists("climate_to_growth_model.pkl"):
    # normalized based on 2021 inflation
    normalized_house_pricing = house_pricing.copy().rename(
        columns={
            "ABS-diff price Q2 2021": "Normalized Growth 2021",
            "Geography": "County",
        }
    )

    # drop NaN, cast % string to float, and normalize
    normalized_house_pricing["Normalized Growth 2021"] = (
        normalized_house_pricing["Normalized Growth 2021"]
        .dropna()
        .str.rstrip("%")
        .astype("float")
        + 1
    ) / (100.0 * (1 + house_inflation["Inflation Rate"][2021 - 1968]))

    # remove price ranges, growth Q42021, latlong, and full county number
    normalized_house_pricing = normalized_house_pricing.drop(
        columns=[
            "Price Range-Q3 2022",
            "growth Q42021",
            "Latitude (generated)",
            "Full County Number",
            "Price Range-Q3 2022.1",
            "Longitude (generated)",
        ]
    )

    # remove first 2 digits of each fips and replace with NOAA code for that state
    fips["noaa_state_code"] = fips["state_name"].map(noaa_state_codes)
    fips["fips"] = fips["fips"].astype("str").apply(lambda x: x[-3:])
    fips["fips"] = (
        (fips["noaa_state_code"] + fips["fips"]).dropna().astype("int")
    )

    fips_simple = fips[["fips", "long_name"]].rename(
        columns={"long_name": "County"}
    )

    # with open("log.txt", "w") as f:
    #     f.write(fips_simple.to_string())

    # convert normalized_house_pricing['State'] to state abbreviation
    normalized_house_pricing["County"] = (
        normalized_house_pricing["County"].str.split(",").str[0]
        + " "
        + (
            normalized_house_pricing["County"]
            .str.split(",")
            .str[1]
            .str.strip()
            .map(state_abbreviations.set_index("state")["code"])
        )
    )

    # add fips column to normalized_house_pricing
    normalized_house_pricing = normalized_house_pricing.merge(
        fips_simple, on="County"
    )

    # dropna for Washington DC
    normalized_house_pricing["fips"] = (
        normalized_house_pricing["fips"].dropna().astype("int")
    )

    # print(normalized_house_pricing)

    full_data = normalized_house_pricing.copy()

    cdd_data_q3_2021 = cdd_data[cdd_data["Year"] == 2021]
    hdd_data_q3_2021 = hdd_data[hdd_data["Year"] == 2021]
    precipitation_data_q3_2021 = precipitation_data[
        precipitation_data["Year"] == 2021
    ]
    avg_temp_data_q3_2021 = avg_temp_data[avg_temp_data["Year"] == 2021]
    min_temp_data_q3_2021 = min_temp_data[min_temp_data["Year"] == 2021]
    max_temp_data_q3_2021 = max_temp_data[max_temp_data["Year"] == 2021]

    for (df, colname) in [
        (cdd_data_q3_2021, "cooling degree days"),
        (hdd_data_q3_2021, "heating degree days"),
        (precipitation_data_q3_2021, "precipitation"),
        (avg_temp_data_q3_2021, "average temperature"),
        (min_temp_data_q3_2021, "minimum temperature"),
        (max_temp_data_q3_2021, "maximum temperature"),
    ]:
        # print(df)
        full_data = full_data.merge(
            df.rename(
                {month: f"{month} {colname}" for month in MONTHS},
                axis=1,
            ),
            left_on=["fips"],
            right_on=["FIPS"],
        ).drop(
            columns=["FIPS", "Code", "Year"],
            axis=1,
        )

    # print(full_data)

    # remove the commas in between the prices in the Price Range-Q3 2022, Q1 2022,
    # and Q4 2021 columns
    for col in ("Price Q3 2022", "Q1 2022", "Q4 2021"):
        full_data[col] = (
            full_data[col]
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .astype(float)
        )
    full_data = full_data.rename(
        columns={
            "Price Q3 2022": "Q3 2022 ($)",
            "Q1 2022": "Q1 2022 ($)",
            "Q4 2021": "Q4 2021 ($)",
        }
    )

    # with open("log.txt", "w") as f:
    #     # sort by county name
    #     f.write(full_data.sort_values(by="County").to_string())

    # split the data into training and testing sets
    X = full_data.drop(
        columns=[
            "County",
            "Normalized Growth 2021",
            "Q3 2022 ($)",
            "Q1 2022 ($)",
            "Q4 2021 ($)",
            "fips",
        ]
    )
    y = full_data["Normalized Growth 2021"]

    # print(X.to_string(), file=open("log.txt", "w"))
    # print(X)

    # X = StandardScaler().fit_transform(X)
    # X = PolynomialFeatures(degree=2).fit_transform(X)

    # print(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    # region ridgereg
    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2),
        Ridge(alpha=7),
        # LinearRegression(),
    )
    # print(X_test.shape, "248")
    model.fit(X_train, y_train)

    # pickle model
    pickle.dump(model, open("climate_to_growth_model.pkl", "wb"))
else:
    print("loading model from pickle file")
    model = pickle.load(open("climate_to_growth_model.pkl", "rb"))
    with np.printoptions(threshold=np.inf) as _:
        print(model.named_steps["polynomialfeatures"].powers_, file=open("log.txt", "a"))
        print(model.named_steps["ridge"].coef_, file=open("log.txt", "a"))
# endregion

# region evaluate
# y_pred = model.predict(X_test)
# mean_squared_error = mse(y_test, y_pred)
# r2_score = r2e(y_test, y_pred)

# print("Mean squared error: %.6f" % mean_squared_error)
# print("Sqrt of mean squared error: %.6f" % (mean_squared_error) ** 0.5)
# print("Variance score: %.2f" % r2_score)

# # plot the predicted normalized growth vs the actual normalized growth
# plt.scatter(y_test, y_pred, color="black")
# plt.plot(
#     [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=4
# )
# # line of best fit
# plt.plot(
#     np.unique(y_test),
#     np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)),
#     color="red",
#     linewidth=3,
#     label="Line of Best Fit",
# )
# plt.legend()
# plt.title("Predicted vs Actual Growth")
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.show()
# endregion

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

FIPS_TO_FORECAST = int(input("Enter FIPS code: "))
YEAR_TO_FORECAST = int(input("Enter year to forecast: "))
if YEAR_TO_FORECAST < 2021:
    raise ValueError("Year must be 2021 or later")

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

print("From climate change:", total_growth)


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

print("In total:", total_growth)

# endregion
