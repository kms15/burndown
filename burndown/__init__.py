# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2022, Kendrick Shaw
import pandas as pd
import numpy as np
import matplotlib.pyplot
from scipy.interpolate import interp1d
import re


def read_csv(filename: str):
    result = pd.read_csv(filename, dtype={"points": float})

    # convert datecols to np.datetime64 values, forcing errors if failed
    datecols = ["created", "committed", "completed"]
    for col in datecols:
        result[col] = pd.to_datetime(result[col], utc=True)

    # fill missing notes and nextsteps with blanks
    result.fillna(
        {
            "notes": "",
            "nextstep": "",
        },
        inplace=True,
    )

    # raise an exception if any created dates are missing
    missing_created = result.loc[pd.isnull(result["created"])]
    if missing_created.shape[0] > 0:
        line = missing_created.index[0] + 2
        raise Exception(f"Missing creation time on line {line}")

    return result


def burndown_timeseries(credit_times_or_df, debit_times=None, points=None):
    # if this looks like a dataframe
    if len(credit_times_or_df.shape) > 1:
        # use the 'committed', 'completed', and 'points' columns
        assert debit_times is None
        assert points is None
        credit_times = credit_times_or_df["committed"]
        debit_times = credit_times_or_df["completed"]
        points = credit_times_or_df["points"]
    else:
        # confirm that we were passed at least the points
        assert not points is None
        credit_times = credit_times_or_df
        # if debit_times is None, assume that there are no debits
        if debit_times is None:
            debit_times = pd.to_datetime([""] * len(points), utc=True)

    credits = points[credit_times.notna()].set_axis(credit_times.dropna())
    debits = (-1 * points[debit_times.notna()]).set_axis(debit_times.dropna())
    deltas = pd.concat([credits, debits]).sort_index()
    return deltas.cumsum()


def with_tag(df, tag):
    return df.loc[df["notes"].str.contains("#" + tag + "\\b", regex=True)]


def without_tag(df, tag):
    return df.loc[~df["notes"].str.contains("#" + tag + "\\b", regex=True)]


def prepare_stackplot_df(
    df,
    category_tags,
    burnup=False,
):

    # Make the times unique by adding the row number to the nanoseconds
    # (needed to avoid errors from pandas about duplicate indexes)
    df = df.reset_index()
    for j, colname in enumerate(["committed", "completed"]):
        df[colname] += pd.Series([pd.Timedelta(i + j * df.shape[0]) for i in df.index])

    without_triaged_df = without_tag(df, "triaged")
    completed_ts = burndown_timeseries(
        without_triaged_df["completed"], points=without_triaged_df["points"]
    )

    # start by treating everything as uncategorized
    uncategorized_ts = burndown_timeseries(df)

    # add in all of the tagged timeseries
    timeseries = {"completed": completed_ts, "uncategorized": uncategorized_ts}
    for tag in category_tags:
        timeseries[tag] = burndown_timeseries(with_tag(df, tag))

    # group all of the timeseries into a single dataframe
    df_result = pd.DataFrame(timeseries)

    # fill in nulls
    df_result = df_result.fillna(method="ffill").fillna(0)

    # round to the nearest second (to remove nanoseconds added to create unique times)
    df_result.set_index(
        df_result.index
        - pd.Series([pd.Timedelta(ns) for ns in df_result.index.nanosecond]),
        inplace=True,
    )

    # use the most recent result for duplicate times
    df_result = df_result.reset_index().groupby("index").last()

    # subtracted the tagged/categorized timeseries from the uncategorized total
    for tag in category_tags:
        df_result["uncategorized"] -= df_result[tag]

    if not burnup:
        df_result.drop("completed", inplace=True, axis=1)

    return df_result


def burndown_plot(
    df,
    category_tags,
    burnup=False,
    stackplot_function=matplotlib.pyplot.stackplot,
    prepare_df_function=prepare_stackplot_df,
    time_scaler=lambda x: x,
    **kwargs,
):

    df_result = prepare_df_function(df, category_tags, burnup)

    colors = [
        "silver",
        "indianred",
        "coral",
        "goldenrod",
        "yellowgreen",
        "forestgreen",
        "darkcyan",
        "cornflowerblue",
        "blueviolet",
        "orchid",
    ]

    # show "completed" data as transparent
    if burnup:
        colors.insert(0, (0, 0, 0, 0))

    return stackplot_function(
        time_scaler(df_result.index),
        df_result.transpose(),
        labels=df_result.columns,
        colors=colors,
        **kwargs,
    )


def get_spanning_days(dts):
    return pd.date_range(
        start=dts.min().normalize(), end=dts.max() + pd.to_timedelta("1D"), freq="D"
    )


def is_weekend(dts):
    return dts.day_of_week >= 5


def get_time_scaler(spanning_days, is_workday, non_workday_value=1 / 16):
    y = np.cumsum(
        [0] + [(1 if workday else non_workday_value) for workday in is_workday[:-1]]
    )
    x = spanning_days.to_julian_date()
    f = interp1d(x, y)

    return lambda dts: f(dts.to_julian_date())


def extract_holidays_from_ical(ical):
    lines = ical.split("\n")
    result = []
    for line in lines:
        m = re.match(r"DTSTART;VALUE=DATE:([0-9]{8,8})", line.strip())
        if m:
            result.append(m.group(1))
    return pd.to_datetime(result, utc=True).to_list()


def is_workday(dts, holidays):
    return [not is_weekend(d) and not d in holidays for d in dts]
