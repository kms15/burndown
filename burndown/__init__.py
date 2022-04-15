# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2022, Kendrick Shaw
import pandas as pd
import numpy as np
import matplotlib.pyplot


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


def burndown_plot(
    df,
    category_tags,
    burnup=False,
    stackplot_func=matplotlib.pyplot.stackplot,
    **kwargs,
):
    completed_ts = burndown_timeseries(df["completed"], points=df["points"])

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

    # use the most recent result for duplicate times
    df_result = df_result.reset_index().groupby("index").last()

    # subtracted the tagged/categorized timeseries from the uncategorized total
    for tag in category_tags:
        df_result["uncategorized"] -= df_result[tag]

    colors = [
        "white",
        "silver",
        "coral",
        "goldenrod",
        "forestgreen",
        "cornflowerblue",
        "darkorchid",
    ]

    # remove extra burnup data for a standard burndown plot
    if not burnup:
        df_result.drop("completed", inplace=True, axis=1)
        colors = colors[1:]

    return stackplot_func(
        df_result.index,
        df_result.to_numpy().transpose(),
        labels=df_result.columns,
        colors=colors,
        **kwargs,
    )
