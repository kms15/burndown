# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2022, Kendrick Shaw
import pandas as pd
import numpy as np


def read_csv(filename: str):
    result = pd.read_csv(filename, dtype={"points" : float})

    # convert datecols to np.datetime64 values, forcing errors if failed
    datecols = ["created", "committed", "completed"]
    for col in datecols:
        result[col] = pd.to_datetime(result[col])

    # fill missing notes and nextsteps with blanks
    result.fillna({
        'notes': '',
        'nextstep': '',
        }, inplace=True)

    # raise an exception if any created dates are missing
    missing_created = result.loc[pd.isnull(result['created'])]
    if missing_created.shape[0] > 0:
        line = missing_created.index[0] + 2
        raise Exception(f"Missing creation time on line {line}")

    return result


def calculate_point_deltas():
    """ Return a time-series of points with times from the given column

    >>> calculate_point_deltas()
    3

    """
    return 3
