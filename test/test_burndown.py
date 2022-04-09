# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2022, Kendrick Shaw

import burndown
import pandas as pd
import pytest
import numpy as np

def test_read_csv():
    csv = burndown.read_csv("test/read_test.csv")

    # should have the right number of rows and columns
    assert csv.shape[0] == 3
    assert csv.shape[1] == 7

    # date columns should act like dates
    assert (csv["created"].iloc[2] - csv["created"].iloc[0]).total_seconds() == 120
    assert (csv["completed"].iloc[2] - csv["committed"].iloc[0]).total_seconds() == 240
    assert pd.isnull(csv["completed"].iloc[0])

    # remaining columns should exist
    assert csv["points"].iloc[1] == 4
    assert csv["nextstep"].iloc[0] == 'do something'
    assert csv["notes"].iloc[2] == 'secret notes'

    # for strings, empty strings should be used instead of "missing"
    assert csv["notes"].iloc[0] == ''
    assert csv["nextstep"].iloc[2] == ''

    # should raise error if creation date does not exist
    with pytest.raises(Exception) as e:
        burndown.read_csv("test/missing_created_test.csv")
    assert str(e.value) == "Missing creation time on line 3"

    # should raise an error for bad data in a date column
    with pytest.raises(Exception) as e:
        burndown.read_csv("test/bad_date_format.csv")

    # should raise an error if points value is not a number
    with pytest.raises(Exception) as e:
        burndown.read_csv("test/non_numeric_points.csv")

def test_calculate_point_deltas():
    pass
