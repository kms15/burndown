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
    assert csv["nextstep"].iloc[0] == "do something"
    assert csv["notes"].iloc[2] == "secret notes"

    # for strings, empty strings should be used instead of "missing"
    assert csv["notes"].iloc[0] == ""
    assert csv["nextstep"].iloc[2] == ""

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


def test_burndown_timeseries():
    start_times = pd.to_datetime(
        pd.Series(
            [
                "2022-04-11T12:12Z",
                "2022-04-11T12:16Z",
                "",
                "2022-04-11T12:20Z",
                "2022-04-11T12:24Z",
            ]
        )
    )
    end_times = pd.to_datetime(
        pd.Series(
            [
                "2022-04-11T12:17Z",
                "",
                "",
                "2022-04-11T12:29Z",
                "2022-04-11T12:27Z",
            ]
        )
    )
    points = pd.Series([2, 3, 33, 7, 11])

    expected = pd.Series(
        data=[2, 5, 3, 10, 21, 10, 3],
        index=pd.to_datetime(
            [
                "2022-04-11T12:12Z",
                "2022-04-11T12:16Z",
                "2022-04-11T12:17Z",
                "2022-04-11T12:20Z",
                "2022-04-11T12:24Z",
                "2022-04-11T12:27Z",
                "2022-04-11T12:29Z",
            ]
        ),
    )

    # should support passing in three series
    result = burndown.burndown_timeseries(start_times, end_times, points)
    assert result.equals(expected)

    # should support passing in a dataframe
    result = burndown.burndown_timeseries(
        pd.DataFrame(
            {"committed": start_times, "completed": end_times, "points": points}
        )
    )
    assert result.equals(expected)

    # should support passing in just start times and points
    expected = pd.Series(
        data=[2, 5, 12, 23],
        index=pd.to_datetime(
            [
                "2022-04-11T12:12Z",
                "2022-04-11T12:16Z",
                "2022-04-11T12:20Z",
                "2022-04-11T12:24Z",
            ]
        ),
    )
    result = burndown.burndown_timeseries(start_times, points=points)
    assert result.equals(expected)


def test_with_and_without_tag():
    times = pd.to_datetime(
        [
            "2022-04-11T12:12Z",
            "2022-04-11T12:16Z",
            "2022-04-11T12:17Z",
            "2022-04-11T12:20Z",
            "2022-04-11T12:24Z",
            "2022-04-11T12:27Z",
            "2022-04-11T12:29Z",
        ]
    )
    points = pd.Series([1, 1, 2, 3, 5, 8, 13])
    notes = pd.Series(
        [
            "#foo",
            "#foo and then something",
            "ends with #foo",
            "no foo here #bar",
            "a#foo and #bar",
            "#foozle not foo",
            "have #foo, will #travel",
        ]
    )
    original = pd.DataFrame({"times": times, "points": points, "notes": notes})
    has_tag = pd.Series([True, True, True, False, True, False, True])

    result = burndown.with_tag(original, "foo")
    expected = original.loc[has_tag]
    assert result.equals(expected)

    result = burndown.without_tag(original, "foo")
    expected = original.loc[~has_tag]
    assert result.equals(expected)


def test_prepare_stackplot_df():

    points = [1, 2, 3, 5, 7, 11]
    committed = pd.to_datetime(
        [
            "2022-04-11T12:12Z",
            "2022-04-11T12:12Z",
            "",
            "2022-04-11T12:14Z",
            "2022-04-11T12:14Z",
            "2022-04-11T12:15Z",
        ]
    )
    completed = pd.to_datetime(
        [
            "2022-04-11T12:16Z",
            "2022-04-11T12:13Z",
            "",
            "",
            "2022-04-11T12:16Z",
            "2022-04-11T12:17Z",
        ]
    )
    notes = [
        "no #relevant tags",
        "#foo #manchu",
        "unscheduled",
        "#bar",
        "#bar",
        "#bar #triaged",
    ]

    tasks = pd.DataFrame(
        {
            "committed": committed,
            "completed": completed,
            "points": points,
            "notes": notes,
        },
        index=[1, 2, 4, 5, 1, 2],
    )

    expected_df = pd.DataFrame(
        {
            "completed": [0.0, 2.0, 2.0, 2.0, 10.0, 10.0],
            "uncategorized": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            "foo": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "bar": [0.0, 0.0, 12.0, 23.0, 16.0, 5.0],
        },
        index=pd.to_datetime(
            [
                "2022-04-11T12:12Z",
                "2022-04-11T12:13Z",
                "2022-04-11T12:14Z",
                "2022-04-11T12:15Z",
                "2022-04-11T12:16Z",
                "2022-04-11T12:17Z",
            ]
        ),
    )

    # start with the burnup plot version, since that's a superset of the
    # burndown plot.
    result_df = burndown.prepare_stackplot_df(
        tasks,
        ["foo", "bar"],
        burnup=True,
    )
    assert result_df.equals(expected_df)

    # next, check the ordinary burndown case
    result_df = burndown.prepare_stackplot_df(
        tasks,
        ["foo", "bar"],
        burnup=False,
    )
    assert result_df.equals(expected_df.drop("completed", axis=1))


def test_burndown_plot():

    # start with the burnup plot version, since that's a superset of the
    burnup_expected = True

    dummy_df = pd.DataFrame(
        {"completed": [1, 2], "x": [3, 4], "y": [5, 6]}, index=[7, 8]
    )

    # burndown plot.
    expected_kwargs = {"baz": 3, "foozle": 4}
    expected_x = dummy_df.index
    expected_y = dummy_df.transpose()
    expected_labels = dummy_df.columns
    expected_colors = [
        (0, 0, 0, 0),
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

    stackplot_called = False

    def moc_stackplot(x, y, *args, labels, colors, **kwargs):
        nonlocal stackplot_called
        stackplot_called = True

        assert expected_x.equals(x)
        assert np.array_equal(expected_y, y)
        assert np.array_equal(labels, expected_labels)
        assert colors == expected_colors
        assert kwargs == expected_kwargs

        return "dummy_result"

    def moc_prepare_df(
        df,
        category_tags,
        burnup=False,
    ):
        assert df == "dummy"
        assert category_tags == ["foo", "bar"]
        assert burnup == burnup_expected
        return dummy_df

    assert "dummy_result" == burndown.burndown_plot(
        "dummy",
        ["foo", "bar"],
        burnup=burnup_expected,
        baz=3,
        foozle=4,
        stackplot_function=moc_stackplot,
        prepare_df_function=moc_prepare_df,
    )
    assert stackplot_called

    # next, check the ordinary burndown case
    burnup_expected = False
    expected_colors = expected_colors[1:]

    stackplot_called = False
    assert "dummy_result" == burndown.burndown_plot(
        "dummy",
        ["foo", "bar"],
        baz=3,
        foozle=4,
        stackplot_function=moc_stackplot,
        prepare_df_function=moc_prepare_df,
    )
    assert stackplot_called


def test_get_spanning_days():
    result = burndown.get_spanning_days(
        pd.to_datetime(
            [
                "2022-04-12T08:00Z",
                "2022-04-13T11:00Z",
                "2022-04-15T13:00Z",
                "2022-04-16T13:00Z",
            ]
        )
    )

    assert (
        result.to_list()
        == pd.to_datetime(
            [
                "2022-04-12T00:00Z",
                "2022-04-13T00:00Z",
                "2022-04-14T00:00Z",
                "2022-04-15T00:00Z",
                "2022-04-16T00:00Z",
                "2022-04-17T00:00Z",
            ]
        ).to_list()
    )


def test_is_weekday():
    result = burndown.is_weekday(
        pd.to_datetime(
            [
                "2022-06-14T00:00Z",
                "2022-06-15T00:00Z",
                "2022-06-16T00:00Z",
                "2022-06-17T00:00Z",
                "2022-06-18T00:00Z",
                "2022-06-19T00:00Z",
                "2022-06-20T00:00Z",
                "2022-06-21T00:00Z",
            ]
        )
    )
    assert list(result) == [
        True,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
    ]


def test_get_time_scaler():
    spanning_days = pd.to_datetime(
        [
            "2022-06-14T00:00Z",
            "2022-06-15T00:00Z",
            "2022-06-16T00:00Z",
            "2022-06-17T00:00Z",
            "2022-06-18T00:00Z",
            "2022-06-19T00:00Z",
        ]
    )
    is_workday = [
        True,
        False,
        False,
        True,
        True,
        False,
    ]

    time_scaler = burndown.get_time_scaler(spanning_days, is_workday, 1 / 16)

    assert list(time_scaler(spanning_days)) == [
        0,
        1.0,
        1 + 1 / 16,
        1 + 2 / 16,
        2 + 2 / 16,
        3 + 2 / 16,
    ]
    assert list(
        time_scaler(
            pd.to_datetime(
                [
                    "2022-06-14T12:00Z",
                    "2022-06-15T12:00Z",
                    "2022-06-16T12:00Z",
                ]
            )
        )
    ) == [0.5, 1 + 1 / 32, 1 + 3 / 32]
