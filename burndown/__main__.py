"""burndown/burnup chart generaton

usage:
    burndown [options] <inputfile>... --output <file>
    burndown -h | --help

Options:
    -o <file> --output=<file>  output (figure) file
    -h --help                  Show this screen
    -u --burnup                Generate a burnup chart rather than a burndown
                               chart.
    --categories=<tags>        Use different colors for tasks with the given
                               tags (where tags is a comma separated list
                               without spaces)
"""

# This will be tested with a simple integration test, so don't include it in
# the code coverage for the unit tests.
if True:  # pragma: no cover
    from docopt import docopt
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    import burndown

    # parse the parameters
    args = docopt(__doc__, version="0.0.1")
    if args["--categories"]:
        categories = args["--categories"].split(",")
    else:
        categories = []

    # read the input files
    csvs = []
    holidays = []
    for filename in args["<inputfile>"]:
        if filename[-5:] == ".ical":
            with open(filename) as f:
                holidays += burndown.extract_holidays_from_ical(f.read())
        elif filename[-4:] == ".csv":
            csvs.append(burndown.read_csv(filename))
        else:
            raise ValueError(f"Unrecognized file extension for '{filename}'")
    tasks = pd.concat(csvs)

    # set up the scaled axes
    days = burndown.get_spanning_days(
        pd.Series(tasks["committed"].to_list() + tasks["completed"].to_list())
    )
    is_workday = burndown.is_workday(days, holidays)
    scaler = burndown.get_time_scaler(days, is_workday)

    # setup the tick labels
    max_labels = 32
    first_label_pos = scaler(days[0])
    last_label_pos = scaler(days[-1])
    min_label_spacing = (last_label_pos - first_label_pos) / (max_labels - 1)
    tick_labels = []
    prev_label_pos = -np.inf
    for day, workday in zip(days, is_workday):
        pos = scaler(day)
        if (
            workday
            and pos - prev_label_pos >= min_label_spacing
            and last_label_pos - pos >= min_label_spacing
        ) or day == days[-1]:
            tick_labels.append(day.strftime("%Y-%m-%d"))
            prev_label_pos = pos
        else:
            tick_labels.append("")

    # plot the figure
    fig = plt.figure(figsize=(10, 5))
    burndown.burndown_plot(
        tasks, categories, burnup=args["--burnup"], time_scaler=scaler
    )
    plt.grid(alpha=0.2)
    plt.legend(loc="upper left")
    plt.xticks(scaler(days), tick_labels, rotation=30, horizontalalignment="right")
    plt.savefig(args["--output"], bbox_inches="tight")
