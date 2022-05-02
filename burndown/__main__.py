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
    from matplotlib import pyplot as plt
    import burndown

    # parse the parameters
    args = docopt(__doc__, version="0.0.1")
    csvs = [burndown.read_csv(filename) for filename in args["<inputfile>"]]
    tasks = pd.concat(csvs)
    if args["--categories"]:
        categories = args["--categories"].split(",")
    else:
        categories = []

    # plot the figure
    fig = plt.figure(figsize=(10, 5))
    burndown.burndown_plot(tasks, categories, burnup=args["--burnup"])
    plt.legend()
    plt.savefig(args["--output"], bbox_inches="tight")
