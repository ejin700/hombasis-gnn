from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn


def default_progressbar():
    # define our progress bar
    return Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    )
