# Progress meters

As a solve progresses, progress meters offer the ability to have some kind of output indicating how far along the solve has progressed. For example, to display a text output every now and again, or to fill a [tqdm](https://github.com/tqdm/tqdm) progress bar.

??? abstract "`optimistix.AbstractProgressMeter`"

    ::: optimistix.AbstractProgressMeter
        options:
            members: false

---

::: optimistix.NoProgressMeter
    options:
        members:
            - __init__

::: optimistix.TextProgressMeter
    options:
        members:
            - __init__

::: optimistix.TqdmProgressMeter
    options:
        members:
            - __init__
