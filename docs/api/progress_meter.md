# Progress meters

Progress meters display how far a solve has progressed. They are passed as the `progress_meter` argument to [`optimistix.minimise`][], [`optimistix.least_squares`][], [`optimistix.root_find`][], and [`optimistix.fixed_point`][].

??? abstract "`optimistix.AbstractProgressMeter`"

    ::: optimistix.AbstractProgressMeter
        options:
            members:
                - init
                - step
                - close

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
