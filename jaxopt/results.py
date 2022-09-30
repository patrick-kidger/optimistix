import equinox.internal as eqxi


class RESULTS(metaclass=eqxi.ContainerMeta):
  successful = ""
  max_steps_reached = "The maximum number of solver steps was reached. Try increasing `max_steps`."
  linear_singular = "The matrix for the linear solve was singular. Try using a linear solver that supports singular matrices, such as `SVD` or `CG`."
  nonlinear_divergence = "Nonlinear solve diverged."
