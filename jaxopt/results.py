from diffrax.misc import ContainerMeta


class RESULTS(metaclass=ContainerMeta):
  successful = ""
  max_steps_reached = "The maximum number of solver steps was reached. Try increasing `max_steps`."
  singular = "The matrix for the linear solve was singular. Try using a linear solver that support singular matrices."
