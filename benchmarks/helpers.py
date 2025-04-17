import jax.numpy as jnp

from .cutest import BT1, BT2, BT4


unconstrained_problems = (
    # fn, y0, args, expected_result
    (
        lambda y, args: jnp.sum((y - 1) ** 2),  # dummy problem
        2 * jnp.ones(10),
        None,
        jnp.ones(10),
    ),
)


constrained_problems = (
    BT1(),
    BT2(),
    BT4(y0_iD=0),
    BT4(y0_iD=1),
    # BT4(y0_iD=2,  # TODO: currently this one fails
)
