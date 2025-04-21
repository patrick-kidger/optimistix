import jax.numpy as jnp

from .cutest import BT1, BT2, BT4, BT5, BT8, FLT


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
    # BT4(y0_iD=2),  # TODO: currently this one fails
    BT5(y0_iD=0),
    BT5(y0_iD=1),
    BT5(y0_iD=2),
    BT8(y0_iD=0),
    BT8(y0_iD=1),
    FLT(),
    # GENHS28(),  # TODO: IPOPTLike currently fails, other solvers not tested
)
