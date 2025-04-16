import jax.numpy as jnp


unconstrained_problems = (
    # fn, y0, args, expected_result
    (
        lambda y, args: jnp.sum((y - 1) ** 2),  # dummy problem
        2 * jnp.ones(10),
        None,
        jnp.ones(10),
    ),
)
