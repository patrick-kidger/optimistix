import diffrax as dfx
import jax.numpy as jnp
import matplotlib.pyplot as plt  # pyright: ignore


def van_der_pol(t, state, damping_factor):
    # The Van der Pol oscillator, expressed as a system of two first-order ODEs, as
    # given, for example, here:
    # https://en.wikipedia.org/wiki/Van_der_Pol_oscillator#Two-dimensional_form
    x, y = state
    dx = y
    dy = damping_factor * (1 - x**2) * y - x
    return jnp.array([dx, dy])


def solve(initial_state, damping_factor, timepoints):
    term = dfx.ODETerm(van_der_pol)
    t0 = timepoints[0]
    t1 = timepoints[-1]
    saveat = dfx.SaveAt(ts=timepoints)
    solution = dfx.diffeqsolve(
        term,
        dfx.Tsit5(),
        t0,
        t1,
        0.01,
        initial_state,
        damping_factor,
        saveat=saveat,
    )
    return solution.ys


fig, ax = plt.subplots(1, 1, figsize=(8, 5))

timepoints = jnp.linspace(0, 10, 1000)
damping_factor = 1.0
for x in range(-4, 5):
    for y in range(-4, 5):
        initial_state = jnp.array([x, y])
        trajectory = solve(initial_state, damping_factor, timepoints)
        tx, ty = jnp.asarray(trajectory).T
        ax.plot(tx, ty, color="lightseagreen", alpha=0.2)

origin = jnp.array([0, 0])
ax.plot(*origin, "+", label="Unstable steady state", color="darkred")
trajectory = solve(origin - 0.01, damping_factor, timepoints)
tx, ty = jnp.asarray(trajectory).T
ax.plot(tx, ty, color="orangered", marker="v", markevery=[-1], alpha=0.65)
ax.set_title("Van der Pol Oscillator for $\mu=1$")  # pyright: ignore

limits = (-4, 4)
ax.set_aspect("equal")
ax.set_xlim(*limits)
ax.set_ylim(*limits)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xticks([-4, -2, 0, 2, 4])
ax.set_yticks([-4, -2, 0, 2, 4])
ax.legend()

plt.subplots_adjust(left=0.25, right=0.75)  # Pad to adjust size in docs
fig.tight_layout()
fig.savefig("van_der_pol.png", dpi=300)
