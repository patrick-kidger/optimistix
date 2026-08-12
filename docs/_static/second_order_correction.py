import jax.numpy as jnp
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]


# Definitions
def paraboloid(x, y):
    return (x + 1) ** 2 + (y + 1) ** 2


# Setup
fig, ax = plt.subplots(1, 1)
limits = (-2.3, 3)

# Make contour plot
x = jnp.linspace(*limits, 100)
y = jnp.linspace(*limits, 100)
X, Y = jnp.meshgrid(x, y)
Z = paraboloid(X, Y)
ax.contourf(X, Y, Z, levels=40, alpha=0.5, cmap="Grays", vmin=0, vmax=10)

# Add constraint
constraint_color = "olive"
circle = plt.Circle((0, 0), 1, color=constraint_color, fill=False, label="Constraint")
ax.add_patch(circle)
ax.axvline(
    x=1,
    color=constraint_color,
    linestyle="-.",
    label="Null space of Jac(c(x))",
    alpha=0.75,
    linewidth=1,
)

# Current point
current_point = jnp.array([1.0, 0.0])
ax.plot(*current_point, "o", color="firebrick", label="Current point", zorder=10)

# Steps
gradient_tip = jnp.array([-3, -2])
normal_tip = jnp.array([0.8, -1.3])
corrected_tip = jnp.array([-0.85, -1.25])


def make_arrow(tip, color, label):
    props = dict(arrowstyle="<-", color=color)
    ax.annotate(label, xytext=tip, xy=current_point, arrowprops=props)


# make_arrow(0.7 * gradient_tip, "firebrick", "grad")
make_arrow(normal_tip, "blue", "step")
make_arrow(corrected_tip, "darkviolet", "corrected_step")

# Layouting
ax.set_xlim(limits)
ax.set_ylim(limits)
ax.set_aspect("equal", adjustable="box")

ticks = jnp.array([-1.5, 0, 1.5, 3])
ax.set_xticks(ticks)
ax.set_yticks(ticks)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Second-order correction")

# Save the figure
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("second_order_correction.png", dpi=600)
