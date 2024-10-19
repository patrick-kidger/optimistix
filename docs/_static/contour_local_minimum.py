import jax.numpy as jnp
import matplotlib.pyplot as plt  # pyright: ignore


fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

# Create grid of points
x = jnp.linspace(-4, 4, 200)
y = jnp.linspace(-4, 4, 200)
X, Y = jnp.meshgrid(x, y)

# Calculate Z values
Z = (
    -jnp.exp(-0.35 * (X + 1) ** 2 - 0.45 * Y**2)
    - 0.4 * jnp.exp(-0.5 * (X - 2.5) ** 2 - (Y - 1) ** 2)
    + 2
)

# Create contour plot
plt.figure(figsize=(10, 8))
ax.contour(X, Y, Z, levels=30, cmap="viridis")
ax.axvline(1.25, color="firebrick", linestyle="--", label="upper bound on x")
ax.legend()

# Format plot and add labels
ax.set_xlim(-4, 4)
ax.set_ylim(-3, 4)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_yticks([])
ax.set_xticks([])
ax.spines[["right", "top"]].set_visible(False)

fig.savefig("contour_local_minimum.png", bbox_inches="tight")
