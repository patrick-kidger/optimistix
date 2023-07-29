import matplotlib.pyplot as plt
import numpy as np


def quadratic_bowl(x, y):
    return x**2 + 0.8 * y**2 - 0.3 * x * y


x_min, x_max = -10, 10
y_min, y_max = -10, 10
x_values, y_values = np.meshgrid(
    np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
)
z_values = quadratic_bowl(x_values, y_values)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x_values, y_values, z_values, cmap="coolwarm")
plt.xticks([-10, -5, 0, 5, 10])
plt.yticks([-10, -5, 0, 5, 10])
ax.set_zticks([0, 50, 100, 150, 200])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.set_xlabel("x", labelpad=-28)
ax.set_ylabel("y", labelpad=-28)
ax.set_zlabel("f(x, y)\n\n\n\n\n\n\n\n", labelpad=-30)
ax.zaxis.set_rotate_label(False)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
fig.set_facecolor("none")
plt.savefig("quadratic_bowl.png", transparent=True, bbox_inches="tight")
