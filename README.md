<h1 align='center'>Optimistix</h1>

Optimistix is a [JAX](https://github.com/google/jax) library for nonlinear solvers: root finding, minimisation, fixed points, and least squares.

Features include:

- interoperable solvers: e.g. autoconvert root find problems to least squares problems, then solve using a minimisation algorithm.
- modular optimisers: e.g. use a BFGS quadratic bowl with a dogleg descent path with a trust region update.
- using a PyTree as the state.
- fast compilation and runtimes.
- interoperability with [Optax](https://github.com/deepmind/optax).
- all the benefits of working with JAX: autodiff, autoparallism, GPU/TPU support etc.

## Installation

```bash
pip install optimistix
```

Requires Python 3.9+ and JAX 0.4.14+ and [Equinox](https://github.com/patrick-kidger/equinox) 0.11.0+.

## Quick example

```python
import jax.numpy as jnp
import optimistix as optx

# Let's solve the ODE dy/dt=tanh(y(t)) with the implicit Euler method.
# We need to find y1 s.t. y1 = y0 + tanh(y1)dt.

y0 = jnp.array(1.)
dt = jnp.array(0.1)

def fn(y, args):
    return y0 + jnp.tanh(y) * dt

solver = optx.Newton(rtol=1e-5, atol=1e-5)
sol = optx.fixed_point(fn, solver, y0)
y1 = sol.value  # satisfies y1 == fn(y1)
```

## Finally

### JAX ecosystem

[jaxtyping](https://github.com/google/jaxtyping): type annotations for shape/dtype of arrays.

[Equinox](https://github.com/patrick-kidger/equinox): neural networks.

[Optax](https://github.com/deepmind/optax): first-order gradient (SGD, Adam, ...) optimisers.

[Diffrax](https://github.com/patrick-kidger/diffrax): numerical differential equation solvers.

[Lineax](https://github.com/google/lineax): linear solvers.

[BlackJAX](https://github.com/blackjax-devs/blackjax): probabilistic+Bayesian sampling.

[sympy2jax](https://github.com/google/sympy2jax): SymPy<->JAX conversion; train symbolic expressions via gradient descent.

[Eqxvision](https://github.com/paganpasta/eqxvision): computer vision models.

[Levanter](https://github.com/stanford-crfm/levanter): scalable+reliable training of foundation models (e.g. LLMs).

[PySR](https://github.com/milesCranmer/PySR): symbolic regression. (Non-JAX honourable mention!)

### Disclaimer

This is not an official Google product.

### Credit

Optimistix was primarily built by Jason Rader (@packquickly): [Twitter](https://twitter.com/packquickly); [GitHub](https://github.com/packquickly); [Website](https://www.packquickly.com/).
