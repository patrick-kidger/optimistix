<h1 align='center'>Optimistix</h1>

Optimistix is a [JAX](https://github.com/google/jax) library for nonlinear solvers: root finding, minimisation, fixed points, and least squares.

Features include:

- interoperable solvers: e.g. autoconvert root find problems to least squares problems, then solve using a minimisation algorithm.
- modular optimisers: e.g. use a BFGS quadratic bowl with a dogleg descent path with a trust region update.
- using a PyTree as the state.
- fast compilation and runtimes.
- interoperability with [Optax](https://github.com/deepmind/optax).
- all the benefits of working with JAX: autodiff, autoparallelism, GPU/TPU support etc.

## Installation

```bash
pip install optimistix
```

Requires Python 3.10+ and JAX 0.4.38+ and [Equinox](https://github.com/patrick-kidger/equinox) 0.11.11+.

## Documentation

Available at [https://docs.kidger.site/optimistix](https://docs.kidger.site/optimistix).

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

## Citation

If you found this library to be useful in academic work, then please cite: ([arXiv link](https://arxiv.org/abs/2402.09983))

```bibtex
@article{optimistix2024,
    title={Optimistix: modular optimisation in JAX and Equinox},
    author={Jason Rader and Terry Lyons and Patrick Kidger},
    journal={arXiv:2402.09983},
    year={2024},
}
```

## See also: other libraries in the JAX ecosystem

**Always useful**  
[Equinox](https://github.com/patrick-kidger/equinox): neural networks and everything not already in core JAX!  
[jaxtyping](https://github.com/patrick-kidger/jaxtyping): type annotations for shape/dtype of arrays.  

**Deep learning**  
[Optax](https://github.com/deepmind/optax): first-order gradient (SGD, Adam, ...) optimisers.  
[Orbax](https://github.com/google/orbax): checkpointing (async/multi-host/multi-device).  
[Levanter](https://github.com/stanford-crfm/levanter): scalable+reliable training of foundation models (e.g. LLMs).  

**Scientific computing**  
[Diffrax](https://github.com/patrick-kidger/diffrax): numerical differential equation solvers.  
[Lineax](https://github.com/patrick-kidger/lineax): linear solvers.  
[BlackJAX](https://github.com/blackjax-devs/blackjax): probabilistic+Bayesian sampling.  
[sympy2jax](https://github.com/patrick-kidger/sympy2jax): SymPy<->JAX conversion; train symbolic expressions via gradient descent.  
[PySR](https://github.com/milesCranmer/PySR): symbolic regression. (Non-JAX honourable mention!)  

**Awesome JAX**  
[Awesome JAX](https://github.com/n2cholas/awesome-jax): a longer list of other JAX projects.  

## Credit

Optimistix was primarily built by Jason Rader (@packquickly): [Twitter](https://twitter.com/packquickly); [GitHub](https://github.com/packquickly); [Website](https://www.packquickly.com/).
