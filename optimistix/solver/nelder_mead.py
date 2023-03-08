from typing import Callable

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import ArrayLike, Float, PyTree

from ..custom_types import Scalar
from ..minimise import AbstractMinimiser
from ..misc import max_norm
from ..solution import RESULTS


class _NMStats(eqx.Module):
    n_reflect: Scalar
    n_inner_contract: Scalar
    n_outer_contract: Scalar
    n_expand: Scalar
    n_shrink: Scalar


class _NelderMeadState(eqx.Module):
    simplex: PyTree
    f_simplex: PyTree
    indices: PyTree
    best: PyTree
    worst: PyTree
    second_worst: PyTree
    step: PyTree
    stats: _NMStats
    result: RESULTS


def _tree_where(cond, true_pytree, false_pytree):
    true_leaves, treedef = jtu.tree_flatten(true_pytree)
    false_leaves, treedef = jtu.tree_flatten(false_pytree)
    leaves = zip(true_leaves, false_leaves)
    out_leaves = [
        jnp.where(cond, true_leaf, false_leaf) for (true_leaf, false_leaf) in leaves
    ]
    return jtu.tree_unflatten(treedef, out_leaves)


def _check_simplex(y):
    if len(y.shape) == 2:
        is_simplex = True

    else:
        if len(y.shape) > 1:
            raise ValueError("y must be a PyTree with leaves of rank 0, 1 or 2.")
        is_simplex = False
    return is_simplex


class NelderMead(AbstractMinimiser):

    rtol: float
    atol: float
    reflection_const: Float[ArrayLike, ""] = jnp.array(1.0)
    expand_const: Float[ArrayLike, ""] = jnp.array(2.0)
    contract_const: Float[ArrayLike, ""] = jnp.array(0.5)
    shrink_const: Float[ArrayLike, ""] = jnp.array(0.5)
    norm: Callable = max_norm
    iters_per_terminate_check: int = 5
    rdelta: float = 5e-2
    adelta: float = 2.5e-4

    def init(self, problem, y, args, options):
        ###
        # WARNING: right now this is false! We are not using those defaults
        # our default constant values are those found in
        # "Implementing the Nelder-Mead Simplex Algorithm with Adaptive Parameters"
        # by Gau and Han. This is the choice of constants found in scipy, and scaled the
        # Nelder Mead constants based upon dimension (I strongly oppose their use
        # of the word "adaptive" to describe this choice).
        #
        # Typically, Nelder-Mead is parameterised by 4 parameters with typical values:
        # `alpha`: 1
        # `beta`: 2
        # `gamma`: 0.5
        # `sigma`: 0.5
        #
        # All constants are applied in the form centroid + const * (centroid - worst).
        # The typical parametrisation of Nelder-Mead is then
        # Reflection const: `alpha`
        # Expansion const: `alpha` * `beta`
        # Outer contraction const: `alpha` * `gamma`
        # Inner contraction const: -`gamma`
        # Shrink const: `sigma`
        # and indeed this is the parameterisation used in SciPy.
        #
        # Minpack chooses instead to parameterise these constants individually (with
        # inner and outer contraction differing by a factor of -1).
        # We choose the latter approach.
        ###

        if self.iters_per_terminate_check < 1:
            raise ValueError("`iters_per_termination_check` must at least one")

        is_simplex = jtu.tree_all((ω(y).call(_check_simplex)).ω)

        if is_simplex:
            simplex = y
        else:
            ###
            # The standard approach to creating the init simplex from a single vector
            # is to add a small constant times each unit vector to the initial vector.
            # The constant is different if the unit vector is 0 in the direction of the
            # unit vector. Just because this is standard, does not mean it's well
            # justified and there is certainly room to improve this init (there is
            # literature on this, but it does not seem popular with implementations
            # yet.) We choose to use an absolute and relative constant adelta, rdelta
            # and add rdelta y[i] + adelta to the ith unit vector for each unit
            # vector to create the simplex
            ###

            lens = ω(y).call(jnp.size).ω
            size = jtu.tree_reduce(lambda x, y: x + y, lens)
            leaves, treedef = jtu.tree_flatten(y)

            running_size = 0

            new_leaves = []

            for index, leaf in enumerate(leaves):
                leaf_size = jnp.size(leaf)
                broadcast_leaves = jnp.repeat(leaf, size + 1, axis=0)
                indices = jnp.arange(
                    running_size + 1, running_size + leaf_size + 1, dtype=jnp.int16
                )
                if len(leaf.shape) != 0:
                    indices = jnp.unravel_index(indices, shape=leaf.shape)
                    indices = jnp.stack(indices).T
                    broadcast_leaves = broadcast_leaves.at[indices].add(
                        self.adelta + self.rdelta * leaf[indices - running_size]
                    )
                else:
                    broadcast_leaves = broadcast_leaves.at[indices].add(
                        self.adelta + self.rdelta * leaf
                    )
                running_size = running_size + leaf_size
                new_leaves.append(broadcast_leaves)

            simplex = jtu.tree_unflatten(treedef, new_leaves)

        f_simplex = jax.vmap(lambda x: problem.fn(x, args))(simplex)

        (f_best,), (best_index,) = lax.top_k(-f_simplex, 1)
        f_best, best_index = -f_best, best_index

        f_vals, f_index = lax.top_k(f_simplex, 2)
        (f_worst, f_second_worst) = f_vals
        worst_index, _ = f_index
        leaves, treedef = jtu.tree_flatten(simplex)

        best = jtu.tree_unflatten(treedef, [leaf[best_index] for leaf in leaves])
        worst = jtu.tree_unflatten(treedef, [leaf[worst_index] for leaf in leaves])

        stats = _NMStats(
            jnp.array(0), jnp.array(0), jnp.array(0), jnp.array(0), jnp.array(0)
        )

        return _NelderMeadState(
            simplex=simplex,
            f_simplex=f_simplex,
            indices=(best_index, worst_index),
            best=(f_best, best),
            worst=(f_worst, worst),
            second_worst=f_second_worst,
            step=jnp.array(0),
            stats=stats,
            result=jnp.array(RESULTS.successful),
        )

    def step(self, problem, y, args, options, state):

        leaves, treedef = jtu.tree_flatten(state.simplex)

        best_index, worst_index = state.indices
        f_best, best = state.best
        f_worst, worst = state.worst
        f_second_worst = state.second_worst

        size = len(state.f_simplex)

        # TODO(Jason): Calculate the centroid mean based upon the
        # previous centroid mean. This helps scale in dimension.

        centroid = (
            (size / (size - 1))
            * (ω(jtu.tree_map(jnp.mean, state.simplex)) - ω(worst) / size)
        ).ω

        search_direction = (ω(centroid) - ω(worst)).ω

        reflection = (ω(centroid) + self.reflection_const * ω(search_direction)).ω
        f_reflection, _ = problem.fn(reflection, args)

        expand = f_reflection < f_best
        inner_contract = f_reflection > f_worst
        contract = f_reflection > f_second_worst
        outer_contract = jnp.invert(expand | contract)
        reflect = (f_reflection > f_best) & (f_reflection < f_second_worst)

        contract_const = jnp.where(
            inner_contract, -self.contract_const, self.contract_const
        )

        new_vertex = _tree_where(
            expand,
            (ω(centroid) + self.expand_const * ω(search_direction)).ω,
            reflection,
        )
        new_vertex = _tree_where(
            contract,
            (ω(centroid) + contract_const * ω(search_direction)).ω,
            new_vertex,
        )

        f_new_vertex, _ = problem.fn(new_vertex, args)
        f_new_vertex = f_new_vertex.reshape()
        shrink = f_new_vertex > f_worst
        leaves, treedef = jtu.tree_flatten(state.simplex)

        def update_simplex(best):
            new_vertex_leaves, _ = jtu.tree_flatten(new_vertex)

            new_simplex_leaves = []
            for leaf, new_leaf in zip(leaves, new_vertex_leaves):
                new_simplex_leaves.append(leaf.at[worst_index].set(new_leaf))

            f_simplex = state.f_simplex.at[worst_index].set(f_new_vertex)

            simplex = jtu.tree_unflatten(treedef, new_simplex_leaves)

            return f_simplex, simplex

        def shrink_simplex(best):
            diff_best = (ω(state.simplex) - ω(best)).ω
            simplex = (ω(best) + self.shrink_const * ω(diff_best)).ω
            f_simplex, _ = eqx.filter_vmap(problem.fn)(simplex, args)
            return f_simplex, simplex

        f_simplex, simplex = lax.cond(
            shrink,
            shrink_simplex,
            update_simplex,
            best,
        )

        ###
        # TODO(Jason): only 1 value is updated when not shrinking. This implies
        # that in most cases, rather than do a top_k search in log time, we can
        # just compare f_next_vector and f_best and choose best between those two
        # in constant time. Implement this. A similar thing could likely be done with
        # worst and second worst, with recomputation occuring only when f_new < f_worst
        # but f_new > f_second_worst (otherwise, there will have been a shrink).
        ###

        (f_best,), (best_index,) = lax.top_k(-f_simplex, 1)
        f_best, best_index = -f_best, best_index

        f_vals, f_index = lax.top_k(f_simplex, 2)
        (f_worst, f_second_worst) = f_vals
        worst_index, _ = f_index

        best = jtu.tree_unflatten(treedef, [leaf[best_index] for leaf in leaves])
        worst = jtu.tree_unflatten(treedef, [leaf[worst_index] for leaf in leaves])

        stats = _NMStats(
            state.stats.n_reflect + jnp.where(reflect, 1, 0),
            state.stats.n_expand + jnp.where(expand, 1, 0),
            state.stats.n_inner_contract + jnp.where(inner_contract, 1, 0),
            state.stats.n_outer_contract + jnp.where(outer_contract, 1, 0),
            state.stats.n_shrink + jnp.where(shrink, 1, 0),
        )

        new_state = _NelderMeadState(
            simplex=simplex,
            f_simplex=f_simplex,
            indices=(best_index, worst_index),
            best=(f_best, best),
            worst=(f_worst, worst),
            second_worst=f_second_worst,
            step=state.step + 1,
            stats=stats,
            result=state.result,
        )

        is_simplex = jtu.tree_all((ω(y).call(_check_simplex)).ω)
        if is_simplex:
            out = simplex
        else:
            out = best

        return out, new_state, None

    def terminate(self, problem, y, args, options, state):

        f_best, best_index = lax.top_k(-state.f_simplex.T, 1)
        best_index = best_index.flatten()[0]
        f_best = -f_best.flatten()
        leaves, treedef = jtu.tree_flatten(state.simplex)
        best = jtu.tree_unflatten(treedef, [leaf[best_index] for leaf in leaves])

        x_scale = (self.atol + self.rtol * ω(best).call(jnp.abs)).ω
        f_scale = (self.atol + self.rtol * ω(f_best).call(jnp.abs)).ω
        x_diff = ω((ω(state.simplex) - ω(best)).ω).call(jnp.abs).ω
        x_diff = (ω(x_diff) / ω(x_scale)).ω
        x_conv = self.norm(x_diff)
        # TODO(Jason): look for more principled algorithmic improvements to this
        # rather ad hoc criteria.
        x_conv = x_conv < 1e-1
        f_diff = ω((ω(state.f_simplex) - ω(f_best)).ω).call(jnp.abs).ω
        f_diff = (ω(f_diff) / ω(f_scale)).ω
        f_conv = self.norm(f_diff) < 1e-1

        ###
        # minpack does a further test here where it takes for each unit vector e_i a
        # perturbation "delta" and asserts that f(x + delta e_i) > f(x) and
        # f(x - delta e_i) > f(x). ie. a rough assertion that it is indeed a local
        # minimum. If it fails the algo resets completely. thus process scales as
        # O(dim T(f)), where T(f) is the cost of evaluating f.
        ###

        converged = x_conv & f_conv
        diverged = jnp.any(jnp.invert(jnp.isfinite(f_conv)))
        terminate = converged | diverged
        result = jnp.where(diverged, RESULTS.nonlinear_divergence, RESULTS.successful)
        return terminate, result
        # Q: How to perform check only every k steps?
        # terminate_args = (problem, y, args, options, state)
        # lax.cond(
        # state.step % self.iters_per_terminate_check != 0,
        # pass_terminate,
        # terminate_internal,
        # terminate_args
        # )
