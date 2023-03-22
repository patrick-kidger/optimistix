import functools as ft
from typing import Callable

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import PyTree

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
    best: PyTree
    worst: PyTree
    second_worst: PyTree
    step: PyTree
    stats: _NMStats
    result: RESULTS


def _tree_where(pred, true, false):
    keep = lambda a, b: jnp.where(pred, a, b)
    return jtu.tree_map(keep, true, false)


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
    norm: Callable = max_norm
    iters_per_terminate_check: int = 5
    rdelta: float = 5e-2
    adelta: float = 2.5e-4
    y0_is_simplex: bool = False

    def __post_init__(self):
        if self.iters_per_terminate_check < 1:
            raise ValueError("`iters_per_termination_check` must at least one")

    def init(self, problem, y, args, options):
        if self.y0_is_simplex:
            simplex = y
        else:
            #
            # The standard approach to creating the init simplex from a single vector
            # is to add a small constant times each unit vector to the initial vector.
            # The constant is different if the unit vector is 0 in the direction of the
            # unit vector. Just because this is standard, does not mean it's well
            # justified. We add rdelta * y[i] + adelta y[i] in the ith unit direction.
            #
            size = sum(jnp.size(x) for x in jtu.tree_leaves(y))
            leaves, treedef = jtu.tree_flatten(y)

            running_size = 0
            new_leaves = []

            for index, leaf in enumerate(leaves):
                leaf_size = jnp.size(leaf)
                broadcast_leaves = jnp.repeat(leaf, size + 1, axis=0)
                indices = jnp.arange(
                    running_size + 1, running_size + leaf_size + 1, dtype=jnp.int16
                )
                running_indices = jnp.unravel_index(
                    indices - running_size, shape=leaf.shape
                )
                broadcast_leaves = broadcast_leaves.at[indices].add(
                    self.adelta + self.rdelta * leaf[running_indices]
                )
                running_size = running_size + leaf_size
                new_leaves.append(broadcast_leaves)

            simplex = jtu.tree_unflatten(treedef, new_leaves)

        f_simplex = jax.vmap(lambda x: problem.fn(x, args))(simplex)

        (f_best,), (best_index,) = lax.top_k(-f_simplex, 1)
        f_best = -f_best

        (f_worst, f_second_worst), (worst_index, _) = lax.top_k(f_simplex, 2)
        best = ω(simplex)[best_index].ω
        worst = ω(simplex)[worst_index].ω

        stats = _NMStats(
            jnp.array(0), jnp.array(0), jnp.array(0), jnp.array(0), jnp.array(0)
        )

        return _NelderMeadState(
            simplex=simplex,
            f_simplex=f_simplex,
            best=(f_best, best, best_index),
            worst=(f_worst, worst, worst_index),
            second_worst=f_second_worst,
            step=jnp.array(0),
            stats=stats,
            result=jnp.array(RESULTS.successful),
        )

    def step(self, problem, y, args, options, state):
        #
        # The paper "Implementing the Nelder-Mead Simplex Algorithm with Adaptive
        # Parameters" by Gau and Han argues that the NM constants should be scaled
        # by the dimension of the problem. I strongly oppose their use of the word
        # "adaptive" to describe this choice.
        # This choice of constants is found in scipy
        #
        # This will later be replaced with the more general line search api.
        #
        dim = jnp.size(y)

        reflect_const = 1
        expand_const = 1 + 2 / dim
        contract_const = 0.75 - 1 / (2 * dim)
        shrink_const = 1 - 1 / dim

        f_best, best, best_index = state.best
        f_worst, worst, worst_index = state.worst
        f_second_worst = state.second_worst

        n_vertices = len(state.f_simplex)

        # TODO(RaderJason): Calculate the centroid mean based upon the
        # previous centroid mean. This helps scale in dimension.

        mean_first_axis = ft.partial(jnp.mean, axis=0)
        centroid = (
            (n_vertices / (n_vertices - 1))
            * ω(
                jtu.tree_map(
                    mean_first_axis, (ω(state.simplex) - ω(worst) / n_vertices).ω
                )
            )
        ).ω

        search_direction = (ω(centroid) - ω(worst)).ω

        reflection = (ω(centroid) + reflect_const * ω(search_direction)).ω

        def eval_new_vertices(vertex_carry, i):
            vertex, (f_vertex, _) = vertex_carry

            def g(f_vertex):
                expand = f_vertex < f_best
                inner_contract = f_vertex > f_worst
                contract = f_vertex > f_second_worst
                # outer_contract = jnp.invert(expand | contract)
                # reflect = (f_vertex > f_best) & (f_vertex < f_second_worst)
                signed_contract = jnp.where(
                    inner_contract, -contract_const, contract_const
                )

                new_vertex = _tree_where(
                    expand,
                    (ω(centroid) + expand_const * ω(search_direction)).ω,
                    vertex,
                )

                new_vertex = _tree_where(
                    contract,
                    (ω(centroid) + signed_contract * ω(search_direction)).ω,
                    new_vertex,
                )
                return new_vertex

            out = lax.cond(i == 1, g, lambda x: vertex, f_vertex)
            return (out, problem.fn(out, args)), None

        (new_vertex, (f_new_vertex, _)), _ = jax.lax.scan(
            eval_new_vertices, (reflection, (None, None)), jnp.arange(2)
        )
        shrink = f_new_vertex > f_worst

        def update_simplex(best, simplex):
            simplex = ω(simplex).at[worst_index].set(ω(new_vertex)).ω
            f_simplex = state.f_simplex.at[worst_index].set(f_new_vertex)

            return f_simplex, simplex

        def shrink_simplex(best, simplex):
            diff_best = (ω(simplex) - ω(best)).ω
            simplex = (ω(best) + shrink_const * ω(diff_best)).ω
            f_simplex, _ = jax.vmap(lambda x: problem.fn(x, args))(simplex)
            return f_simplex, simplex

        f_simplex, simplex = lax.cond(
            shrink,
            shrink_simplex,
            update_simplex,
            *(best, state.simplex),
        )

        #
        # TODO(RaderJason): only 1 value is updated when not shrinking. This implies
        # that in most cases, rather than do a top_k search in log time, we can
        # just compare f_next_vector and f_best and choose best between those two
        # in constant time. Implement this. A similar thing could likely be done with
        # worst and second worst, with recomputation occuring only when f_new < f_worst
        # but f_new > f_second_worst (otherwise, there will have been a shrink).
        #

        (f_best,), (best_index,) = lax.top_k(-f_simplex, 1)
        f_best, best_index = -f_best, best_index

        f_vals, f_index = lax.top_k(f_simplex, 2)
        (f_worst, f_second_worst) = f_vals
        worst_index, _ = f_index

        best = ω(simplex)[best_index].ω
        worst = ω(simplex)[worst_index].ω

        # stats = _NMStats(
        #     state.stats.n_reflect + jnp.where(reflect, 1, 0),
        #     state.stats.n_expand + jnp.where(expand, 1, 0),
        #     state.stats.n_inner_contract + jnp.where(inner_contract, 1, 0),
        #     state.stats.n_outer_contract + jnp.where(outer_contract, 1, 0),
        #     state.stats.n_shrink + jnp.where(shrink, 1, 0),
        # )
        stats = state.stats

        new_state = _NelderMeadState(
            simplex=simplex,
            f_simplex=f_simplex,
            best=(f_best, best, best_index),
            worst=(f_worst, worst, worst_index),
            second_worst=f_second_worst,
            step=state.step + 1,
            stats=stats,
            result=state.result,
        )

        if self.y0_is_simplex:
            out = simplex
        else:
            out = best

        return out, new_state, None

    def terminate(self, problem, y, args, options, state):
        #
        # TODO(RaderJason): only check terminate every k steps
        #
        (f_best,), (best_index,) = lax.top_k(-state.f_simplex, 1)
        f_best = -f_best

        best = ω(state.simplex)[best_index].ω

        x_scale = (self.atol + self.rtol * ω(best).call(jnp.abs)).ω
        f_scale = (self.atol + self.rtol * ω(f_best).call(jnp.abs)).ω

        x_diff = ((ω(state.simplex) - ω(best)).call(jnp.abs) / ω(x_scale)).ω
        x_conv = self.norm(x_diff) < 1

        f_diff = ((ω(state.simplex) - ω(best)).call(jnp.abs) / (f_scale)).ω
        f_conv = self.norm(f_diff) < 1

        #
        # minpack does a further test here where it takes for each unit vector e_i a
        # perturbation "delta" and asserts that f(x + delta e_i) > f(x) and
        # f(x - delta e_i) > f(x). ie. a rough assertion that it is indeed a local
        # minimum. If it fails the algo resets completely. thus process scales as
        # O(dim T(f)), where T(f) is the cost of evaluating f.
        #

        converged = x_conv & f_conv
        diverged = jnp.any(jnp.invert(jnp.isfinite(f_best)))
        terminate = converged | diverged
        result = jnp.where(diverged, RESULTS.nonlinear_divergence, RESULTS.successful)
        return terminate, result
