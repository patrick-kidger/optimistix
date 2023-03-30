import functools as ft
from typing import Callable, Tuple

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import ArrayLike, Bool, PyTree

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
    """
    Information to update and store the simplex of the Nelder Mead update. If
    `dim` is the dimension of the problem, we expect there to be
    `n_vertices` = `dim` + 1 vertices. We expect the leading axis of each leaf
    to be of len `n_vertices`, and the sum of the rest of the axes of all leaves
    together to be `dim`.

    - `simplex`: a PyTree with leading axis of leaves `n_vertices` and sum of the
        rest of the axes of all leaves `dim`.
    - `f_simplex`: a 1-dimensional array of size `n_vertices`.
        The values of the problem function evaluated on each vertex of
        simplex.
    - `best`: A tuple of shape (Scalar, PyTree, Scalar). The tuple contains
        (`f(best_vertex)`, `best_vertex`, index of `best_vertex`) where
        `best_vertex` is the vertex which minimises `f` among all vertices in
        `simplex`.
    - `worst`: A tuple of shape (Scalar, PyTree, Scalar). The tuple contains
        (`f(worst_vertex)`, `worst_vertex`, index of `worst_vertex`) where
        `worst_vertex` is the vertex which maximises `f` among all vertices in
        `simplex`.
    -`second_worst`: A scalar, which is `f(second_worst_vertex)` where
        `second_worst_vertex` is the vertex which maximises `f` among all vertices
        in `simplex` with `worst_vertex` removed.
    - `step`: A scalar. How many steps have been taken so far.
    - `stats`: A _NMStats PyTree. This tracks information about the Nelder Mead
        algorithm. Specifically, how many times each of the operations reflect,
        expand, inner contract, outer contract, and shrink are performed.
    - `result`: a RESULTS object which indicates if we have diverged during the
        course of optimisation.
    - `first_passk`: A bool which indicates if this is the first call to Nelder Mead
        which allows for extra setup. This ultimately exists to save on compilation
        time.
    """

    simplex: PyTree
    f_simplex: PyTree
    best: Tuple[Scalar, PyTree, Scalar]
    worst: Tuple[Scalar, PyTree, Scalar]
    second_worst: Scalar
    step: Scalar
    stats: _NMStats
    result: RESULTS
    first_pass: Bool[ArrayLike, ""]


def _tree_where(pred, struct, true, false):
    keep = lambda a, b: jnp.where(pred, a, b)
    return jtu.tree_map(lambda x, y, z: keep(y[...], z[...]), struct, true, false)


def _update_stats(
    stats,
    reflect=False,
    inner_contract=False,
    outer_contract=False,
    expand=False,
    shrink=False,
) -> _NMStats:
    return _NMStats(
        stats.n_reflect + jnp.where(reflect, 1, 0),
        stats.n_inner_contract + jnp.where(inner_contract, 1, 0),
        stats.n_outer_contract + jnp.where(outer_contract, 1, 0),
        stats.n_expand + jnp.where(expand, 1, 0),
        stats.n_shrink + jnp.where(shrink, 1, 0),
    )


class NelderMead(AbstractMinimiser):

    rtol: float
    atol: float
    norm: Callable = max_norm
    rdelta: float = 5e-2
    adelta: float = 2.5e-4

    def init(self, problem, y, args, options):
        try:
            y0_simplex = options["y0_simplex"]
        except KeyError:
            y0_simplex = False

        if y0_simplex:
            simplex = y
            leaves, treedef = jtu.tree_flatten(simplex)
            y_dtype = leaves[0].dtype
            leaf_vertices = [leaf.shape[0] for leaf in leaves]
            sizes = [jnp.size(leaf[0]) for leaf in leaves]
            n_vertices = leaf_vertices[0]
            size = sum(sizes)
            if n_vertices != size + 1:
                raise ValueError(
                    f"The PyTree must form a valid simplex. Got \
                {n_vertices} vertices but dimension {size}."
                )
            if any(n_vertices != leaf_shape for leaf_shape in leaf_vertices[1:]):
                raise ValueError(
                    "The PyTree must form a valid simplex. \
                    Got different leading dimension (number of vertices) \
                    for each leaf"
                )
        else:
            #
            # The standard approach to creating the init simplex from a single vector
            # is to add a small constant times each unit vector to the initial vector.
            # The constant is different if the unit vector is 0 in the direction of the
            # unit vector. Just because this is standard, does not mean it's well
            # justified. We add rdelta * y[i] + adelta y[i] in the ith unit direction.
            #
            size = sum(jnp.size(x) for x in jtu.tree_leaves(y))
            n_vertices = size + 1
            leaves, treedef = jtu.tree_flatten(y)
            y_dtype = leaves[0].dtype

            running_size = 0
            new_leaves = []

            for index, leaf in enumerate(leaves):
                leaf_size = jnp.size(leaf)
                broadcast_leaves = jnp.repeat(leaf[None, ...], size + 1, axis=0)
                indices = jnp.arange(
                    running_size + 1, running_size + leaf_size + 1, dtype=jnp.int16
                )
                relative_indices = jnp.unravel_index(
                    indices - running_size, shape=leaf.shape
                )
                indices = jnp.unravel_index(indices, shape=broadcast_leaves.shape)
                broadcast_leaves = broadcast_leaves.at[indices].add(
                    self.adelta + self.rdelta * leaf[relative_indices]
                )
                running_size = running_size + leaf_size
                new_leaves.append(broadcast_leaves)

            simplex = jtu.tree_unflatten(treedef, new_leaves)

        try:
            f_dtype = options["struct"].dtype
        except KeyError:
            f_dtype = y_dtype

        f_simplex = jnp.zeros(n_vertices, dtype=f_dtype)
        # Shrink will be called the first time step is called, so remove one from stats
        # at init.
        stats = _NMStats(
            jnp.array(0), jnp.array(0), jnp.array(0), jnp.array(0), jnp.array(-1)
        )

        return _NelderMeadState(
            simplex=simplex,
            f_simplex=f_simplex,
            best=(jnp.array(0.0), ω(simplex)[0].ω, jnp.array(0, dtype=jnp.int32)),
            worst=(jnp.array(0.0), ω(simplex)[0].ω, jnp.array(0, dtype=jnp.int32)),
            second_worst=jnp.array(0.0),
            step=jnp.array(0),
            stats=stats,
            result=jnp.array(RESULTS.successful),
            first_pass=jnp.array(True),
        )

    def step(self, problem, y, args, options, state):
        # This will later be replaced with the more general line search api.
        reflect_const = 2
        expand_const = 3
        out_const = 1.5
        in_const = 0.5
        shrink_const = 0.5

        f_best, best, best_index = state.best
        f_worst, worst, worst_index = state.worst
        f_second_worst = state.second_worst
        stats = state.stats

        (n_vertices,) = state.f_simplex.shape

        def init_step(state):
            simplex = state.simplex
            f_new_vertex = jnp.array(1.0, dtype=state.f_simplex.dtype)
            # f_worst is 0., set this so that
            # shrink = f_new_simplex > f_worst is True
            return (
                state,
                simplex,
                ω(simplex, structure=y)[0].ω,
                f_new_vertex,
                state.stats,
            )

        def main_step(state):
            # TODO(raderj): Calculate the centroid and search dir based upon
            # the prior one.
            simplex_sum = lambda x: jtu.tree_map(ft.partial(jnp.sum, axis=0), x)

            search_direction = (
                ((ω(state.simplex, structure=y) - ω(worst)) / (n_vertices - 1))
                .call(simplex_sum)
                .ω
            )

            reflection = (ω(worst) + reflect_const * ω(search_direction)).ω

            def eval_new_vertices(vertex_carry, i):
                vertex, (f_vertex, _), stats = vertex_carry

                def internal_eval(f_vertex, stats):
                    expand = f_vertex < f_best
                    inner_contract = f_vertex > f_worst
                    contract = f_vertex > f_second_worst
                    outer_contract = jnp.invert(expand | contract)
                    reflect = (f_vertex > f_best) & (f_vertex < f_second_worst)
                    contract_const = jnp.where(inner_contract, in_const, out_const)
                    new_vertex = _tree_where(
                        expand,
                        (ω(worst) + expand_const * ω(search_direction)).ω,
                        (ω(worst) + expand_const * ω(search_direction)).ω,
                        vertex,
                    )

                    new_vertex = _tree_where(
                        contract,
                        (ω(worst) + contract_const * ω(search_direction)).ω,
                        (ω(worst) + contract_const * ω(search_direction)).ω,
                        new_vertex,
                    )
                    stats = _update_stats(
                        stats, reflect, inner_contract, outer_contract, expand
                    )
                    return new_vertex, stats

                out, stats = lax.cond(
                    i == 1,
                    internal_eval,
                    lambda x, y: (vertex, stats),
                    *(f_vertex, stats),
                )

                return (out, problem.fn(out, args), stats), None

            #
            # We'd like to avoid making two calls to problem.fn in the step to
            # avoid compiling the potentially large problem.fn twice. Instead, we
            # wrap the two evaluations in a single scan and pass different inputs
            # each time.
            #
            # the first iteration of scan will have lax.cond(False) and will return
            # f(reflection) for use in the second iteration which calls internal_eval
            # which uses f(reflection) to determine the next vertex, and returns the
            # vertex along with f(new_vertex) and stats.
            #
            # TODO(raderj): pull out this entire thing into line search api.
            #
            try:
                aux_struct = options["aux_struct"]
            except KeyError:
                aux_struct = None

            (new_vertex, (f_new_vertex, aux), stats), _ = lax.scan(
                eval_new_vertices,
                (reflection, (jnp.array(0.0), aux_struct), state.stats),
                jnp.arange(2),
            )

            return (state, state.simplex, new_vertex, f_new_vertex, stats)

        state, simplex, new_vertex, f_new_vertex, stats = lax.cond(
            state.first_pass, init_step, main_step, state
        )
        new_best = f_new_vertex < f_best
        best = _tree_where(new_best, new_vertex, new_vertex, best)
        f_best = jnp.where(new_best, f_new_vertex, f_best)

        #
        # On the initial call, f_worst is initialized to 0. and f_new_vertex
        # is initialized to 1. This is to deliberately set shrink = True
        # and call shrink_simplex. This lets us call vmap(problem.fn) only in
        # shrink simplex, which reduces compile time
        #
        shrink = f_new_vertex > f_worst
        stats = _update_stats(stats, shrink=shrink)

        def shrink_simplex(best, new_vertex, simplex, first_pass):
            shrink_simplex = ω(simplex, structure=best).at[...].add(-ω(best))
            shrink_simplex = ω(simplex, structure=best).at[...].multiply(shrink_const)
            shrink_simplex = ω(simplex, structure=best).at[...].add(ω(best)).ω

            simplex = _tree_where(first_pass, simplex, simplex, shrink_simplex)
            # Q: Can we avoid computing this?
            unwrapped_simplex = ω(simplex, structure=y).call(lambda x: x[...]).ω
            f_simplex, _ = jax.vmap(lambda x: problem.fn(x, args))(unwrapped_simplex)
            return f_simplex, simplex

        def update_simplex(best, new_vertex, simplex, first_pass):
            simplex = (
                ω(simplex, structure=new_vertex).at[worst_index].set(ω(new_vertex)).ω
            )
            f_simplex = state.f_simplex.at[worst_index].set(f_new_vertex)

            return f_simplex, simplex

        f_new_simplex, new_simplex = lax.cond(
            shrink,
            shrink_simplex,
            update_simplex,
            *(best, new_vertex, state.simplex, state.first_pass),
        )
        #
        # TODO(raderj): only 1 value is updated when not shrinking. This implies
        # that in most cases, rather than do a top_k search in log time, we can
        # just compare f_next_vector and f_best and choose best between those two
        # in constant time. Implement this. A similar thing could likely be done with
        # worst and second worst, with recomputation occuring only when f_new < f_worst
        # but f_new > f_second_worst (otherwise, there will have been a shrink).
        #
        (f_best,), (best_index,) = lax.top_k(-f_new_simplex, 1)
        f_best = -f_best

        f_vals, (worst_index, _) = lax.top_k(f_new_simplex, 2)
        (f_worst, f_second_worst) = f_vals

        best = ω(simplex, structure=y)[best_index].ω
        worst = ω(simplex, structure=y)[worst_index].ω

        new_state = _NelderMeadState(
            simplex=new_simplex,
            f_simplex=f_new_simplex,
            best=(f_best, best, best_index),
            worst=(f_worst, worst, worst_index),
            second_worst=f_second_worst,
            step=state.step + 1,
            stats=stats,
            result=state.result,
            first_pass=jnp.array(False),
        )
        try:
            y0_simplex = options["y0_simplex"]
        except KeyError:
            y0_simplex = False

        if y0_simplex:
            out = simplex
        else:
            out = best

        return out, new_state, None

    def terminate(self, problem, y, args, options, state):
        # TODO(raderj): only check terminate every k
        (f_best,), (best_index,) = lax.top_k(-state.f_simplex, 1)
        f_best = -f_best

        best = ω(state.simplex, structure=y)[best_index].ω

        x_scale = (self.atol + self.rtol * ω(best).call(jnp.abs)).ω
        f_scale = (self.atol + self.rtol * ω(f_best).call(jnp.abs)).ω

        unwrapped_simplex = ω(state.simplex, structure=y).call(lambda x: x[...]).ω
        x_diff = ((ω(unwrapped_simplex) - ω(best)).call(jnp.abs) / ω(x_scale)).ω
        x_conv = self.norm(x_diff) < 1

        f_diff = ((ω(state.f_simplex) - ω(f_best)).call(jnp.abs) / f_scale).ω
        f_conv = self.norm(f_diff) < 1
        #
        # minpack does a further test here where it takes for each unit vector e_i a
        # perturbation "delta" and asserts that f(x + delta e_i) > f(x) and
        # f(x - delta e_i) > f(x). ie. a rough assertion that it is indeed a local
        # minimum. If it fails the algo resets completely. thus process scales as
        # O(dim(y) * T(f)), where T(f) is the cost of evaluating f.
        #
        converged = x_conv & f_conv
        diverged = jnp.any(jnp.invert(jnp.isfinite(f_best)))
        terminate = converged | diverged
        result = jnp.where(diverged, RESULTS.nonlinear_divergence, RESULTS.successful)
        return terminate, result

    def buffer(self, state):
        return state.simplex
