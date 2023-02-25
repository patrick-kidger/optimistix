- Tests:
  - all the complexity of {PyTree,TransposeJacobian,Tangent}LinearOperator
  - transposing all linear solvers
  - Test JAX #13452
  - passing AD closures in to `linear_solve_p.impl`
- Think about passing Refs into custom primitives. Do the abstract eval rules?
- LBFGs etc.
- Handle low-rank + diagonal solving using Woodbury
- LM
  - Geodesic acceleration for LM
- Let @felix_m_koehler (Twitter) and @botev (DeepMind) know once this is ready, they may have opinions.

- Falsi
- Line search / learning rates for GN, Newton, etc.
 - Hager-Zhang [https://github.com/JuliaNLSolvers/Optim.jl/issues/153 claims 2005 paper is worse than 2012 paper]
 - Zoom
 - Frank Wolfe
 - Armijo backatracking
 - Fista
- Handling Optax-like things

Problems with existing JAXopt:
- LM etc. evaluates during init, so 2x as long to compile
- Base solvers unroll an iteration, so 2x as long to compile: `https://github.com/google/jaxopt/blob/ddd6c30953f6c02d8022c1358aa67d9114aed3a6/jaxopt/_src/base.py#L159`
- Normal CG is ludicrously inefficient to compile: evaluate matvec many times.
- Implicit diff done using reverse mode instead of forward mode?
- Loops are inefficient, e.g. under vmap

Lesser problems with JAXopt:
- LM doesn't work with PyTrees.
- Insufficiently advanced linear solvers
- https://github.com/google/jaxopt/blob/c3ed0fb76054fcae4ecae1746fef3addeadfb0fa/jaxopt/_src/gauss_newton.py#L131
