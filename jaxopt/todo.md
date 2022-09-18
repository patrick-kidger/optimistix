- Examine termination conditions for ~Newton, GN/LM~ all methods. (e.g. FixedPointIteration has a plain `tol`)
- QR for rectangular
- LM
- Geodesic acceleration for LM
- Adam/LBFGS/etc.
- Falsi
- Line search / learning rates for GN, Newton, etc.
 - Hager-Zhang
 - Zoom
 - Frank Wolfe
 - Armijo backatracking
 - Fista
- Handling Optax-like things
- Linear solve adjoints
- Aux output

Done?
- Handling Newton/Chord when n != m (or at least add a check that n==m)
  - At the very least, least_squares seems to require m >= n. But what about the appending-zeros-to-state case?
- Need to think about choice of linear solvers: in particular LU vs lstsq
- Consider when you have roots of higher multiplicity
- Handling implicit differentiation + use while_loop not bounded_while_loop when doing so
- Handling bounded while loops efficiently (current approach uses no checkpointing because max_steps=16?)
- Have newton take in a linear solver for the backward pass
- Make sure Newton can use symmetric-capable solvers for when solving minimisations problems (root = grad(fn) so jac(root) = hessian(fn) is symmetric)
- Figuring out the relationship between least squares, root finding, and fixed points

Problems with existing JAXopt:
- LM etc. evaluates during init, so 2x as long to compile
- Base solvers unroll an iteration, so 2x as long to compile: `https://github.com/google/jaxopt/blob/ddd6c30953f6c02d8022c1358aa67d9114aed3a6/jaxopt/_src/base.py#L159`
- Normal CG is ludicrously inefficient to compile: evaluate matvec many times.
- Implicit diff done using reverse mode instead of forward mode?
- Loops are inefficient, e.g. under vmap
Lesser problems with JAXopt:
- LM doesn't work with PyTrees.
