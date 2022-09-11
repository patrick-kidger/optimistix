- Gauss-Newton
- LM
- Adam
- Line search / learning rates for GN, Newton, etc.
- Figuring out the relationship between least squares, root finding, and fixed points
- Figuring out the relationship between Newton and Gauss-Newton
- Handling implicit differentiation + use while_loop not bounded_while_loop when doing so
- Handling bounded while loops efficiently (current approach uses no checkpointing because max_steps=16?)
- Handling Optax-like things
- Have newton take in a linear solver for the backward pass
- Make sure Newton can use symmetric-capable solvers for when solving minimisations problems (root = grad(fn) so jac(root) = hessian(fn) is symmetric)

Done?
- Handling Newton/Chord when n != m (or at least add a check that n==m)
  - At the very least, least_squares seems to require m >= n. But what about the appending-zeros-to-state case?
- Need to think about choice of linear solvers: in particular LU vs lstsq
- Consider when you have roots of higher multiplicity
