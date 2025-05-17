import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: Human review needed to verify the implementation matches the problem definition
class LUKSAN11LS(AbstractUnconstrainedMinimisation, strict=True):
    """Problem 11 (chained serpentine) in the Luksan collection.

    From the paper:
        L. Luksan
        Hybrid methods in large sparse nonlinear least squares
        J. Optimization Theory & Applications 89(3) 575-595 (1996)

    SIF input: Nick Gould, June 2017.
    Classification: SUR2-AN-V-0

    This is the least-squares version of the problem.
    """

    # Variable dimension
    n: int = 99  # Default dimension from the SIF file

    def objective(self, y, args):
        """Compute the objective function for the chained serpentine problem.

        The problem includes residuals of the form:
        1. e_2i-1 = 20*x_i/(1 + x_i^2) - 10*x_{i+1} where i=1,2,...,n-1
        2. e_2i = 1.0 (constants) where i=1,...,n-1

        The objective is the sum of squares of these residuals.
        """

        # Element function (from lines 126-130):
        # 20*v/(1 + v^2)
        def element_function(v):
            return 20.0 * v / (1.0 + v**2)

        # First set of residuals: e_2i-1 = 20*x_i/(1 + x_i^2) - 10*x_{i+1}
        # for i=1,...,n-1
        # For all elements except the last one
        x_head = y[:-1]  # All elements except the last one
        x_tail = y[1:]  # All elements except the first one

        # Apply the element function to each element in x_head and subtract 10*x_tail
        first_residuals = jax.vmap(element_function)(x_head) - 10.0 * x_tail

        # Second set of residuals: e_2i = 1.0 for i=1,...,n-1
        second_residuals = jnp.ones(self.n - 1)

        # Combine all residuals
        all_residuals = jnp.concatenate([first_residuals, second_residuals])

        # Return the sum of squared residuals
        return jnp.sum(all_residuals**2)

    def y0(self):
        """Initial point for the problem: all variables set to -0.8 (line 72)."""
        return jnp.full(self.n, -0.8)

    def args(self):
        """No additional arguments needed."""
        return None

    def expected_result(self):
        """The expected solution is not specified in the SIF file."""
        return None

    def expected_objective_value(self):
        """The optimal objective value is 0.0 (line 109)."""
        return jnp.array(0.0)


# TODO: Human review needed to verify the implementation matches the problem definition
class LUKSAN12LS(AbstractUnconstrainedMinimisation, strict=True):
    """Problem 12 (chained modified barrier) in the Luksan collection.

    From the paper:
        L. Luksan
        Hybrid methods in large sparse nonlinear least squares
        J. Optimization Theory & Applications 89(3) 575-595 (1996)

    SIF input: Nick Gould, June 2017.
    Classification: SUR2-AN-V-0

    This is the least-squares version of the problem.
    """

    # Variable dimension
    n: int = 99  # Default dimension

    def objective(self, y, args):
        """Compute the objective function for the chained modified barrier problem.

        The problem includes residuals of the form:
        1. e_2i-1 = 2*sqrt(1 + x_i^2) - x_i - x_{i+1} where i=1,2,...,n-1
        2. e_2i = 1.0 (constants) where i=1,...,n-1

        The objective is the sum of squares of these residuals.
        """

        # Element function: 2*sqrt(1 + v^2) - v
        def element_function(v):
            return 2.0 * jnp.sqrt(1.0 + v**2) - v

        # First set of residuals: e_2i-1 = 2*sqrt(1 + x_i^2) - x_i - x_{i+1}
        # for i=1,...,n-1
        x_head = y[:-1]  # All elements except the last one
        x_tail = y[1:]  # All elements except the first one

        # Apply the element function to each element in x_head and subtract x_tail
        first_residuals = jax.vmap(element_function)(x_head) - x_tail

        # Second set of residuals: e_2i = 1.0 for i=1,...,n-1
        second_residuals = jnp.ones(self.n - 1)

        # Combine all residuals
        all_residuals = jnp.concatenate([first_residuals, second_residuals])

        # Return the sum of squared residuals
        return jnp.sum(all_residuals**2)

    def y0(self):
        """Initial point for the problem."""
        return jnp.full(self.n, -1.2)

    def args(self):
        """No additional arguments needed."""
        return None

    def expected_result(self):
        """The expected solution is not specified in the SIF file."""
        return None

    def expected_objective_value(self):
        """The optimal objective value is assumed to be 0.0."""
        return jnp.array(0.0)


# TODO: Human review needed to verify the implementation matches the problem definition
class LUKSAN13LS(AbstractUnconstrainedMinimisation, strict=True):
    """Problem 13 (chained singular) in the Luksan collection.

    From the paper:
        L. Luksan
        Hybrid methods in large sparse nonlinear least squares
        J. Optimization Theory & Applications 89(3) 575-595 (1996)

    SIF input: Nick Gould, June 2017.
    Classification: SUR2-AN-V-0

    This is the least-squares version of the problem.
    """

    # Variable dimension
    n: int = 99  # Default dimension

    def objective(self, y, args):
        """Compute the objective function for the chained singular problem.

        The problem includes residuals of the form:
        1. e_2i-1 = x_i^4 - x_{i+1} + x_i - 1 where i=1,2,...,n-1
        2. e_2i = 1.0 (constants) where i=1,...,n-1

        The objective is the sum of squares of these residuals.
        """
        # First set of residuals: e_2i-1 = x_i^4 - x_{i+1} + x_i - 1 for i=1,...,n-1
        x_head = y[:-1]  # All elements except the last one
        x_tail = y[1:]  # All elements except the first one

        first_residuals = x_head**4 - x_tail + x_head - 1.0

        # Second set of residuals: e_2i = 1.0 for i=1,...,n-1
        second_residuals = jnp.ones(self.n - 1)

        # Combine all residuals
        all_residuals = jnp.concatenate([first_residuals, second_residuals])

        # Return the sum of squared residuals
        return jnp.sum(all_residuals**2)

    def y0(self):
        """Initial point for the problem."""
        return jnp.full(self.n, -1.0)

    def args(self):
        """No additional arguments needed."""
        return None

    def expected_result(self):
        """The expected solution is not specified in the SIF file."""
        return None

    def expected_objective_value(self):
        """The optimal objective value is assumed to be 0.0."""
        return jnp.array(0.0)


# TODO: Human review needed to verify the implementation matches the problem definition
class LUKSAN14LS(AbstractUnconstrainedMinimisation, strict=True):
    """Problem 14 (chained Wood) in the Luksan collection.

    From the paper:
        L. Luksan
        Hybrid methods in large sparse nonlinear least squares
        J. Optimization Theory & Applications 89(3) 575-595 (1996)

    SIF input: Nick Gould, June 2017.
    Classification: SUR2-AN-V-0

    This is the least-squares version of the problem.
    """

    # Variable dimension
    n: int = 100  # Default dimension (must be even)

    def __check_init__(self):
        """Validates that n is even."""
        if self.n % 2 != 0:
            raise ValueError("n must be even for the LUKSAN14LS problem")

    def objective(self, y, args):
        """Compute the objective function for the chained Wood problem.

        The residuals follow a pattern based on the classic Wood function,
        but chained together in overlapping blocks.
        """
        # In the Wood problem, variables are processed in groups of 4,
        # but with overlap, so we need to handle this carefully in JAX.

        # Get the even and odd indexed elements
        even_indices = jnp.arange(0, self.n, 2)
        odd_indices = jnp.arange(1, self.n, 2)

        # For elements that require even-indexed variables
        x_even = y[even_indices]
        x_odd = y[odd_indices]

        # The first type of residuals: 10(x_{2i+1} - x_{2i}^2) for i=0,...,n/2-1
        r1 = 10.0 * (x_odd - x_even**2)

        # The second type of residuals: 1 - x_{2i} for i=0,...,n/2-1
        r2 = 1.0 - x_even

        # The third and fourth type need to be handled with care due to indexing
        # We'll use only the valid pairs up to n-2
        valid_pairs = min(len(even_indices), len(odd_indices) - 1)

        if valid_pairs > 0:
            # The third type: 90(x_{2i+3} - x_{2i+1}^2) for i=0,...,n/2-2
            r3 = 90.0 * (x_odd[1 : valid_pairs + 1] - x_odd[:valid_pairs] ** 2)

            # The fourth type: 1 - x_{2i+1} for i=0,...,n/2-2
            r4 = 1.0 - x_odd[:valid_pairs]

            # The fifth type: 10(x_{2i+1} + x_{2i+3} - 2) for i=0,...,n/2-2
            r5 = 10.0 * (x_odd[:valid_pairs] + x_odd[1 : valid_pairs + 1] - 2.0)

            # The sixth type: 10(x_{2i} - x_{2i+1}) for i=0,...,n/2-2
            r6 = 10.0 * (x_even[:valid_pairs] - x_odd[:valid_pairs])

            # Combine all residuals
            all_residuals = jnp.concatenate([r1, r2, r3, r4, r5, r6])
        else:
            # If there are not enough elements, just use the first two types
            all_residuals = jnp.concatenate([r1, r2])

        # Return the sum of squared residuals
        return jnp.sum(all_residuals**2)

    def y0(self):
        """Initial point for the problem."""
        # Create an array of repeating pattern [-3, -1, -3, -1, ...]
        y0_values = jnp.array([-3.0, -1.0, -3.0, -1.0])
        repeats = self.n // 4 + (1 if self.n % 4 != 0 else 0)
        repeated_pattern = jnp.tile(y0_values, repeats)

        # Truncate to the required length
        return repeated_pattern[: self.n]

    def args(self):
        """No additional arguments needed."""
        return None

    def expected_result(self):
        """The expected solution is not specified in the SIF file."""
        return None

    def expected_objective_value(self):
        """The optimal objective value is assumed to be 0.0."""
        return jnp.array(0.0)


# TODO: Human review needed to verify the implementation matches the problem definition
class LUKSAN15LS(AbstractUnconstrainedMinimisation, strict=True):
    """Problem 15 (chained Powell singular) in the Luksan collection.

    From the paper:
        L. Luksan
        Hybrid methods in large sparse nonlinear least squares
        J. Optimization Theory & Applications 89(3) 575-595 (1996)

    SIF input: Nick Gould, June 2017.
    Classification: SUR2-AN-V-0

    This is the least-squares version of the problem.
    """

    # Variable dimension
    n: int = 100  # Default dimension (must be divisible by 4)

    def __check_init__(self):
        """Validates that n is divisible by 4."""
        if self.n % 4 != 0:
            raise ValueError("n must be divisible by 4 for the LUKSAN15LS problem")

    def objective(self, y, args):
        """Compute the objective function for the chained Powell singular problem.

        The residuals follow a pattern based on the classic Powell singular function,
        but chained together in overlapping blocks.
        """
        # For the Powell singular function, we process variables in groups of 4
        # num_groups = self.n // 4  # Unused variable

        # Get the variables in each position within their respective groups
        x1 = y[0::4]  # x_{4i+1} for i=0,...,n/4-1
        x2 = y[1::4]  # x_{4i+2} for i=0,...,n/4-1
        x3 = y[2::4]  # x_{4i+3} for i=0,...,n/4-1
        x4 = y[3::4]  # x_{4i+4} for i=0,...,n/4-1

        # Calculate the 4 types of residuals for each group
        r1 = x1 + 10.0 * x2
        r2 = 5.0**0.5 * (x3 - x4)
        r3 = (x2 - 2.0 * x3) ** 2
        r4 = 10.0**0.5 * (x1 - x4) ** 2

        # Combine all residuals
        all_residuals = jnp.concatenate([r1, r2, r3, r4])

        # Return the sum of squared residuals
        return jnp.sum(all_residuals**2)

    def y0(self):
        """Initial point for the problem."""
        # Create an array of repeating pattern [3, -1, 0, 1, ...]
        y0_values = jnp.array([3.0, -1.0, 0.0, 1.0])
        repeats = self.n // 4
        return jnp.tile(y0_values, repeats)

    def args(self):
        """No additional arguments needed."""
        return None

    def expected_result(self):
        """The expected solution is not specified in the SIF file."""
        return None

    def expected_objective_value(self):
        """The optimal objective value is assumed to be 0.0."""
        return jnp.array(0.0)


# Placeholder for remaining LUKSAN problems
class LUKSAN16LS(AbstractUnconstrainedMinimisation, strict=True):
    """Problem 16 (chained Rosenbrock) in the Luksan collection."""

    n: int = 100  # Default dimension

    def objective(self, y, args):
        raise NotImplementedError("LUKSAN16LS implementation to be completed")

    def y0(self):
        return jnp.ones(self.n) * (-1.2)

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return jnp.array(0.0)


class LUKSAN17LS(AbstractUnconstrainedMinimisation, strict=True):
    """Problem 17 (chained Cragg and Levy) in the Luksan collection."""

    n: int = 100  # Default dimension

    def objective(self, y, args):
        raise NotImplementedError("LUKSAN17LS implementation to be completed")

    def y0(self):
        return jnp.ones(self.n) * 1.0

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return jnp.array(0.0)


class LUKSAN21LS(AbstractUnconstrainedMinimisation, strict=True):
    """Problem 21 (chained Broyden tridiagonal) in the Luksan collection."""

    n: int = 100  # Default dimension

    def objective(self, y, args):
        raise NotImplementedError("LUKSAN21LS implementation to be completed")

    def y0(self):
        return jnp.ones(self.n) * (-1.0)

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return jnp.array(0.0)


class LUKSAN22LS(AbstractUnconstrainedMinimisation, strict=True):
    """Problem 22 (chained Broyden banded) in the Luksan collection."""

    n: int = 100  # Default dimension

    def objective(self, y, args):
        raise NotImplementedError("LUKSAN22LS implementation to be completed")

    def y0(self):
        return jnp.ones(self.n) * (-1.0)

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return jnp.array(0.0)
