from .problem import AbstractUnconstrainedMinimisation


# TODO: Placeholder class. Needs human implementation due to size/complexity.
class _LRBase(AbstractUnconstrainedMinimisation, strict=True):
    """Base class for regularized logistic regression problems from LIBSVM collection.

    The problem is to minimize the cross-entropy function for binary classification
    with an added regularization term:

    C(w) = -1/|samples| sum_i (samples) [(y_i - l)/(u - l) log p(w,x_i) +
           (u - y_i)/(u - l) log(1 - p(w,x_i))]

    where:
    - p(x,w) = 1/(1 + exp(-l(x,w)))
    - l(x,w) = sum_j (features) w_j x_j

    With added regularization: sigma sum_j (features) x_j^2 / (1 + x_j^2)

    Source: the LIBSVM collection
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

    See: Chih-Chung Chang and Chih-Jen Lin,
    LIBSVM: a library for support vector machines.
    ACM Trans. Intelligent Systems & Technology, 2:27:1--27:27, 2011.

    SIF input: Nick Gould, Jan 2023
    """

    def objective(self, y, args):
        """Compute the regularized logistic regression objective function."""
        raise NotImplementedError(
            "Placeholder. LR problems need human implementation due to size/complexity."
        )

    def y0(self):
        """Initial point for the problem."""
        raise NotImplementedError(
            "Placeholder. LR problems need human implementation due to size/complexity."
        )

    def args(self):
        """Additional arguments for the objective function."""
        return None

    def expected_result(self):
        """Expected result of the optimization problem."""
        return None

    def expected_objective_value(self):
        """Expected value of the objective function at the optimal solution."""
        return None


# TODO: Placeholder class. Needs human implementation due to size/complexity.
class LRA9A(_LRBase, strict=True):
    """Regularized logistic regression on the a9a dataset from LIBSVM.

    Original SIF file: 15MB with 32561 samples and 123 features.
    """

    pass


# TODO: Placeholder class. Needs human implementation due to size/complexity.
class LRCOVTYPE(_LRBase, strict=True):
    """Regularized logistic regression on the covtype dataset from LIBSVM.

    Original SIF file: 235MB, extremely large dataset.
    """

    pass


# TODO: Placeholder class. Needs human implementation due to size/complexity.
class LRIJCNN1(_LRBase, strict=True):
    """Regularized logistic regression on the ijcnn1 dataset from LIBSVM.

    Original SIF file: 22MB, large dataset.
    """

    pass


# TODO: Placeholder class. Needs human implementation due to size/complexity.
class LRW1A(_LRBase, strict=True):
    """Regularized logistic regression on the w1a dataset from LIBSVM.

    Original SIF file: 975KB with 2477 samples and 300 features.
    """

    pass


# TODO: Placeholder class. Needs human implementation due to size/complexity.
class LRW8A(_LRBase, strict=True):
    """Regularized logistic regression on the w8a dataset from LIBSVM.

    Original SIF file: 20MB, large dataset.
    """

    pass
