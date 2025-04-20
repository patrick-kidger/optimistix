import optimistix as optx

from .cutest import FLETCHER


def main(problem):
    problem = problem()

    solver = optx.IPOPTLike(rtol=1e-2, atol=1e-2)

    solution = optx.minimise(
        problem.objective,
        solver,
        problem.y0(),
        problem.args(),
        constraint=problem.constraint,
        bounds=problem.bounds(),
    )

    print("Solution value: {}".format(solution.value))


if __name__ == "__main__":
    main(FLETCHER)
