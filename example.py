from functools import lru_cache

from pde import heston_pde_refined
from refinement import run_controller


LEVELS = {
    "nx": [81, 121, 181, 271, 401],
    "nv": [31, 41, 61, 91, 121],
    "nt": [40, 80, 160, 320, 640],
}
TOLERANCES = {"price": 0.002, "delta": 0.0003, "gamma": 0.00003}
PARAMS = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.5, "rho": -0.7}


@lru_cache(maxsize=None)
def solve(grid):
    nx, nv, nt = grid
    result = heston_pde_refined(
        S=100, K=100, B=97, T=0.1, r=0.03, q=0.015, params=PARAMS,
        nx=nx, nv=nv, nt=nt, return_grid=True,
    )
    return {key: result[key] for key in ("price", "delta", "gamma", "elapsed", "min_value")}


def main():
    # Supplemental 901 × 241 × 640 reference, used only after each solve.
    reference_gamma = 0.0395111235393237
    reference_spread = 7.216785606639631e-6
    print("method             stop      solves   node-work       gamma     |error|   audit")
    for method in ("price_only", "uniform_joint", "directional_joint"):
        result = run_controller(solve, LEVELS, TOLERANCES, method)
        error = abs(result["gamma"] - reference_gamma)
        if error + reference_spread <= TOLERANCES["gamma"]:
            audit = "pass"
        elif error - reference_spread > TOLERANCES["gamma"]:
            audit = "fail"
        else:
            audit = "ambiguous"
        decision = "accept" if result["accepted"] else "reject"
        print(f"{method:18s} {decision:7s} {result['solves']:6d} "
              f"{result['charged_node_work']:11,d} {result['gamma']:11.8f} "
              f"{error:11.4g}   {audit}")


if __name__ == "__main__":
    main()
