import numpy as np


OUTPUTS = ('price', 'delta', 'gamma')


AXES = ('nx', 'nv', 'nt')


def node_work(grid):
    return int(np.prod(grid))


def normalized_change(a, b, tol, targets=OUTPUTS):
    return max((abs(a[k] - b[k]) / tol[k] for k in targets))


class ChargedEvaluator:

    def __init__(self, solve):
        self.solve = solve
        self.values = {}
        self.history = []

    def __call__(self, grid, role):
        grid = tuple((int(x) for x in grid))
        if grid not in self.values:
            a = self.solve(grid)
            self.values[grid] = a
            self.history.append(dict(nx=grid[0], nv=grid[1], nt=grid[2], role=role,
                price=a['price'], delta=a['delta'], gamma=a['gamma'], elapsed=a['elapsed'],
                min_value=a.get('min_value', np.nan), node_work=node_work(grid)))
        return self.values[grid]

    def finish(self, grid, accepted, status, decisions):
        a = self.values[tuple(grid)]
        return dict(accepted=bool(accepted), status=status, grid=list(grid),
            **{k: float(a[k]) for k in OUTPUTS},
            charged_seconds=sum((x['elapsed'] for x in self.history)),
            charged_node_work=sum((x['node_work'] for x in self.history)),
            solves=len(self.history), history=self.history, decisions=decisions,
            min_value=a.get('min_value', np.nan))


def run_controller(solve, levels, tolerances, method='directional_joint', max_evaluations=24):
    """Allocate refinements using price, delta and gamma changes per added unit of work."""
    if method not in {'price_only', 'uniform_joint', 'directional_joint'}:
        raise ValueError(method)
    ls = [levels[k] for k in AXES]
    ev = ChargedEvaluator(solve)
    idx = [0, 0, 0]
    decisions = []
    grid = lambda ii: tuple((ls[j][ii[j]] for j in range(3)))
    current = grid(idx)
    a = ev(current, 'initial')
    if method != 'directional_joint':
        targets = ('price',) if method == 'price_only' else OUTPUTS
        streak = 0
        for step in range(1, min(map(len, ls))):
            nxt = grid([step] * 3)
            b = ev(nxt, 'balanced_refinement')
            discrepancy = normalized_change(a, b, tolerances, targets)
            decisions.append(dict(grid=list(nxt), indicator=discrepancy, targets=list(targets)))
            streak = streak + 1 if discrepancy <= 1 else 0
            current = nxt
            a = b
            if streak >= 2:
                return ev.finish(current, True, 'two_successive_checks', decisions)
        return ev.finish(current, False, 'grid_cap_reached', decisions)
    while len(ev.history) < max_evaluations:
        probes = []
        for axis in range(3):
            if idx[axis] >= len(ls[axis]) - 1:
                continue
            nxtidx = idx.copy()
            nxtidx[axis] += 1
            nxt = grid(nxtidx)
            if len(ev.history) >= max_evaluations and nxt not in ev.values:
                break
            b = ev(nxt, 'axis_' + AXES[axis])
            d = normalized_change(a, b, tolerances)
            added = max(node_work(nxt) - node_work(current), 1)
            probes.append((axis, nxtidx, nxt, b, d, d / added))
        if not probes:
            return ev.finish(current, False, 'grid_cap_reached', decisions)
        all_available = len(probes) == 3
        maxd = max((x[4] for x in probes))
        decisions.append(dict(grid=list(current),
            axis_indicators={AXES[x[0]]: x[4] for x in probes}))
        if all_available and maxd <= 1:
            combinedidx = [j + 1 for j in idx]
            combined = grid(combinedidx)
            if len(ev.history) >= max_evaluations and combined not in ev.values:
                return ev.finish(current, False, 'evaluation_budget', decisions)
            c = ev(combined, 'combined_confirmation')
            cd = normalized_change(a, c, tolerances)
            decisions.append(dict(combined=list(combined), indicator=cd))
            if cd <= 1:
                return ev.finish(combined, True, 'axial_and_combined_checks', decisions)
            idx = combinedidx
            current = combined
            a = c
            continue
        failing = [x for x in probes if x[4] > 1]
        selected = max(failing or probes, key=lambda x: x[5])
        axis, idx, current, a, d, score = selected
        decisions.append(dict(selected_axis=AXES[axis], indicator=d, score_per_added_work=score))
    return ev.finish(current, False, 'evaluation_budget', decisions)
