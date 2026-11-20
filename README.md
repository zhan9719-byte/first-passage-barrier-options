# First-Passage Modeling and Barrier Option Valuation

Near a knock-out barrier, can the option price appear to converge while its Gamma remains inaccurate? This study connects first-passage probabilities to the pricing PDE, calibrates Heston to SPXW vanilla quotes on 24 trading days, and tests how reliably finite differences and Monte Carlo estimate Gamma.

| Experiment | Result |
| --- | --- |
| PDE refinement | Checking price, Delta and Gamma while refining one grid direction at a time cut total measured runtime from 49.25 s to 34.07 s (30.8%) versus balanced joint refinement across 11 cases. Both methods flagged two severe low-variance cases as unresolved. |
| Monte Carlo Gamma | Conditioning on survival and integrating the final step analytically improved sampling efficiency by 1.66–4.75× in 7 of 8 scenarios. In the low-variance stress case, efficiency fell to 0.14× the baseline. Efficiency compares sampling variance times runtime. |

[Research report (PDF)](barrier-valuation.pdf) · [Gamma efficiency by scenario](gamma-efficiency.png) · [PDE results](refinement.csv) · [Monte Carlo results](gamma.csv)

## Reproduce the near-barrier case

With Python 3.11 or later, run from the repository root:

```sh
python -m pip install -r requirements.txt
python example.py
```

The example prices a call with spot and strike 100, barrier 97, and 0.1 years to expiry. Price-only stopping accepts a Gamma error of 5.81e-5 against a tolerance of 3e-5. Directional refinement brings the error to 4.01e-6. The independent reference enters only after the grid has been selected.
