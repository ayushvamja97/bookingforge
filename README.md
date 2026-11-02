# BookingForge

Predictive Intelligence & Optimization Engine — an SDE-focused AI/ML/DL research and software engineering portfolio project.

## Status
Phase 0 scaffold. No model has been trained and no performance claims are made.

## Goals
- Reproducible cancellation-risk classification with leakage-aware evaluation.
- Forecasting comparisons: seasonal baseline, tabular models, and PyTorch sequence models.
- Algorithmic decision experiments: Poisson-binomial DP, Monte Carlo simulation, and constrained room assignment.
- Production-style API, tests, CI, and benchmark documentation.

## Research contract
1. Define prediction timestamp before choosing features.
2. Keep outcome and post-decision information out of features.
3. Use chronological evaluation; ensure labels are known at each training cutoff.
4. Keep simulated capacity, costs, and counterfactual decisions clearly labeled assumptions.
5. Report negative results and test uncertainty.
6. Never claim a live hotel deployment or actual revenue improvement.

## Source data
Planned: Hotel Booking Demand dataset (António, Almeida & Nunes, 2019). Data acquisition and license checks pending. No employer or guest records permitted.

## Roadmap
1. Dataset provenance, schema, timestamp/leakage audit, baseline protocol.
2. Custom kNN / NumPy MLP and standard ML comparisons.
3. Demand forecasting and deep learning experiments.
4. Exact DP, heuristic search, constrained matching, and simulation.
5. FastAPI service, integration, CI, and interactive benchmark UI.

## Getting started
Run `python -m unittest discover -s tests -v` from the project root. No external dependencies needed for the initial scaffold.
