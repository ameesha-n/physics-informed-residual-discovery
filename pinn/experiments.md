Experiment: PINN Alpha Recovery

True Alpha -> Learned Alpha

0.3 -> 0.234
0.7 -> 0.527
1.0 -> 0.855
1.5 -> 1.226

Linear Fit:
Recovered ≈ 0.842 * True - 0.026

Adaptive Weighting (α=0.7):
0.527 -> 0.632

Observation:
Adaptive weighting improves alpha recovery but becomes unstable due to rapidly increasing learned weights.