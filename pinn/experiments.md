Alpha Recovery Experiment

Initial Result:
Systematic underestimation observed.

Investigation:
Hidden residual term was unintentionally included
during dataset generation.

Fixes:
1. Added include_hidden_residual flag.
2. Regenerated datasets.
3. Increased Nx from 256 to 512.

Final Result:

0.3 -> 0.300000
0.7 -> 0.700000
1.0 -> 1.000000
1.5 -> 1.500000

Conclusion:
Learnable alpha recovers the true forcing coefficient
to numerical precision when trained on correctly
specified datasets.