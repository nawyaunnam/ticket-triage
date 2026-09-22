# Engineering notes: TicketTriage — explainable text classification

## Problem and flow

Labeled training tickets → training-only vocabulary and class counts → smoothed likelihoods → log-space inference → normalized probabilities and abstention. Evaluation reports macro F1, accuracy, coverage, confusion counts, and a majority-class baseline.

## Current boundaries

Small English hand-authored dataset and simple bag-of-words model. Probabilities are uncalibrated; confidence is not a guarantee. Demo test examples are held out from training but are not an independent benchmark. No claim of production ticket-routing performance.

## Interview walkthrough

1. Run the demo and explain each output in terms of the code.
2. Show a test that exercises a failure rather than only a successful call.
3. Trace one input through the core implementation and its stored state.
4. Explain the tradeoff made by the current storage or algorithm choice.
5. Describe what would change with 100× the data or concurrent users.
6. Make a small extension and add a regression test before using this in a resume.

## Validation

See `test_engine.py` for executable assertions and `docs/demo-output.txt` for
captured results. CI is configured but remote CI results are not assumed.
