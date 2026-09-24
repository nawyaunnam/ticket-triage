# TicketTriage — explainable text classification

A trainable support-ticket classifier with multinomial Naive Bayes, confidence-based abstention, JSON model persistence, and per-class evaluation.

## What the recorded evaluation shows

The [committed public-issue snapshot](live-report.json) contains 214 labeled issues: 171 for training and the newest 43 for evaluation. On that snapshot, macro F1 is **0.428**, accuracy is **58.1%**, and documentation recall is **0%**. The training set has only six documentation examples. These results describe this sample, not production routing quality.

The missed class matters more than an attractive overall score. The next experiment is to compare a TF-IDF linear model and inspect documentation errors using a separate validation set, without tuning against the held-out test set.

See the [classifier and metrics](engine.py), [live data preparation](live.py), and [leakage/abstention tests](test_engine.py).

## Tech stack

| Layer | Technologies used |
| --- | --- |
| Language | Python 3.11+ |
| Model | Multinomial Naive Bayes implemented in Python |
| NLP and evaluation | Regex tokenization, temporal split, macro F1, confusion matrix, abstention |
| Model artifact | Versioned JSON model; no pickle |
| Live source | GitHub REST API: labeled scikit-learn issues |
| Dashboard | HTML5, CSS, vanilla JavaScript; Python HTTP server |
| Data transport | urllib.request, verified TLS, JSON, ETag caching |
| Testing and CI | unittest, GitHub Actions; Python 3.11–3.13 matrix |

The implementation uses the Python standard library; no external Python packages are required.

## Run in two commands

From this project directory:

```sh
python3 demo.py
python3 -m unittest discover -v
```

No API keys, paid services, or package downloads are required. The offline demo uses
synthetic or hand-authored examples and temporary storage; it does not access
personal data. See [demo output](demo-output.txt) for a captured local run.

## Design

Labeled training tickets → training-only vocabulary and class counts → smoothed likelihoods → log-space inference → normalized probabilities and abstention. Evaluation reports macro F1, accuracy, coverage, confusion counts, and a majority-class baseline.

## Review the implementation

- [Core implementation](engine.py): domain logic and persistence/algorithms.
- [Executable demo](demo.py): a complete sample workflow.
- [Tests](test_engine.py): expected behavior and edge/failure cases.
- [Architecture and interview notes](DESIGN.md).
- GitHub Actions runs the tests and demo on Python 3.11–3.13 after publishing.

## Scope and limits

Small English hand-authored dataset and simple bag-of-words model. Probabilities are uncalibrated; confidence is not a guarantee. Demo test examples are held out from training but are not an independent benchmark. No claim of production ticket-routing performance.

This is a portfolio implementation, not evidence of production use or business
impact. Any reported metrics describe only the included demonstration data.

## Live public-data workflow

Trains on real, publicly labeled [scikit-learn issues](https://github.com/scikit-learn/scikit-learn/issues). Uses Bug, Documentation, and New Feature labels as weak supervision, removes pull requests and duplicate titles, then evaluates on the newest 20% of collected issues. Reports class balance and baseline performance; it does not hide low scores. Source titles remain attributed by issue URL.

```sh
python3 live.py                         # Fetch real public data, save snapshot.json and report.json
python3 live.py --replay snapshot.json   # Reproduce analysis from the captured data
python3 dashboard.py                    # Open http://127.0.0.1:8090
# In a separate terminal, to keep fetching while viewing the dashboard:
python3 live.py --watch 120
```

The report includes acquisition timestamps and source URLs. HTTP requests use
verified TLS, timeouts, bounded retries, ETags, and a minimum polling interval.
Network failures are explicit; synthetic data is never substituted for live data.
`demo.py` remains an offline synthetic example for tests and onboarding.

The dashboard is a local demonstration, not a publicly deployed service.
Stop the dashboard and live collector with Ctrl+C. Automated CI runs offline tests;
it does not repeatedly call third-party APIs.

## Verified run

- **9 automated tests passed** locally on Python 3.14.
- Live data was fetched successfully; [captured report](live-report.json).
- [Snapshot](snapshot.json) records real data and source acquisition timestamps.
- Offline replay was verified from a fresh temporary working directory.
- [Test log](test-output.txt) and [demo log](demo-output.txt) are included.

To inspect the bundled report without fetching data:

```sh
python3 dashboard.py --report live-report.json
```

## Next engineering milestone

Improve documentation recall using a separate validation set; compare a TF-IDF linear model without tuning on the held-out test set.

## License and provenance

Original code: [MIT](LICENSE). [Source attribution](DATA-SOURCES.md).
