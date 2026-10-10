"""Run the full analysis: diagnostics, learners, evaluation, business simulation, figures.

Writes reports/figures/*.png, reports/results.md and artifacts/summary.json.
"""

import json

from uplift_campaign.pipeline import run

if __name__ == "__main__":
    summary = run()
    print(
        json.dumps({k: summary[k] for k in ("best_model_by_cv", "model_metrics", "business")}, indent=2, default=float)
    )
