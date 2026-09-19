"""Compatibility CLI: prefer python -m medworld.evaluation.evaluate."""
from .evaluation.evaluate import main
from .downstream_tasks.classification import classification_metrics
from .downstream_tasks.report import report_diagnostics

if __name__ == "__main__":
    main()
