"""Optional uncertainty analysis layer (semantic entropy, AUROC, conformal abstention). Exploratory."""
from evals.uncertainty.conformal import CalibrationTooSmall, calibrate, conformal_abstention, split_by_scenario
from evals.uncertainty.evaluation import (auroc, auroc_bootstrap, build_eval_table, compare_predictors,
                                          selective_summary, uncertainty_risk_coverage)
from evals.uncertainty.report import format_report, uncertainty_report
from evals.uncertainty.semantic_entropy import (Sample, cluster_by_entailment, discrete_semantic_entropy,
                                                score_case, sequence_logprob_score)

__all__ = ["CalibrationTooSmall", "Sample", "auroc", "auroc_bootstrap", "build_eval_table", "calibrate",
           "cluster_by_entailment", "compare_predictors", "conformal_abstention", "discrete_semantic_entropy",
           "format_report", "score_case", "selective_summary", "sequence_logprob_score", "split_by_scenario",
           "uncertainty_report", "uncertainty_risk_coverage"]
