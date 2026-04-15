import numpy as np
from scipy import stats
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from benchmark.types import RunMeasurement

@dataclass
class MetricSummary:
    metric_name: str
    framework: str
    task_id: str
    n: int
    mean: float
    median: float
    std_dev: float
    min: float
    max: float
    p5: float
    p95: float
    ci_lower: float
    ci_upper: float
    cv: float

@dataclass
class PairwiseComparison:
    metric_name: str
    task_id: str
    framework_a: str
    framework_b: str
    mean_a: float
    mean_b: float
    difference: float
    percent_difference: float
    p_value: float
    is_significant: bool
    effect_size: float
    effect_magnitude: str
    winner: Optional[str]
    caveat: Optional[str]

def calculate_summary(data: List[float], framework: str, task_id: str, metric_name: str) -> MetricSummary:
    n = len(data)
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data, ddof=1) if n > 1 else 0.0
    
    # 95% Confidence Interval using t-distribution
    if n > 1:
        se = std_dev / np.sqrt(n)
        ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
        ci_lower, ci_upper = ci[0], ci[1]
    else:
        ci_lower, ci_upper = mean, mean
        
    return MetricSummary(
        metric_name=metric_name,
        framework=framework,
        task_id=task_id,
        n=n,
        mean=mean,
        median=median,
        std_dev=std_dev,
        min=np.min(data),
        max=np.max(data),
        p5=np.percentile(data, 5),
        p95=np.percentile(data, 95),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        cv=std_dev / mean if mean != 0 else 0.0
    )

def compare_frameworks(data_a: List[float], data_b: List[float], framework_a: str, framework_b: str, task_id: str, metric_name: str) -> PairwiseComparison:
    mean_a = np.mean(data_a)
    mean_b = np.mean(data_b)
    diff = mean_a - mean_b
    pct_diff = (diff / mean_b * 100) if mean_b != 0 else 0.0

    # Welch's t-test (unequal variances)
    t_stat, p_value = stats.ttest_ind(data_a, data_b, equal_var=False)

    # Effect Size (Cohen's d)
    pooled_std = np.sqrt(((len(data_a)-1)*np.var(data_a) + (len(data_b)-1)*np.var(data_b)) / (len(data_a) + len(data_b) - 2))
    d = diff / pooled_std if pooled_std != 0 else 0.0
    
    if abs(d) < 0.2: mag = "negligible"
    elif abs(d) < 0.5: mag = "small"
    elif abs(d) < 0.8: mag = "medium"
    else: mag = "large"

    is_sig = p_value < 0.05
    winner = None
    if is_sig:
        # Assuming lower is better for latency/cost/tokens
        winner = framework_a if diff < 0 else framework_b
        
    return PairwiseComparison(
        metric_name=metric_name,
        task_id=task_id,
        framework_a=framework_a,
        framework_b=framework_b,
        mean_a=mean_a,
        mean_b=mean_b,
        difference=diff,
        percent_difference=pct_diff,
        p_value=p_value,
        is_significant=is_sig,
        effect_size=d,
        effect_magnitude=mag,
        winner=winner,
        caveat=None
    )
