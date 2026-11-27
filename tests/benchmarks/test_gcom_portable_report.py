"""Frozen estimands, linear percentiles and independent batch realization."""
from pathlib import Path
import pytest
from amora.benchmarking.portable_v2.gcom_report import scores
from amora.backends.nvidia.portable_v2_batch import batch_source


def row(native,predicted,eligible=True):
    return dict(native_ms=native,prediction_ms=predicted,timing_qualified=eligible)


def test_percentile_is_linear_and_ineligible_rows_have_no_weight():
    result=scores([row(1,1.01),row(1,1.02),row(1,1.03),row(1,1.04),row(1,100,False)])
    assert result["total_cases"]==5 and result["eligible_cases"]==4
    assert result["mdape"]==pytest.approx(2.5)
    assert result["p95_ape"]==pytest.approx(3.85)
    assert result["pearson"] is None


def test_rank_correlation_handles_ties_and_preserves_bias():
    result=scores([row(1,2),row(1,2),row(3,6),row(4,8)])
    assert result["spearman"]==pytest.approx(1)
    assert result["signed_median_bias"]==pytest.approx(100)
    assert result["mdape"]==pytest.approx(100)


def test_empty_or_too_small_cohort_has_no_correlation():
    assert scores([])["status"]=="not_evaluable"
    result=scores([row(1,2),row(2,3)])
    assert result["pearson"] is None and result["spearman"] is None


def test_batch_variant_preserves_all_kernel_math_and_oracle_code():
    source=(Path(__file__).parents[2]/"amora/backends/nvidia/portable_core_v2.cu").read_text()
    variant=batch_source(source)
    boundary="int main(int argc,char **argv)"
    assert source.split(boundary)[0]==variant.split(boundary)[0]
    assert 'for(int repetition=0;repetition<batch;repetition++)execute(c);' in variant
    assert 'if(name=="--batch")' in variant
    assert 'execute(c);check_cuda(cudaEventRecord(end),"event end");' in variant
    assert 'auto before=compare(c,root)' in variant and 'auto after=compare(c,root)' in variant


def test_batch_adapter_rejects_changed_driver_boundaries():
    with pytest.raises(ValueError):batch_source("unknown source")
