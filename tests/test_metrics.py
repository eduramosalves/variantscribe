from variantscribe.eval.metrics import evaluate
from variantscribe.models import ClinSig, EvalCase

P, LP, VUS, LB, B = (
    ClinSig.PATHOGENIC,
    ClinSig.LIKELY_PATHOGENIC,
    ClinSig.VUS,
    ClinSig.LIKELY_BENIGN,
    ClinSig.BENIGN,
)


def case(gold, pred, conf=None):
    return EvalCase(
        variation_id="x",
        gold=gold,
        predicted=pred,
        abstained=pred is None,
        confidence=conf,
    )


def test_perfect_predictions():
    cases = [case(P, P), case(B, B), case(VUS, VUS)]
    r = evaluate(cases)
    assert r.accuracy == 1.0
    assert r.coverage == 1.0
    assert r.dangerous_errors == 0
    assert r.ordinal_mae == 0.0


def test_dangerous_error_is_counted():
    # gold Pathogenic, predicted Benign -> the clinically dangerous direction
    r = evaluate([case(P, B)])
    assert r.dangerous_errors == 1
    assert r.dangerous_error_rate == 1.0
    assert r.ordinal_mae == 4.0  # full span of the 5-tier scale


def test_benign_called_pathogenic_is_not_dangerous():
    # wrong, and penalised by MAE, but not the dangerous (under-call) direction
    r = evaluate([case(B, P)])
    assert r.dangerous_errors == 0
    assert r.ordinal_mae == 4.0


def test_abstention_lowers_coverage_not_accuracy():
    cases = [case(P, P), case(B, None)]
    r = evaluate(cases)
    assert r.n_answered == 1
    assert r.coverage == 0.5
    assert r.accuracy == 1.0  # the one answered case was correct


def test_three_class_collapse_is_lenient_on_adjacent_tiers():
    # P vs LP differ on the 5-tier scale but collapse to the same actionable group
    r = evaluate([case(P, LP)])
    assert r.accuracy == 0.0
    assert r.three_class_accuracy == 1.0


def test_empty_answered_set_is_safe():
    r = evaluate([case(P, None)])
    assert r.n_answered == 0
    assert r.accuracy == 0.0


def test_calibration_buckets():
    cases = [case(P, P, conf=0.9), case(B, P, conf=0.9), case(VUS, VUS, conf=0.1)]
    r = evaluate(cases)
    # high-confidence bucket has 2 cases at 0.5 accuracy
    high = [b for b in r.calibration if b[0] == "0.8-1.0"]
    assert high and high[0][1] == 2 and high[0][2] == 0.5
