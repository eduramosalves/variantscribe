from variantscribe.eval.calibration import (
    expected_calibration_error,
    reliability_table,
)
from variantscribe.models import ClinSig, EvalCase


def case(correct, conf):
    gold = ClinSig.PATHOGENIC
    pred = ClinSig.PATHOGENIC if correct else ClinSig.BENIGN
    return EvalCase(variation_id="x", gold=gold, predicted=pred, abstained=False, confidence=conf)


def test_perfect_calibration_has_zero_ece():
    # 0.9-confidence bucket that is correct 100% of the time... mismatch -> not zero.
    # Build a perfectly-calibrated set: confidence 1.0 always correct, 0.0 always wrong.
    cases = [case(True, 1.0) for _ in range(5)] + [case(False, 0.0) for _ in range(5)]
    assert expected_calibration_error(cases) == 0.0


def test_overconfident_has_positive_ece():
    # says 0.9 but only 50% correct -> gap ~0.4
    cases = [case(True, 0.9) for _ in range(5)] + [case(False, 0.9) for _ in range(5)]
    ece = expected_calibration_error(cases)
    assert ece is not None and 0.35 < ece < 0.45


def test_no_confidence_returns_none():
    c = EvalCase(variation_id="x", gold=ClinSig.VUS, predicted=ClinSig.VUS, abstained=False)
    assert expected_calibration_error([c]) is None


def test_abstentions_excluded_from_calibration():
    abstained = EvalCase(
        variation_id="a", gold=ClinSig.VUS, predicted=None, abstained=True, confidence=0.9
    )
    cases = [abstained, case(True, 1.0)]
    table = reliability_table(cases)
    assert sum(b.n for b in table) == 1  # only the answered case counts


def test_reliability_table_bins_and_accuracy():
    cases = [case(True, 0.85), case(False, 0.85), case(True, 0.95)]
    table = reliability_table(cases, bins=5)
    top = [b for b in table if b.label == "0.8-1.0"]
    assert top and top[0].n == 3
    assert abs(top[0].accuracy - 2 / 3) < 1e-9
