from variantscribe.agent.telemetry import summarize_run
from variantscribe.eval.sampling import stratified_sample
from variantscribe.models import Classification, ClinSig, GoldRecord, Variant


def _pred(i_tok, o_tok, lat):
    return Classification(
        variation_id="x",
        call=ClinSig.VUS,
        input_tokens=i_tok,
        output_tokens=o_tok,
        latency_ms=lat,
    )


def test_summarize_run_costs_and_latency():
    preds = [_pred(1000, 500, 100.0), _pred(3000, 500, 300.0)]
    t = summarize_run(preds, model="claude-sonnet-4-6")  # $3/$15 per Mtok
    assert t.input_tokens == 4000 and t.output_tokens == 1000
    # 4000/1e6*3 + 1000/1e6*15 = 0.012 + 0.015 = 0.027
    assert round(t.est_cost_usd, 4) == 0.027
    assert t.latency_ms_p50 == 200.0


def test_summarize_run_unknown_model_uses_default_rate():
    t = summarize_run([_pred(1_000_000, 0, 50.0)], model="mystery")
    assert t.est_cost_usd == 3.0  # default input rate $3/Mtok


def _gold(tier, n):
    return [
        GoldRecord(variant=Variant(gene="BRCA1", variation_id=f"{tier.value}-{i}"),
                   gold=tier, review_status="x", gold_stars=2)
        for i in range(n)
    ]


def test_stratified_sample_covers_all_tiers():
    gold = _gold(ClinSig.PATHOGENIC, 100) + _gold(ClinSig.BENIGN, 50) + _gold(ClinSig.VUS, 3)
    sample = stratified_sample(gold, 30, seed=1)
    assert len(sample) == 30
    tiers = {g.gold for g in sample}
    assert ClinSig.VUS in tiers  # rare tier still represented
    assert ClinSig.PATHOGENIC in tiers and ClinSig.BENIGN in tiers


def test_stratified_sample_returns_all_when_n_exceeds():
    gold = _gold(ClinSig.PATHOGENIC, 5)
    assert len(stratified_sample(gold, 99)) == 5
