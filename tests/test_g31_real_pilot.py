import numpy as np

from soul.g31.real_pilot import run_real_pilot


def test_real_pilot_completes_and_stays_exploratory():
    report = run_real_pilot()

    assert report['pilot_classification'] == 'INCONCLUSIVE'
    assert report['threshold'] == 'TO_LOCK'
    assert report['confirm'] == 'NOT RUN — created but untouched'
    assert report['g31_capability'] == 'NOT ESTABLISHED'
    assert report['integrity']['no_confirm_evaluation'] is True
    assert report['dataset']['pilot_transitions'] == 20
    assert set(report['metrics_pilot']) == {
        'model_placeholder', 'persistence', 'mean_delta_baseline'
    }
    for aggregate in (report['metrics_pilot']['model_placeholder'],
                      report['metrics_pilot']['persistence']):
        for values in aggregate.values():
            assert np.isfinite(values['mean'])
