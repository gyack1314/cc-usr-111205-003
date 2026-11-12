"""检查带时间隔离的保序校准、选参与外期复核。"""
import copy
import json
import unittest
from fractions import Fraction
from pathlib import Path

from response_data.calibration import (
    analyze, brier, fit_isotonic, final_probability, log_loss, predict_calibrated,
)
from response_data.io import load


def base():
    return load('data/normal.json')[0]


def _row(id_, process, time, p, outcome, weight=1):
    return {'id': id_, 'process': process, 'issued_time': time, 'lead': 1,
            'p': p, 'outcome': outcome, 'weight': weight}


class IsotonicTests(unittest.TestCase):
    def test_block_pooling_and_merged_equal_p(self):
        rows = [
            {'p': Fraction('1/5'), 'weight': Fraction(1), 'positive': Fraction(1), 'outcome': 0},
        ]
        # 直接用已解析的拟合输入：同p不同结局必须共用一个值。
        model = fit_isotonic([
            {'p': Fraction('0'), 'outcome': 0, 'weight': Fraction(1)},
            {'p': Fraction('1/5'), 'outcome': 1, 'weight': Fraction(2)},
            {'p': Fraction('1/5'), 'outcome': 0, 'weight': Fraction(1)},
            {'p': Fraction('7/10'), 'outcome': 1, 'weight': Fraction(2)},
            {'p': Fraction('1'), 'outcome': 0, 'weight': Fraction(1)},
        ])
        knots = model['knots']
        self.assertEqual([float(p) for p in knots], [0.0, 0.2, 0.7, 1.0])
        self.assertEqual(len(model['blocks']), 3)
        # p=0.7 均值 1 高于 p=1 的 0，两者合并为 (2+0)/(2+1)=2/3；
        # p=0.2 与 p=0.7 不再相邻比较，块结构只由训练p与加权均值决定。
        merged = model['blocks'][2]
        self.assertEqual(merged['calibrated'], Fraction(2, 3))
        self.assertEqual(merged['knots'], [Fraction('7/10'), Fraction('1')])
        self.assertEqual(model['mapping'][Fraction('7/10')], merged['calibrated'])
        self.assertEqual(model['mapping'][Fraction('1')], merged['calibrated'])
        self.assertEqual(merged['weight'], Fraction(3))
        self.assertEqual(merged['positive'], Fraction(2))
        self.assertEqual(model['mapping'][Fraction('0')], Fraction(0))
        self.assertEqual(model['mapping'][Fraction('1/5')], Fraction(2, 3))
        self.assertEqual(model['blocks'][1]['calibrated'], Fraction(2, 3))
        self.assertEqual(model['blocks'][1]['weight'], Fraction(3))
        self.assertEqual(model['mapping'][Fraction('1/5')], merged['calibrated'])
        self.assertEqual(model['certificate']['nondecreasing_block_means'], True)
        self.assertEqual(model['certificate']['interval_pooled_mean_between_endpoints'], True)
        self.assertEqual(model['certificate']['violations'], [])
        # 块结构由不同训练p决定，块内同值不可按原行序拆开。
        self.assertEqual(merged['knots'], [Fraction('7/10'), Fraction('1')])

    def test_optimality_certificate_rejects_unpooled_blocks(self):
        rows = [
            {'p': Fraction('0'), 'outcome': 1, 'weight': Fraction(1)},
            {'p': Fraction('1/2'), 'outcome': 0, 'weight': Fraction(1)},
            {'p': Fraction('1'), 'outcome': 0, 'weight': Fraction(1)},
        ]
        model = fit_isotonic(rows)
        # 手工给一个不合并的单调解会违反区间条件，证明证书可区分。
        bad = [dict(weight=Fraction(1), positive=Fraction(1)),
               dict(weight=Fraction(2), positive=Fraction(0))]
        from response_data.calibration import _certificate
        cert = _certificate(bad)
        self.assertFalse(cert['interval_pooled_mean_between_endpoints'])
        self.assertTrue(model['certificate']['interval_pooled_mean_between_endpoints'])

    def test_right_continuous_step_and_support(self):
        model = fit_isotonic([
            {'p': Fraction('0'), 'outcome': 0, 'weight': Fraction(1)},
            {'p': Fraction('1/2'), 'outcome': 1, 'weight': Fraction(1)},
            {'p': Fraction('1'), 'outcome': 1, 'weight': Fraction(1)},
        ])
        self.assertEqual(predict_calibrated(model, 0.2, 'clamp')[0], Fraction(0))
        self.assertEqual(predict_calibrated(model, 0.5, 'clamp')[0], Fraction(1))
        self.assertEqual(predict_calibrated(model, 0.75, 'clamp')[0], Fraction(1))
        self.assertEqual(predict_calibrated(model, 0.5, 'clamp')[1], 'right_continuous_step')

    def test_clamp_and_unestimable_config(self):
        model = fit_isotonic([
            {'p': Fraction('1/5'), 'outcome': 0, 'weight': Fraction(1)},
            {'p': Fraction('2/5'), 'outcome': 1, 'weight': Fraction(1)},
        ])
        self.assertEqual(predict_calibrated(model, 0.1, 'clamp')[0], Fraction(0))
        self.assertEqual(predict_calibrated(model, 0.1, 'clamp')[1], 'clamped_below_support')
        self.assertEqual(predict_calibrated(model, 0.9, 'clamp')[0], Fraction(1))
        self.assertIsNone(predict_calibrated(model, 0.1, 'unestimable')[0])
        self.assertEqual(final_probability(model, 0.1, Fraction(1), 'unestimable')[0], None)
        # gamma=0 时不需要训练块，原始概率始终可估。
        self.assertEqual(final_probability(model, 0.1, Fraction(0), 'unestimable'),
                         (Fraction('1/10'), 'raw_probability'))


class ShrinkTests(unittest.TestCase):
    def test_gamma_zero_is_identity(self):
        model = fit_isotonic([{'p': Fraction(1/5), 'outcome': 1, 'weight': Fraction(1)}])
        self.assertEqual(final_probability(model, Fraction('3/5'), Fraction(0), 'clamp')[0],
                         Fraction('3/5'))
        self.assertEqual(final_probability(model, 0.2, Fraction(1, 2), 'clamp')[0],
                         Fraction(1, 2) * Fraction(1, 5) + Fraction(1, 2) * Fraction(1))

    def test_tie_picks_smaller_gamma_exactly(self):
        # 所有行 p=1/2，训练集加权阳性率始终 1/2，因此 gamma=0 与 gamma=1
        # 的加权Brier在每个验证折上都恰好等于 1/4，属于精确同分。
        def rows_for(name, base_time, outcomes):
            return [_row(f'{name}-{i}', name, base_time + i, 0.5, y)
                    for i, y in enumerate(outcomes)]
        plan = {
            'rows': (rows_for('event0', 0, [1, 0, 1, 0])
                     + rows_for('event1', 100, [1, 1, 0, 0])
                     + rows_for('event2', 200, [1, 0, 0, 1])
                     + rows_for('event3', 300, [0, 1, 1, 0])
                     + rows_for('event4', 400, [1, 0, 1, 0])
                     + rows_for('event5', 500, [0, 1, 0, 1])),
            'development_processes': ['event0', 'event1', 'event2', 'event3'],
            'external_processes': ['event4', 'event5'],
            'folds': [
                {'train': ['event0'], 'validation': ['event1']},
                {'train': ['event0', 'event1'], 'validation': ['event2']},
                {'train': ['event0', 'event1', 'event2'], 'validation': ['event3']},
            ],
            'shrink_candidates': [0, 1],
            'outside_support': 'clamp',
            'bin_edges': [0, 0.25, 0.5, 0.75, 1],
            'bootstrap_draws': 199,
            'seed': 38,
        }
        report = analyze(plan)
        scores = report['selection']['pooled_gamma_scores']
        self.assertEqual({s['brier'] for s in scores}, {0.25})
        self.assertEqual(report['selection']['selected_gamma'], 0.0)
        self.assertEqual(report['selection']['selected_gamma_exact'], '0')


class MetricTests(unittest.TestCase):
    def test_brier_weighted_and_log_infinite(self):
        rows = [
            {'id': 'x', 'weight': Fraction(1), 'outcome': 1},
            {'id': 'y', 'weight': Fraction(3), 'outcome': 0},
        ]
        preds = [Fraction(0), Fraction(1)]
        value, missing = brier(rows, [Fraction(0), Fraction(1)])
        self.assertEqual(value, Fraction(1))
        self.assertEqual(brier(rows, [Fraction(1), Fraction(0)])[0], Fraction(0))
        # 加权：(1*(1/2-1)^2 + 3*(1/4-0)^2)/4 = (1/4+3/16)/4 = 7/64。
        self.assertEqual(brier(rows, [Fraction(1, 2), Fraction(1, 4)])[0], Fraction(7, 64))
        self.assertEqual(missing, [])
        infinite = log_loss(rows, preds)
        self.assertTrue(infinite['infinite'])
        self.assertEqual(infinite['value'], 'inf')
        self.assertEqual(infinite['infinite_rows'], ['x', 'y'])
        # 预测0但事件发生、预测1但事件未发生都应判为无穷（需按行身份核对）。
        named = [
            {'id': 'a', 'weight': Fraction(1), 'outcome': 1},
            {'id': 'b', 'weight': Fraction(1), 'outcome': 0},
        ]
        infinite = log_loss(named, [Fraction(0), Fraction(1)])
        self.assertTrue(infinite['infinite'])
        self.assertEqual(sorted(infinite['infinite_rows']), ['a', 'b'])


class TimeIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = analyze(base())

    def test_process_isolation_and_ordered_folds(self):
        design = self.report['design']
        self.assertFalse(set(design['development']['processes']) & set(design['external']['processes']))
        for fold in design['folds']:
            self.assertFalse(set(fold['train']) & set(fold['validation']))
            self.assertLess(fold['train_last_issued'], fold['validation_first_issued'])
        self.assertTrue(design['process_isolation']['all_training_strictly_before_validation'])

    def test_gamma_selected_on_development_folds_only(self):
        selection = self.report['selection']
        self.assertEqual(selection['selected_gamma'], 1.0)
        # 前推折上的改善必须严格小于开发内重拟合的表观改善。
        outfold = self.report['improvement_attribution']['development_out_of_fold']
        insample = self.report['improvement_attribution']['development_in_sample']
        self.assertLess(outfold['delta']['brier_improvement_raw_minus_final'],
                        insample['delta']['brier_improvement_raw_minus_final'])
        self.assertGreater(insample['delta']['brier_improvement_raw_minus_final'], 0)

    def test_selection_equals_out_of_fold_pooled_metrics(self):
        selection = self.report['selection']
        outfold = self.report['improvement_attribution']['development_out_of_fold']
        self.assertEqual(selection['raw_brier'], outfold['raw']['brier'])
        self.assertEqual(selection['raw_brier'],
                         self.report['selection']['pooled_gamma_scores'][0]['brier'])
        # 选中的gamma与折外汇总Brier必须一致，避免描述性块外指标另行拼凑。
        self.assertEqual(selection['selected_brier'], outfold['final']['brier'])

    def test_external_outcomes_do_not_move_mapping_or_selection(self):
        plan = base()
        changed = copy.deepcopy(plan)
        for row in changed['rows']:
            if row['process'] in changed['external_processes']:
                row['outcome'] = 1 - row['outcome']
        other = analyze(changed)
        self.assertEqual(self.report['final_mapping']['mapping_sha256'],
                         other['final_mapping']['mapping_sha256'])
        self.assertEqual(self.report['selection']['selected_gamma'],
                         other['selection']['selected_gamma'])
        self.assertEqual(self.report['improvement_attribution']['development_out_of_fold']['delta'],
                         other['improvement_attribution']['development_out_of_fold']['delta'])
        # 外期行本身可复算：结局改动只影响外期评分。
        self.assertNotEqual(self.report['external']['raw']['brier'],
                            other['external']['raw']['brier'])

    def test_external_bins_locked_and_empty_kept(self):
        report = self.report['external']['final']['calibration']
        self.assertEqual([b['lower'] for b in report['bins']], [0.0, 0.25, 0.5, 0.75])
        self.assertEqual(report['bins'][-1]['upper'], 1.0)
        self.assertTrue(report['bins'][-1]['includes_upper_endpoint'])
        self.assertEqual(len(report['bins']), 4)
        self.assertTrue(any(b['rows'] == 0 for b in report['bins']))

    def test_log_loss_infinite_not_clipped(self):
        raw = self.report['external']['raw']['log_loss']
        self.assertTrue(raw['infinite'])
        self.assertEqual(raw['value'], 'inf')
        self.assertEqual(sorted(raw['infinite_rows']), ['p4-0', 'p5-0', 'p5-4'])
        final = self.report['external']['final']['log_loss']
        self.assertFalse(final['infinite'])
        self.assertAlmostEqual(final['value'], 0.6589441129167696, places=12)
        self.assertEqual(
            self.report['improvement_attribution']['external_holdout']['delta']['log_loss_final_minus_raw'],
            {'value': None, 'state': 'infinite', 'final_infinite': False, 'raw_infinite': True})

    def test_bootstrap_uses_frozen_mapping_and_process_units(self):
        boot = self.report['external']['bootstrap']
        self.assertTrue(boot['mapping_frozen'])
        self.assertFalse(boot['refit_during_bootstrap'])
        self.assertEqual(boot['unit'], 'process')
        self.assertEqual(boot['draws'], 199)
        self.assertIn('不含选参', boot['scope'])
        self.assertLess(boot['brier_difference']['high'], 0)
        self.assertEqual(boot['brier_difference']['finite_draws'], 199)

    def test_row_order_does_not_change_results(self):
        shuffled = base()
        shuffled['rows'] = list(reversed(shuffled['rows']))
        other = analyze(shuffled)
        self.assertEqual(self.report['final_mapping']['mapping_sha256'],
                         other['final_mapping']['mapping_sha256'])
        self.assertEqual(self.report['external'], other['external'])

    def test_report_is_json_and_repeatable(self):
        text = json.dumps(self.report, ensure_ascii=False, allow_nan=False)
        self.assertNotIn('NaN', text)
        self.assertEqual(json.loads(text), self.report)
        self.assertEqual(analyze(base()), self.report)


class BoundaryMaterialTests(unittest.TestCase):
    def test_single_outcome(self):
        report = analyze(load('data/single-outcome.json')[0])
        self.assertEqual(report['selection']['selected_gamma'], 1.0)
        self.assertEqual(report['external']['final']['brier'], 0.0)
        self.assertEqual(report['final_mapping']['blocks'][0]['calibrated'], 0.0)
        self.assertEqual(report['final_mapping']['blocks'][0]['p_low'], 0.0)
        self.assertEqual(report['final_mapping']['blocks'][-1]['p_high'], 1.0)

    def test_outside_support_reject(self):
        report = analyze(load('data/outside-support-reject.json')[0])
        self.assertEqual(report['config']['outside_support'], 'unestimable')
        self.assertEqual(report['final_mapping']['support'], [0.2, 0.8])
        self.assertEqual(sorted(report['external']['unestimable_rows']), ['p4-0', 'p4-4'])
        self.assertIsNone(report['external']['final']['brier'])
        self.assertIsNone(report['external']['final']['calibration'])
        self.assertEqual(report['external']['final']['log_loss']['value'], None)
        boot = report['external']['bootstrap']['brier_difference']
        self.assertEqual(boot['finite_draws'], 43)
        self.assertEqual(boot['nonfinite_draws'], 156)
        self.assertIsNotNone(boot['low'])
        self.assertGreater(boot['nonfinite_draws'], 0)
        self.assertLessEqual(boot['finite_draws'], boot['draws'])

    def test_levels_are_finite_or_string(self):
        for path in Path('data').glob('*.json'):
            if path.name.startswith('invalid-'):
                continue
            with self.subTest(file=path.name):
                report = analyze(load(path)[0])
                json.dumps(report, ensure_ascii=False, allow_nan=False)


class ValidationErrorTests(unittest.TestCase):
    def test_rejects_bad_inputs(self):
        cases = []
        plan = base(); plan['development_processes'] = ['event0', 'event4']
        cases.append(plan)
        plan = base(); plan['folds'][0]['train'] = ['event1']
        plan['folds'][0]['validation'] = ['event1']
        cases.append(plan)
        plan = base(); plan['folds'][0]['train'] = ['event2']
        plan['folds'][0]['validation'] = ['event1']
        plan['folds'][0]['train'] = ['event1']
        cases.append(plan)
        plan = base(); plan['shrink_candidates'] = [0, 1.5]
        cases.append(plan)
        plan = base(); plan['bin_edges'] = [0, 0.5, 0.5, 1]
        cases.append(plan)
        plan = base(); plan['bin_edges'] = [0.1, 0.5, 1]
        cases.append(plan)
        plan = base(); plan['outside_support'] = 'extrapolate'
        cases.append(plan)
        plan = base(); plan['rows'][0]['p'] = 1.5
        cases.append(plan)
        plan = base(); plan['rows'][0]['weight'] = 0
        cases.append(plan)
        plan = base(); plan['rows'][0]['outcome'] = 2
        cases.append(plan)
        plan = base(); plan['folds'] = []
        cases.append(plan)
        for case in cases:
            with self.assertRaises((ValueError, KeyError, TypeError)):
                analyze(case)


if __name__ == '__main__':
    unittest.main()
