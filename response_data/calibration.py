"""带时间隔离的加权保序校准、收缩选参和外期复核。

内部用 Fraction 保存概率、权重与块均值，保证同分判定和加权平方误差可复算。
本模块只接收字段已经齐全的资料；拟合时只传入开发窗（或折内训练过程）的行，
外期行仅在最终冻结映射上评分。
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from bisect import bisect_right
from fractions import Fraction

from .io import integer, number, unique


def _exact(x):
    """读取有限数值并转成十进制精确有理数（0.2 即 1/5）。"""
    number(x)
    return Fraction(str(x))


def _exact_or_fraction(x):
    """预测入口接受原始数值或已转换的Fraction。"""
    return x if isinstance(x, Fraction) else _exact(x)


def _num(x):
    return None if x is None else float(x)


def _score(x):
    return None if x is None else float(x)


def _row(r):
    if not isinstance(r, dict):
        raise ValueError('每行必须为对象')
    if not isinstance(r.get('id'), str) or not r['id']:
        raise ValueError('行身份必须为非空字符串')
    if not isinstance(r.get('process'), str) or not r['process']:
        raise ValueError('过程身份必须为非空字符串')
    p = _exact(r.get('p'))
    if not 0 <= p <= 1:
        raise ValueError('概率必须位于[0,1]')
    outcome = r.get('outcome')
    if type(outcome) is not int or outcome not in (0, 1):
        raise ValueError('结局必须为整数0或1')
    w = _exact(r.get('weight'))
    if w <= 0:
        raise ValueError('权重必须为正')
    issued = integer(r.get('issued_time'), 0)
    lead = number(r.get('lead'))
    if lead < 0:
        raise ValueError('提前期不能为负')
    return {
        'id': r['id'], 'process': r['process'], 'issued_time': issued,
        'lead': lead, 'p': p, 'outcome': outcome, 'weight': w,
    }


def _name_list(value, field, allowed=None):
    if not isinstance(value, list) or not value:
        raise ValueError(f'{field}必须为非空过程列表')
    names = []
    for x in value:
        if not isinstance(x, str) or not x:
            raise ValueError(f'{field}中的过程身份必须为非空字符串')
        if x in names:
            raise ValueError(f'{field}中的过程身份不得重复')
        if allowed is not None and x not in allowed:
            raise ValueError(f'{field}只能使用开发过程')
        names.append(x)
    return names


def _raw_list(plan, field):
    value = plan.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f'{field}必须为非空列表')
    return value


def _prepare(plan):
    if not isinstance(plan, dict):
        raise ValueError('顶层资料必须为对象')
    rows = _raw_list(plan, 'rows')
    if not isinstance(rows, list):
        raise ValueError('rows必须为列表')
    unique(rows, 'id')
    rows = [_row(r) for r in rows]
    by_process = {}
    for r in rows:
        by_process.setdefault(r['process'], []).append(r)
    if not rows:
        raise ValueError('资料没有行')

    dev = _name_list(plan.get('development_processes'), 'development_processes')
    ext = _name_list(plan.get('external_processes'), 'external_processes')
    if set(dev) & set(ext):
        raise ValueError('开发外期过程交叠')
    if set(by_process) != set(dev) | set(ext):
        raise ValueError('过程分组必须完整覆盖全部行且不得引用无资料过程')

    folds = _raw_list(plan, 'folds')
    if not isinstance(folds, list):
        raise ValueError('folds必须为列表')
    fold_sets = []
    for i, fold in enumerate(folds):
        if not isinstance(fold, dict):
            raise ValueError('每折必须为对象')
        train = _name_list(fold.get('train'), f'folds[{i}].train', set(dev))
        validation = _name_list(fold.get('validation'), f'folds[{i}].validation', set(dev))
        if set(train) & set(validation):
            raise ValueError('同折训练与验证过程交叠')
        train_last = max(r['issued_time'] for name in train for r in by_process[name])
        validation_first = min(r['issued_time'] for name in validation for r in by_process[name])
        if not train_last < validation_first:
            raise ValueError('训练过程必须整体严格早于验证过程')
        fold_sets.append({
            'index': i, 'train': train, 'validation': validation,
            'train_last_issued': train_last, 'validation_first_issued': validation_first,
        })

    outside = plan.get('outside_support')
    if outside not in ('clamp', 'unestimable'):
        raise ValueError('outside_support只能为clamp或unestimable')

    candidates_raw = _raw_list(plan, 'shrink_candidates')
    gammas = []
    for i, x in enumerate(candidates_raw):
        g = _exact(x)
        if not 0 <= g <= 1:
            raise ValueError(f'shrink_candidates[{i}]必须位于[0,1]')
        gammas.append(g)
    if not gammas:
        raise ValueError('shrink候选不能为空')

    edges_raw = _raw_list(plan, 'bin_edges')
    if not isinstance(edges_raw, list) or len(edges_raw) < 2:
        raise ValueError('bin_edges至少需要两个边界')
    edges = [_exact(x) for x in edges_raw]
    if edges[0] != 0 or edges[-1] != 1:
        raise ValueError('锁定分箱必须覆盖[0,1]')
    if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
        raise ValueError('分箱边界必须严格递增')

    draws = integer(plan.get('bootstrap_draws'), 0)
    seed = integer(plan.get('seed'), 0)

    dev_rows = [r for r in rows if r['process'] in set(dev)]
    ext_rows = [r for r in rows if r['process'] in set(ext)]
    # 以下排序只用于稳定输出与浮点求和；拟合按不同 p 聚合，不使用原行序。
    dev_rows.sort(key=lambda r: (r['p'], r['id']))
    ext_rows.sort(key=lambda r: (r['process'], r['p'], r['id']))
    return {
        'dev': dev, 'ext': ext, 'dev_rows': dev_rows, 'ext_rows': ext_rows,
        'folds': fold_sets, 'outside': outside, 'gammas': gammas, 'edges': edges,
        'draws': draws, 'seed': seed,
    }


def fit_isotonic(rows):
    """加权保序回归：同 p 先合并，块均值随 p 单调不降。"""
    groups = {}
    for r in rows:
        g = groups.setdefault(r['p'], {'weight': Fraction(0), 'positive': Fraction(0)})
        g['weight'] += r['weight']
        g['positive'] += r['weight'] * r['outcome']
    if not groups:
        raise ValueError('没有可用于拟合的行')

    blocks = []
    for p in sorted(groups):
        g = groups[p]
        blocks.append({
            'p_low': p, 'p_high': p, 'knots': [p],
            'weight': g['weight'], 'positive': g['positive'],
        })
        while len(blocks) > 1:
            left, right = blocks[-2], blocks[-1]
            if left['positive'] / left['weight'] <= right['positive'] / right['weight']:
                break
            blocks[-2:] = [{
                'p_low': left['p_low'], 'p_high': right['p_high'],
                'knots': left['knots'] + right['knots'],
                'weight': left['weight'] + right['weight'],
                'positive': left['positive'] + right['positive'],
            }]

    mapping = {}
    block_of_knot = {}
    for index, b in enumerate(blocks):
        b['index'] = index
        b['calibrated'] = b['positive'] / b['weight']
        for p in b['knots']:
            mapping[p] = b['calibrated']
            block_of_knot[p] = index

    knots = sorted(mapping)
    return {
        'blocks': blocks, 'mapping': mapping, 'block_of_knot': block_of_knot,
        'knots': knots, 'support': (knots[0], knots[-1]),
        'certificate': _certificate(blocks),
    }


def _certificate(blocks):
    """保存可人工核对的最优性条件。

    对加权平方损失，单调块均值为最优当且仅当块均值不降，且任意连续块区间
    [i,j] 的合并阳性权重/合并总权重落在 q_i 与 q_j 之间。
    """
    means = [b['positive'] / b['weight'] for b in blocks]
    monotone = all(means[i] <= means[i + 1] for i in range(len(means) - 1))
    intervals = 0
    violations = []
    for i in range(len(blocks)):
        weight = Fraction(0)
        positive = Fraction(0)
        for j in range(i, len(blocks)):
            weight += blocks[j]['weight']
            positive += blocks[j]['positive']
            pooled = positive / weight
            intervals += 1
            if not means[i] <= pooled <= means[j]:
                violations.append({'left': i, 'right': j, 'pooled': _score(pooled),
                                   'left_mean': _score(means[i]), 'right_mean': _score(means[j])})
    return {
        'condition': 'q_i <= sum(positive_i..j)/sum(weight_i..j) <= q_j for every i<=j',
        'nondecreasing_block_means': monotone,
        'interval_pooled_mean_between_endpoints': not violations,
        'intervals_checked': intervals,
        'violations': violations,
    }


def predict_calibrated(model, p, outside):
    """右连续阶梯：取最大训练结点<=p；支撑外按配置夹持或不可估。"""
    x = _exact_or_fraction(p)
    support_low, support_high = model['support']
    if x < support_low:
        if outside == 'unestimable':
            return None, 'unestimable_below_support'
        return model['mapping'][support_low], 'clamped_below_support'
    if x > support_high:
        if outside == 'unestimable':
            return None, 'unestimable_above_support'
        return model['mapping'][support_high], 'clamped_above_support'
    index = bisect_right(model['knots'], x) - 1
    return model['mapping'][model['knots'][index]], 'right_continuous_step'


def final_probability(model, p, gamma, outside):
    """(1-gamma)*p + gamma*isotonic(p)。gamma=0 时不需要训练块。"""
    x = _exact_or_fraction(p)
    if gamma == 0:
        return x, 'raw_probability'
    q, status = predict_calibrated(model, x, outside)
    if q is None:
        return None, status
    return (1 - gamma) * x + gamma * q, status


def brier(rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError('预测数量与行数不一致')
    missing = [r['id'] for r, q in zip(rows, predictions) if q is None]
    if missing:
        return None, missing
    total = sum((r['weight'] for r in rows), Fraction(0))
    if total == 0:
        raise ValueError('总权重必须为正')
    score = sum((r['weight'] * (q - r['outcome']) ** 2 for r, q in zip(rows, predictions)), Fraction(0))
    return score / total, []


def log_loss(rows, predictions):
    """加权log评分；端点预测与结局相反时明确返回无穷，不做裁剪。"""
    if len(rows) != len(predictions):
        raise ValueError('预测数量与行数不一致')
    missing = [r['id'] for r, q in zip(rows, predictions) if q is None]
    if missing:
        return {'value': None, 'infinite': False, 'unestimable': True,
                'unestimable_rows': missing, 'infinite_rows': []}
    total = sum((r['weight'] for r in rows), Fraction(0))
    if total == 0:
        raise ValueError('总权重必须为正')
    terms = []
    infinite = []
    for r, q in zip(rows, predictions):
        if r['outcome'] == 1:
            if q == 0:
                infinite.append(r.get('id'))
                continue
            value = -math.log(float(q))
        else:
            if q == 1:
                infinite.append(r.get('id'))
                continue
            value = -math.log1p(-float(q))
        terms.append(r['weight'].__float__() * value)
    if infinite:
        return {'value': 'inf', 'infinite': True, 'unestimable': False,
                'unestimable_rows': [], 'infinite_rows': infinite}
    return {'value': math.fsum(terms) / float(total), 'infinite': False,
            'unestimable': False, 'unestimable_rows': [], 'infinite_rows': []}


def calibration(rows, predictions, edges):
    """按锁定边界计算ECE；空箱保留，边界绝不随资料改动。"""
    if len(rows) != len(predictions):
        raise ValueError('预测数量与行数不一致')
    missing = [r['id'] for r, q in zip(rows, predictions) if q is None]
    if missing:
        return {'ece': None, 'bins': [], 'unestimable_rows': missing}
    total = sum((r['weight'] for r in rows), Fraction(0))
    if total == 0:
        raise ValueError('总权重必须为正')
    bins = [{'lower': _num(edges[i]), 'upper': _num(edges[i + 1]),
             'rows': 0, 'weight': Fraction(0),
             'predicted_sum': Fraction(0), 'positive_sum': Fraction(0)}
            for i in range(len(edges) - 1)]
    for r, q in zip(rows, predictions):
        bin_index = None
        for i in range(len(edges) - 1):
            last = i == len(edges) - 2
            if edges[i] <= q < edges[i + 1] or (last and q == edges[i + 1]):
                bin_index = i
                break
        if bin_index is None:
            raise ValueError('预测值超出锁定分箱')
        b = bins[bin_index]
        b['rows'] += 1
        b['weight'] += r['weight']
        b['predicted_sum'] += r['weight'] * q
        b['positive_sum'] += r['weight'] * r['outcome']
    ece = Fraction(0)
    out = []
    for b in bins:
        if b['weight'] == 0:
            observed = predicted = gap = contribution = None
        else:
            observed = b['positive_sum'] / b['weight']
            predicted = b['predicted_sum'] / b['weight']
            gap = abs(observed - predicted)
            contribution = b['weight'] / total * gap
            ece += contribution
        out.append({
            'lower': b['lower'], 'upper': b['upper'],
            'includes_upper_endpoint': b['upper'] == 1,
            'rows': b['rows'], 'weight': _score(b['weight']),
            'observed_rate': _score(observed),
            'predicted_mean': _score(predicted),
            'absolute_gap': _score(gap),
            'ece_contribution': _score(contribution),
        })
    return {'ece': _score(ece), 'bins': out, 'unestimable_rows': []}


def metrics(rows, predictions, edges):
    """汇总一组同口径指标；任一行不可估时该组指标为空。"""
    b, missing = brier(rows, predictions)
    if missing:
        return {
            'rows': len(rows), 'weight': _score(sum((r['weight'] for r in rows), Fraction(0))),
            'brier': None, 'calibration': None,
            'log_loss': {'value': None, 'infinite': False, 'unestimable': True,
                         'unestimable_rows': missing, 'infinite_rows': []},
            'unestimable_rows': missing,
        }
    return {
        'rows': len(rows), 'weight': _score(sum((r['weight'] for r in rows), Fraction(0))),
        'brier': _score(b),
        'calibration': calibration(rows, predictions, edges),
        'log_loss': log_loss(rows, predictions),
        'unestimable_rows': [],
    }


def _delta(final, raw):
    if final is None or raw is None:
        return None
    return final - raw


def _log_delta(final_metrics, raw_metrics):
    """最终-原始log损失差；任一侧为无穷或不可估时不做算术，明确记录原因。"""
    f = final_metrics.get('log_loss') or {}
    r = raw_metrics.get('log_loss') or {}
    if f.get('unestimable') or r.get('unestimable'):
        return {'value': None, 'state': 'unestimable'}
    if f.get('infinite') or r.get('infinite'):
        return {'value': None, 'state': 'infinite', 'final_infinite': bool(f.get('infinite')),
                'raw_infinite': bool(r.get('infinite'))}
    return {'value': f['value'] - r['value'], 'state': 'finite'}


def _delta_report(final_metrics, raw_metrics):
    return {
        'brier_final_minus_raw': _delta(final_metrics.get('brier'), raw_metrics.get('brier')),
        'brier_improvement_raw_minus_final':
            None if final_metrics.get('brier') is None or raw_metrics.get('brier') is None
            else raw_metrics['brier'] - final_metrics['brier'],
        'ece_final_minus_raw':
            None if final_metrics.get('calibration') is None or raw_metrics.get('calibration') is None
            else _delta(final_metrics['calibration']['ece'], raw_metrics['calibration']['ece']),
        'log_loss_final_minus_raw': _log_delta(final_metrics, raw_metrics),
    }


def _fold_models(state):
    out = []
    for fold in state['folds']:
        train_rows = [r for r in state['dev_rows'] if r['process'] in set(fold['train'])]
        validation_rows = [r for r in state['dev_rows'] if r['process'] in set(fold['validation'])]
        model = fit_isotonic(train_rows)
        out.append({'fold': fold, 'model': model, 'train_rows': train_rows,
                    'validation_rows': validation_rows})
    return out


def _fold_selection(state, folds):
    gammas = state['gammas']
    outside = state['outside']
    per_gamma = {g: {'numerator': Fraction(0), 'denominator': Fraction(0),
                     'unestimable': [], 'defined': True} for g in gammas}
    fold_rows = []
    raw_numerator = Fraction(0)
    raw_denominator = Fraction(0)
    for fm in folds:
        validation = fm['validation_rows']
        total_weight = sum((r['weight'] for r in validation), Fraction(0))
        raw_denominator += total_weight
        raw_preds = [r['p'] for r in validation]
        raw_score = sum((r['weight'] * (r['p'] - r['outcome']) ** 2 for r in validation), Fraction(0))
        raw_numerator += raw_score
        fold_report = {
            'index': fm['fold']['index'],
            'train': fm['fold']['train'],
            'validation': fm['fold']['validation'],
            'train_last_issued': fm['fold']['train_last_issued'],
            'validation_first_issued': fm['fold']['validation_first_issued'],
            'training_support': [_num(fm['model']['support'][0]), _num(fm['model']['support'][1])],
            'validation_rows': len(validation),
            'validation_weight': _score(total_weight),
            'raw_brier': _score(raw_score / total_weight),
            'gamma_scores': [],
            'training_blocks': _blocks_report(fm['model']),
            'training_certificate': fm['model']['certificate'],
        }
        for g in gammas:
            nums = []
            missing = []
            for r in validation:
                q, _ = final_probability(fm['model'], r['p'], g, outside)
                if q is None:
                    missing.append(r['id'])
                else:
                    nums.append(r['weight'] * (q - r['outcome']) ** 2)
            if missing:
                score = None
                per_gamma[g]['defined'] = False
                per_gamma[g]['unestimable'].extend(missing)
            else:
                score = sum(nums, Fraction(0)) / total_weight
                per_gamma[g]['numerator'] += sum(nums, Fraction(0))
                per_gamma[g]['denominator'] += total_weight
            fold_report['gamma_scores'].append({
                'gamma': _score(g), 'gamma_exact': str(g),
                'brier': _score(score), 'unestimable_rows': missing,
            })
        fold_rows.append(fold_report)

    pooled_scores = []
    for g in gammas:
        p = per_gamma[g]
        if not p['defined'] or p['denominator'] == 0:
            score = None
        else:
            score = p['numerator'] / p['denominator']
        pooled_scores.append({
            'gamma': _score(g), 'gamma_exact': str(g), 'brier': _score(score),
            'unestimable_rows': p['unestimable'],
        })
    feasible = [(g, _exact_score(per_gamma[g])) for g in gammas
                if per_gamma[g]['defined'] and per_gamma[g]['denominator'] != 0]
    if not feasible:
        raise ValueError('所有收缩候选在开发内前推折上均不可估')
    # 同分选较小gamma；分数用Fraction精确比较。
    selected, selected_score = min(feasible, key=lambda item: (item[1], item[0]))
    raw_total = raw_numerator / raw_denominator
    return {
        'folds': fold_rows,
        'rule': '加权行Brier按全部验证行总权重汇总；同分选较小gamma',
        'total_validation_weight': _score(raw_denominator),
        'raw_brier': _score(raw_total),
        'pooled_gamma_scores': pooled_scores,
        'selected_gamma': _score(selected),
        'selected_gamma_exact': str(selected),
        'selected_brier': _score(selected_score),
        'brier_improvement_raw_minus_selected': _score(raw_total - selected_score),
    }


def _exact_score(p):
    return p['numerator'] / p['denominator']


def _blocks_report(model):
    return [{
        'index': b['index'],
        'p_low': _num(b['p_low']), 'p_high': _num(b['p_high']),
        'knots': [_num(p) for p in b['knots']],
        'weight': _score(b['weight']),
        'positive_weight': _score(b['positive']),
        'calibrated': _score(b['calibrated']),
    } for b in model['blocks']]


def _mapping_report(model):
    return [{'p': _num(p), 'calibrated': _score(model['mapping'][p]),
             'block': model['block_of_knot'][p]} for p in model['knots']]


def _hash(obj):
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf8')).hexdigest()


def _development_hash(state):
    rows = [{
        'id': r['id'], 'process': r['process'], 'issued_time': r['issued_time'],
        'lead': r['lead'], 'p': str(r['p']), 'outcome': r['outcome'],
        'weight': str(r['weight']),
    } for r in state['dev_rows']]
    return _hash(rows)


def _percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (pos - low)


def _interval(values, total_draws):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return {
        'draws': total_draws, 'finite_draws': len(values),
        'nonfinite_draws': total_draws - len(values),
        'confidence': 0.95,
        'low': _percentile(values, 0.025),
        'high': _percentile(values, 0.975),
    }


def bootstrap_external(state, raw_predictions, final_predictions):
    """按完整外期过程重抽；映射冻结，不重选gamma，不重新拟合。"""
    processes = sorted(state['ext'])
    if not processes:
        raise ValueError('没有外期过程')
    rows_by_process = {name: [r for r in state['ext_rows'] if r['process'] == name]
                       for name in processes}
    by_id = {r['id']: (raw, final) for r, raw, final in
             zip(state['ext_rows'], raw_predictions, final_predictions)}
    draws = state['draws']
    if draws == 0:
        return {
            'draws': 0, 'seed': state['seed'], 'unit': 'process', 'mapping_frozen': True,
            'scope': '仅描述冻结映射下的外期过程抽样差异；不含选参、块边界或重拟合不确定性',
            'brier_difference': None, 'ece_difference': None, 'log_loss_difference': None,
        }
    rng = random.Random(state['seed'])
    brier_diffs = []
    ece_diffs = []
    log_diffs = []
    log_nonfinite = 0
    invalid = 0
    for _ in range(draws):
        names = [processes[rng.randrange(len(processes))] for _ in processes]
        sample_rows = []
        raw_preds = []
        final_preds = []
        for name in names:
            for r in rows_by_process[name]:
                sample_rows.append(r)
                raw, final = by_id[r['id']]
                raw_preds.append(raw)
                final_preds.append(final)
        raw_metrics = metrics(sample_rows, raw_preds, state['edges'])
        final_metrics = metrics(sample_rows, final_preds, state['edges'])
        if (raw_metrics['brier'] is None or final_metrics['brier'] is None
                or raw_metrics['calibration'] is None or final_metrics['calibration'] is None):
            invalid += 1
            continue
        brier_diffs.append(final_metrics['brier'] - raw_metrics['brier'])
        ece_diffs.append(final_metrics['calibration']['ece'] - raw_metrics['calibration']['ece'])
        raw_log = raw_metrics['log_loss']
        final_log = final_metrics['log_loss']
        if raw_log['infinite'] or final_log['infinite'] or raw_log['value'] is None or final_log['value'] is None:
            log_nonfinite += 1
        else:
            log_diffs.append(final_log['value'] - raw_log['value'])
    return {
        'draws': draws, 'seed': state['seed'], 'unit': 'process', 'mapping_frozen': True,
        'refit_during_bootstrap': False, 'invalid_draws': invalid,
        'scope': '仅描述冻结映射下的外期过程抽样差异；不含选参、块边界或重拟合不确定性',
        'difference': 'final - raw（负值表示最终概率的指标更低）',
        'brier_difference': _interval(brier_diffs, draws),
        'ece_difference': _interval(ece_diffs, draws),
        'log_loss_difference': {**_interval(log_diffs, draws),
                                'nonfinite_log_draws': log_nonfinite},
    }


def analyze(plan):
    state = _prepare(plan)
    folds = _fold_models(state)
    selection = _fold_selection(state, folds)
    selected = Fraction(selection['selected_gamma_exact'])

    final_model = fit_isotonic(state['dev_rows'])
    final_report = {
        'fit_on': 'development rows only',
        'support': [_num(final_model['support'][0]), _num(final_model['support'][1])],
        'blocks': _blocks_report(final_model),
        'mapping': _mapping_report(final_model),
        'certificate': final_model['certificate'],
        'mapping_sha256': _hash({
            'blocks': _blocks_report(final_model),
            'mapping': _mapping_report(final_model),
            'selected_gamma_exact': str(selected),
            'outside_support': state['outside'],
        }),
        'development_input_sha256': _development_hash(state),
    }
    if state['outside'] == 'clamp':
        hits = [r['id'] for r in state['dev_rows'] + state['ext_rows']
                if final_probability(final_model, r['p'], selected, state['outside'])[1]
                .startswith('clamped')]
        final_report['clamped_rows'] = hits
        final_report['unestimable_rows'] = []
    else:
        missing = [r['id'] for r in state['dev_rows'] + state['ext_rows']
                   if final_probability(final_model, r['p'], selected, state['outside'])[0] is None]
        final_report['clamped_rows'] = []
        final_report['unestimable_rows'] = missing
        final_report['mapping_sha256'] = _hash({
            'blocks': _blocks_report(final_model),
            'mapping': _mapping_report(final_model),
            'selected_gamma_exact': str(selected),
            'outside_support': state['outside'],
            'clamped_rows': [],
            'unestimable_rows': missing,
        })

    dev_raw_preds = [r['p'] for r in state['dev_rows']]
    dev_final_preds = [final_probability(final_model, r['p'], selected, state['outside'])[0]
                       for r in state['dev_rows']]
    dev_raw_metrics = metrics(state['dev_rows'], dev_raw_preds, state['edges'])
    dev_final_metrics = metrics(state['dev_rows'], dev_final_preds, state['edges'])

    outfold_rows = []
    outfold_raw_preds = []
    outfold_final_preds = []
    for fm in folds:
        validation = fm['validation_rows']
        outfold_rows.extend(validation)
        outfold_raw_preds.extend([r['p'] for r in validation])
        outfold_final_preds.extend(
            [final_probability(fm['model'], r['p'], selected, state['outside'])[0] for r in validation])
    outfold_raw_metrics = metrics(outfold_rows, outfold_raw_preds, state['edges'])
    outfold_final_metrics = metrics(outfold_rows, outfold_final_preds, state['edges'])

    ext_raw_preds = [r['p'] for r in state['ext_rows']]
    ext_final_preds = [final_probability(final_model, r['p'], selected, state['outside'])[0]
                       for r in state['ext_rows']]
    external_raw_metrics = metrics(state['ext_rows'], ext_raw_preds, state['edges'])
    external_final_metrics = metrics(state['ext_rows'], ext_final_preds, state['edges'])
    bootstrap = bootstrap_external(state, ext_raw_preds, ext_final_preds)

    ext_missing = [r['id'] for r, q in zip(state['ext_rows'], ext_final_preds) if q is None]
    in_sample_delta = _delta_report(dev_final_metrics, dev_raw_metrics)
    outfold_delta = _delta_report(outfold_final_metrics, outfold_raw_metrics)
    external_delta = _delta_report(external_final_metrics, external_raw_metrics)

    report = {
        'config': {
            'outside_support': state['outside'],
            'shrink_candidates': [_score(g) for g in state['gammas']],
            'shrink_candidates_exact': [str(g) for g in state['gammas']],
            'bin_edges': [_num(e) for e in state['edges']],
            'bootstrap_draws': state['draws'],
            'seed': state['seed'],
        },
        'design': {
            'development': {
                'processes': state['dev'],
                'rows': len(state['dev_rows']),
                'weight': _score(sum((r['weight'] for r in state['dev_rows']), Fraction(0))),
                'issued_time_range': [min(r['issued_time'] for r in state['dev_rows']),
                                      max(r['issued_time'] for r in state['dev_rows'])],
            },
            'external': {
                'processes': state['ext'],
                'rows': len(state['ext_rows']),
                'weight': _score(sum((r['weight'] for r in state['ext_rows']), Fraction(0))),
                'issued_time_range': [min(r['issued_time'] for r in state['ext_rows']),
                                      max(r['issued_time'] for r in state['ext_rows'])],
            },
            'folds': [{
                'index': f['index'], 'train': f['train'], 'validation': f['validation'],
                'train_last_issued': f['train_last_issued'],
                'validation_first_issued': f['validation_first_issued'],
            } for f in state['folds']],
            'process_isolation': {
                'development_external_disjoint': True,
                'fold_train_validation_disjoint': True,
                'all_training_strictly_before_validation': True,
            },
        },
        'selection': selection,
        'final_mapping': final_report,
        'external': {
            'outside_support': state['outside'],
            'unestimable_rows': ext_missing,
            'raw': external_raw_metrics,
            'final': external_final_metrics,
            'delta': external_delta,
            'bootstrap': bootstrap,
        },
        'improvement_attribution': {
            'development_in_sample': {
                'label': '开发集内拟合（同一批开发行既拟合又评分）',
                'interpretation': '只说明拟合得更好，属于乐观的表观改善，不能当作留出证据',
                'raw': dev_raw_metrics,
                'final': dev_final_metrics,
                'delta': in_sample_delta,
            },
            'development_out_of_fold': {
                'label': '开发内时间前推折（每折训练过程严格早于验证过程）',
                'interpretation': '验证折用于选gamma和估计折外改善；不是外期，也不能替代外期',
                'raw': outfold_raw_metrics,
                'final': outfold_final_metrics,
                'delta': outfold_delta,
            },
            'external_holdout': {
                'label': '外期留出（过程整组隔离，结局不参与选参、切分、块边界或拟合权重）',
                'interpretation': '冻结的开发映射直接用于外期评分，是真正的留出改善证据',
                'raw': external_raw_metrics,
                'final': external_final_metrics,
                'delta': external_delta,
            },
        },
        'provenance': {
            'gamma_selected_with': 'development folds only',
            'final_mapping_fit_with': 'development rows only',
            'isotonic_block_boundaries_from': 'distinct training p values only',
            'row_order_used_for_fit': False,
            'external_outcomes_used_for_selection': False,
            'external_outcomes_used_for_blocks': False,
            'external_outcomes_used_for_fit_weights': False,
            'external_outcomes_used_for_bin_edges': False,
            'bin_edges_locked_before_scoring': True,
            'empty_bins_retained': True,
            'bootstrap_mapping_frozen': True,
            'bootstrap_scope': '仅描述冻结映射下的外期过程抽样差异；不含选参、块边界或重拟合不确定性',
        },
    }
    return report
