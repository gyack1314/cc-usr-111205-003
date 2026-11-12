"""检查概率和原始结局，清点开发外期过程身份。"""
from .io import unique,number,positive

def summarize(v):
    unique(v['rows'],'id');counts={};positives=0
    for r in v['rows']:
        if not 0<=number(r['p'])<=1 or type(r['outcome']) is not int or r['outcome'] not in (0,1):raise ValueError('概率或结局无效')
        positive(r['weight']);counts[r['process']]=counts.get(r['process'],0)+1;positives+=r['outcome']
    if set(v['development_processes'])&set(v['external_processes']):raise ValueError('开发外期过程交叠')
    return {'rows':len(v['rows']),'positive_outcomes':positives,'process_rows':dict(sorted(counts.items()))}
