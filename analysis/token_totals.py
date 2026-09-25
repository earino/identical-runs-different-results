"""Study 2's token totals (Appendix D): input, output, reasoning, generated, and the input-to-generated ratio."""
import csv
reas = {(r['harness'], r['model'], int(r['seed'])): int(r['reasoning_tokens'])
        for r in csv.DictReader(open('data/study2/reasoning_tokens.csv'))}
tin = tout = tre = 0
for r in csv.DictReader(open('data/study2/cells.csv')):
    if r['counted'] != 'True' or not r.get('tokens_in'):
        continue
    tin += int(float(r['tokens_in'])); tout += int(float(r['tokens_out'] or 0))
    tre += reas.get((r['harness'], r['model'], int(r['seed'])), 0)
print(f'input {tin/1e9:.2f}B  output {tout/1e6:.1f}M  reasoning {tre/1e6:.1f}M  generated {(tout+tre)/1e6:.1f}M  ratio {tin/(tout+tre):.0f} to 1')
