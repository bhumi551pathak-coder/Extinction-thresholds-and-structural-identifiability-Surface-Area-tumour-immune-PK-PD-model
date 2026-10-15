"""Run every additional verification introduced in the October 2026 manuscript revision."""
from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
print('\n=== new_revision_numerics.py ===')
subprocess.check_call([sys.executable,str(ROOT/'scripts'/'new_revision_numerics.py')],cwd=ROOT)
print('\n=== treated_identifiability_exact.py ===')
out=subprocess.check_output([sys.executable,str(ROOT/'scripts'/'treated_identifiability_exact.py')],cwd=ROOT,text=True)
(ROOT/'results'/'treated_identifiability_rank_results.txt').write_text(out,encoding='utf-8')
print(out)
print('All supplementary update outputs regenerated successfully.')
