import itertools
import pandas as pd
from pathlib import Path

curdir = Path(__file__).parent.resolve()
df = pd.read_csv(f"{curdir}/a.csv")


cycles = []
soft_cycles = []

for c1, c2 in itertools.combinations(df.columns, 2):
    sel = df[(df[c1] > 0) & (df[c2] > 0)]
    if not sel.empty:
        is_gt = (sel[c1] > sel[c2]).any()
        is_lt = (sel[c1] < sel[c2]).any()
        is_lte = (sel[c1] <= sel[c2]).any()
        if is_gt and is_lt:
            cycles += [(c1, c2)]
        elif is_gt and is_lte:
            soft_cycles += [(c1, c2)]

print(f"cycles[{len(cycles)}]: ", cycles)
print(f"soft cycles[{len(soft_cycles)}]: ", soft_cycles)
