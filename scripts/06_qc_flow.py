"""Systematic Flow-channel health check across UCDDB records."""

from pathlib import Path

import numpy as np
import pyedflib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset-rm" / "dataset-rm" / "ucddb" / "DESTINATION"

print(f"{'record':<12}{'flow_std':>10}{'flow_unique':>13}{'flow_sat':>11}{'sum_std':>10}{'sum_unique':>12}")
suspect = []
for path in sorted(DATA.glob("ucddb*.rec")):
    edf = pyedflib.EdfReader(str(path))
    labels = [edf.getLabel(i) for i in range(edf.signals_in_file)]
    flow = edf.readSignal(labels.index("Flow")).astype(np.float64)
    total = edf.readSignal(labels.index("Sum")).astype(np.float64)
    edf.close()
    fu = len(np.unique(np.round(flow, 3)))
    su = len(np.unique(np.round(total, 3)))
    sat = float(np.mean(np.abs(flow) > 18.0))
    flag = flow.std() < 1.5 or fu < 500
    if flag:
        suspect.append(path.stem)
    print(f"{path.stem:<12}{flow.std():>10.4f}{fu:>13}{sat:>11.4f}{total.std():>10.4f}{su:>12}"
          + ("   <-- SUSPECT" if flag else ""))

print("\nsuspect records:", suspect)
