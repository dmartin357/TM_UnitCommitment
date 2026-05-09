"""
One-shot script: transform thermal_generators.json to add all 8 reserve fields.

For every generator that does NOT already have flex_up_limit:
  - old ramp_up_limit   → flex_up_limit   (20-min rate)
  - old ramp_down_limit → flex_down_limit
  - new ramp_up_limit   = flex * 3        (hourly rate)
  - new ramp_down_limit = flex * 3
  - spin_up_limit       = flex / 2
  - spin_down_limit     = flex / 2
  - reg_up_limit        = flex / 4
  - reg_down_limit      = flex / 4
  - ramp_startup_limit / ramp_shutdown_limit: unchanged

For 315_CT_6 (already transformed): leave as-is, just verify field ordering.
"""

import json
from pathlib import Path

PATH = Path(r"C:\Users\warha\OneDrive\Documents\Dev\TM_UnitCommitment\PG_Lib_RTS_GMLC_Instance\data\2020-01-27\thermal_generators.json")

FIELD_ORDER = [
    "must_run",
    "power_output_minimum",
    "power_output_maximum",
    "ramp_up_limit",
    "ramp_down_limit",
    "flex_up_limit",
    "flex_down_limit",
    "spin_up_limit",
    "spin_down_limit",
    "reg_up_limit",
    "reg_down_limit",
    "ramp_startup_limit",
    "ramp_shutdown_limit",
    "time_up_minimum",
    "time_down_minimum",
    "power_output_t0",
    "unit_on_t0",
    "time_down_t0",
    "time_up_t0",
    "startup",
    "piecewise_production",
    "name",
]


def reorder(gen: dict) -> dict:
    ordered = {}
    for k in FIELD_ORDER:
        if k in gen:
            ordered[k] = gen[k]
    for k in gen:
        if k not in ordered:
            ordered[k] = gen[k]
    return ordered


with open(PATH, encoding="utf-8") as f:
    data = json.load(f)

changed = 0
skipped = 0

for name, gen in data.items():
    if "flex_up_limit" in gen:
        # Already transformed (315_CT_6) — just reorder
        data[name] = reorder(gen)
        skipped += 1
        continue

    flex_up = float(gen["ramp_up_limit"])
    flex_dn = float(gen["ramp_down_limit"])

    gen["ramp_up_limit"]   = round(flex_up * 3, 6)
    gen["ramp_down_limit"] = round(flex_dn * 3, 6)
    gen["flex_up_limit"]   = flex_up
    gen["flex_down_limit"] = flex_dn
    gen["spin_up_limit"]   = round(flex_up / 2, 6)
    gen["spin_down_limit"] = round(flex_dn / 2, 6)
    gen["reg_up_limit"]    = round(flex_up / 4, 6)
    gen["reg_down_limit"]  = round(flex_dn / 4, 6)

    data[name] = reorder(gen)
    changed += 1

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Done. Transformed: {changed}  Already done: {skipped}  Total: {changed + skipped}")

# Sanity check
with open(PATH, encoding="utf-8") as f:
    check = json.load(f)

errors = []
for name, gen in check.items():
    if "flex_up_limit" not in gen:
        errors.append(f"  MISSING flex_up_limit: {name}")
    else:
        expected_ramp = round(gen["flex_up_limit"] * 3, 4)
        actual_ramp   = round(gen["ramp_up_limit"], 4)
        if abs(expected_ramp - actual_ramp) > 0.01:
            errors.append(f"  RAMP MISMATCH {name}: ramp={actual_ramp}  flex*3={expected_ramp}")

if errors:
    print("ERRORS:")
    for e in errors:
        print(e)
else:
    print("Sanity check passed: all generators have correct reserve fields.")
