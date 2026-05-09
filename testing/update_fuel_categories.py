"""
One-shot script: add fuel_category field to thermal_generators.json from gen.csv.

Reads PG_Lib_RTS_GMLC_Instance/data/gen.csv (Category column, index 5) and
writes a fuel_category field into each thermal generator.  Generators not found
in gen.csv are skipped with a warning.

Run once from repo root:
    python testing/update_fuel_categories.py
"""

import csv
import json
from pathlib import Path

GEN_CSV   = Path(__file__).parent.parent / "PG_Lib_RTS_GMLC_Instance" / "data" / "gen.csv"
JSON_PATH = Path(__file__).parent.parent / "PG_Lib_RTS_GMLC_Instance" / "data" / "2020-01-27" / "thermal_generators.json"

FIELD_ORDER = [
    "must_run",
    "fuel_category",
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
    ordered = {k: gen[k] for k in FIELD_ORDER if k in gen}
    for k in gen:
        if k not in ordered:
            ordered[k] = gen[k]
    return ordered


# ── Load gen.csv: GEN UID → Category (column index 5) ────────────────────────
csv_map: dict[str, str] = {}
with open(GEN_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        uid      = row["GEN UID"].strip()
        category = row["Category"].strip()
        csv_map[uid] = category

# ── Load and update JSON ──────────────────────────────────────────────────────
with open(JSON_PATH, encoding="utf-8") as f:
    data = json.load(f)

updated = 0
skipped = 0

for name, gen in data.items():
    if name in csv_map:
        gen["fuel_category"] = csv_map[name]
        data[name] = reorder(gen)
        updated += 1
    else:
        print(f"  WARNING: {name} not found in gen.csv — fuel_category not set")
        skipped += 1

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Done. Updated: {updated}  Skipped: {skipped}")

# ── Sanity check ──────────────────────────────────────────────────────────────
with open(JSON_PATH, encoding="utf-8") as f:
    check = json.load(f)

categories = {}
for name, gen in check.items():
    cat = gen.get("fuel_category", "(missing)")
    categories.setdefault(cat, []).append(name)

print("\nfuel_category distribution:")
for cat, names in sorted(categories.items()):
    print(f"  {cat:15s}: {len(names):3d}  ({', '.join(names[:3])}{'...' if len(names) > 3 else ''})")
