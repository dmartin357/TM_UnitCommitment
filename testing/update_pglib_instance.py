"""
Apply our thermal generator enrichments to the live power-grid-lib instance JSON.

For each thermal generator in the instance:
  - Adds fuel_category from gen.csv (Category column)
  - Adds flex/spin/reg reserve limit fields derived from the original ramp limits
  - Promotes original ramp_up/down_limit to hourly rate (flex * 3)

Safe to re-run: generators already enriched (have flex_up_limit) are skipped for
the reserve transform; fuel_category is always updated from gen.csv.

Usage (from repo root):
    python testing/update_pglib_instance.py
"""

import csv
import json
from pathlib import Path

PGLIB_JSON = Path("C:/gitrepos/power-grid-lib/pglib-uc/rts_gmlc/2020-01-27.json")
GEN_CSV    = Path(__file__).parent.parent / "PG_Lib_RTS_GMLC_Instance" / "data" / "gen.csv"


def main() -> None:
    # ── Load fuel categories from gen.csv ─────────────────────────────────────
    csv_map: dict[str, str] = {}
    with open(GEN_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            csv_map[row["GEN UID"].strip()] = row["Category"].strip()
    print(f"Loaded {len(csv_map)} entries from gen.csv")

    # ── Load full instance JSON ───────────────────────────────────────────────
    with open(PGLIB_JSON, encoding="utf-8") as f:
        data = json.load(f)

    thermal_gens: dict = data["thermal_generators"]
    print(f"Instance: {len(thermal_gens)} thermal generators, "
          f"{data.get('time_periods')} periods")

    fuel_updated    = 0
    reserve_updated = 0
    no_csv_entry    = []

    for name, gen in thermal_gens.items():
        # Fuel category
        if name in csv_map:
            gen["fuel_category"] = csv_map[name]
            fuel_updated += 1
        else:
            no_csv_entry.append(name)

        # Reserve fields (skip if already enriched)
        if "flex_up_limit" not in gen:
            flex_up = float(gen.get("ramp_up_limit", 0.0))
            flex_dn = float(gen.get("ramp_down_limit", 0.0))
            gen["ramp_up_limit"]   = round(flex_up * 3, 6)
            gen["ramp_down_limit"] = round(flex_dn * 3, 6)
            gen["flex_up_limit"]   = flex_up
            gen["flex_down_limit"] = flex_dn
            gen["spin_up_limit"]   = round(flex_up / 2, 6)
            gen["spin_down_limit"] = round(flex_dn / 2, 6)
            gen["reg_up_limit"]    = round(flex_up / 4, 6)
            gen["reg_down_limit"]  = round(flex_dn / 4, 6)
            reserve_updated += 1

    if no_csv_entry:
        print(f"WARNING: {len(no_csv_entry)} generators not found in gen.csv "
              f"(fuel_category not set): {no_csv_entry}")

    data["thermal_generators"] = thermal_gens

    with open(PGLIB_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\nDone.")
    print(f"  fuel_category added/updated : {fuel_updated}")
    print(f"  reserve fields added        : {reserve_updated}")
    print(f"  already had reserve fields  : {len(thermal_gens) - reserve_updated}")

    # ── Sanity check ──────────────────────────────────────────────────────────
    with open(PGLIB_JSON, encoding="utf-8") as f:
        check = json.load(f)

    errors = []
    for name, gen in check["thermal_generators"].items():
        if "flex_up_limit" not in gen:
            errors.append(f"  MISSING flex_up_limit: {name}")
        if "fuel_category" not in gen:
            errors.append(f"  MISSING fuel_category: {name}")

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(e)
    else:
        print("Sanity check passed: all generators have reserve fields and fuel_category.")

    # Show distribution
    cats: dict[str, int] = {}
    for gen in check["thermal_generators"].values():
        cat = gen.get("fuel_category", "(missing)")
        cats[cat] = cats.get(cat, 0) + 1
    print("\nfuel_category distribution:")
    for cat, cnt in sorted(cats.items()):
        print(f"  {cat:15s}: {cnt}")


if __name__ == "__main__":
    main()
