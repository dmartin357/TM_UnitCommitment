import json, uuid

with open('PG_Lib_FERC_Instance/Ferc_EDA_Interactive.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# ── New dropdown widget cell (inserted at index 2) ────────────────────────────
widget_cell = {
    "cell_type": "code",
    "id": uuid.uuid4().hex[:12],
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "import re\n",
        "import ipywidgets as widgets\n",
        "from IPython.display import display\n",
        "\n",
        "# \u2500\u2500 Collection directory \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
        "# Update INSTANCE_DIR here if your pglib-uc clone is in a different location.\n",
        "INSTANCE_DIR = r'C:\\gitrepos\\power-grid-lib\\pglib-uc\\ferc'\n",
        "\n",
        "# Auto-discover available dates from the ferc/ directory (no separate file needed)\n",
        "_available_dates = sorted(set(\n",
        "    re.sub(r'_(hw|lw)\\.json$', '', f)\n",
        "    for f in os.listdir(INSTANCE_DIR) if f.endswith('.json')\n",
        "))\n",
        "\n",
        "# \u2500\u2500 Dropdown widget \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
        "instance_dropdown = widgets.Dropdown(\n",
        "    options=_available_dates,\n",
        "    value=_available_dates[0],\n",
        "    description='Date:',\n",
        "    style={'description_width': 'initial'},\n",
        "    layout=widgets.Layout(width='320px'),\n",
        ")\n",
        "\n",
        "def _on_instance_change(change):\n",
        "    global INSTANCE_DATE\n",
        "    INSTANCE_DATE = change['new']\n",
        "\n",
        "INSTANCE_DATE = instance_dropdown.value\n",
        "instance_dropdown.observe(_on_instance_change, names='value')\n",
        "\n",
        "display(widgets.VBox([\n",
        "    widgets.Label('Select a FERC instance date, then re-run all cells below:'),\n",
        "    instance_dropdown,\n",
        "]))",
    ],
}

# ── Simplify the existing paths cell (now index 3 after insertion) ─────────────
# Remove the date-listing logic — the dropdown owns that now.
# Keep only path derivation and confirmation print.
paths_cell_source = [
    "# \u2500\u2500 Derive file paths from the selected instance \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "# INSTANCE_DIR and INSTANCE_DATE are set by the dropdown cell above.\n",
    "# Re-run this cell and all cells below after changing the dropdown selection.\n",
    "\n",
    "JSON_PATH_HW = os.path.join(INSTANCE_DIR, f'{INSTANCE_DATE}_hw.json')\n",
    "JSON_PATH_LW = os.path.join(INSTANCE_DIR, f'{INSTANCE_DATE}_lw.json')\n",
    "\n",
    "THERMAL_PATH      = f'data/{INSTANCE_DATE}/thermal_generators.json'\n",
    "DEMAND_PATH       = f'data/{INSTANCE_DATE}/demand.json'\n",
    "RENEWABLE_HW_PATH = f'data/{INSTANCE_DATE}/renewable_generators_hw.json'\n",
    "RENEWABLE_LW_PATH = f'data/{INSTANCE_DATE}/renewable_generators_lw.json'\n",
    "\n",
    "print(f'Instance : {INSTANCE_DATE}')\n",
    "print(f'HW file  : {JSON_PATH_HW}')\n",
    "print(f'LW file  : {JSON_PATH_LW}')",
]

# Insert widget cell before the old paths cell (currently at index 2)
nb['cells'].insert(2, widget_cell)
# Old paths cell is now at index 3 — replace its source
nb['cells'][3]['source'] = paths_cell_source

with open('PG_Lib_FERC_Instance/Ferc_EDA_Interactive.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done. Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    src_preview = ''.join(cell['source'])[:60].replace('\n', ' ')
    print(f"  [{i}] {cell['cell_type']}  {src_preview}")
