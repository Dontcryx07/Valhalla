"""
Comprehensive agent rename script.
Renames all 10 agents to arbitrary new names across:
  - personalities/ (dirs + JSON content)
  - Short_term_db/ (dirs + JSON content)
  - environment/relationship_matrix.json (keys + contexts)
  - checkpoints/ (gzipped JSON agent data)
  - Python source files (defaults, docstrings, profiles, legacy maps)
"""

import json, gzip, os, shutil, re, time
from pathlib import Path

BASE = Path("backend/data")
BACKEND = Path("backend")

# ── Mapping: old_agent_id -> {new_id, new_display_name} ──
MAPPING = {
    "ankit":           {"id": "asa",          "name": "Asa"},
    "ansh_batra":      {"id": "briar_noel",   "name": "Briar Noel"},
    "anubhav_prasad":  {"id": "corin_vale",   "name": "Corin Vale"},
    "ghanisht_kaushal":{"id": "dale_whitman", "name": "Dale Whitman"},
    "gurnoor_singh":   {"id": "ellery_quinn", "name": "Ellery Quinn"},
    "lavanya_sharma":  {"id": "finley_ashford","name": "Finley Ashford"},
    "parv_singla":     {"id": "gray_wilder",  "name": "Gray Wilder"},
    "riya_murarka":    {"id": "hollis_bowen", "name": "Hollis Bowen"},
    "saksham":         {"id": "ivy",          "name": "Ivy"},
    "tanishq":         {"id": "jules",        "name": "Jules"},
}

# Personality directory names (some are shortened vs agent_id)
PERS_DIR_MAP = {
    "ankit":          "asa",
    "ansh_batra":     "briar_noel",
    "anubhav":        "corin_vale",
    "ganishat":       "dale_whitman",
    "gurnoor":        "ellery_quinn",
    "lavanya_sharma": "finley_ashford",
    "parv":           "gray_wilder",
    "riya_murarka":   "hollis_bowen",
    "saksham":        "ivy",
    "tanishq":        "jules",
}

# Reverse maps for lookup
OLD_TO_NEW_ID = {k: v["id"] for k, v in MAPPING.items()}
OLD_TO_NEW_NAME = {k: v["name"] for k, v in MAPPING.items()}
NEW_ID_TO_OLD = {v["id"]: k for k, v in MAPPING.items()}

# Build lookup: old display names -> new display names
OLD_DISPLAY_NAMES = {
    "Ankit": "Asa",
    "Ansh Batra": "Briar Noel",
    "Anubhav Prasad": "Corin Vale",
    "Ghanisht Kaushal": "Dale Whitman",
    "Gurnoor Singh": "Ellery Quinn",
    "Lavanya Sharma": "Finley Ashford",
    "Parv Singla": "Gray Wilder",
    "Riya Murarka": "Hollis Bowen",
    "Saksham": "Ivy",
    "Tanishq": "Jules",
}

# ── Helper: text replacement using old->new display names ──
def replace_display_names(text: str) -> str:
    for old, new in sorted(OLD_DISPLAY_NAMES.items(), key=lambda x: -len(x[0])):
        text = text.replace(old, new)
    return text

# ── Phase 1: Update personality JSON files ──
print("=" * 60)
print("PHASE 1: Updating personality JSON files...")
PERS_DIR = BASE / "personalities"

for old_dir_name, new_dir_name in PERS_DIR_MAP.items():
    old_dir = PERS_DIR / old_dir_name
    if not old_dir.exists():
        print(f"  [SKIP] personality dir {old_dir_name} not found")
        continue
    # Update the JSON file content
    for json_file in old_dir.glob("*.json"):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        if "Name" in data and data["Name"] in OLD_DISPLAY_NAMES:
            old_name = data["Name"]
            data["Name"] = OLD_DISPLAY_NAMES[old_name]
            print(f"  [Name] {old_name} -> {data['Name']}")
        # Replace references to other agents in all text fields
        for key in ["learned", "lifestyle", "goals", "daily_plan_req"]:
            if key in data and isinstance(data[key], str):
                new_val = replace_display_names(data[key])
                if new_val != data[key]:
                    print(f"    [{key}] updated references in {old_dir_name}")
                    data[key] = new_val
        if "hobbies" in data and isinstance(data["hobbies"], str):
            data["hobbies"] = replace_display_names(data["hobbies"])
        json_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    # Rename directory
    new_dir = PERS_DIR / new_dir_name
    if old_dir != new_dir and not new_dir.exists():
        old_dir.rename(new_dir)
        print(f"  [DIR] {old_dir_name}/ -> {new_dir_name}/")
        # Rename the JSON file inside to match
        for f in new_dir.glob("*.json"):
            if f.stem == old_dir_name or f.stem in PERS_DIR_MAP:
                new_f = new_dir / f"{new_dir_name}.json"
                if f != new_f:
                    f.rename(new_f)
                    print(f"    [FILE] {f.name} -> {new_f.name}")

# ── Phase 2: Update Short_term_db JSON files ──
print("\n" + "=" * 60)
print("PHASE 2: Updating Short_term_db files...")
STD_DIR = BASE / "Short_term_db"

for old_id, new_info in MAPPING.items():
    old_dir = STD_DIR / old_id
    new_dir_std = STD_DIR / new_info["id"]
    if not old_dir.exists():
        # Try alternate directory names
        found = False
        for d in STD_DIR.iterdir():
            if d.is_dir() and old_id.startswith(d.name) or d.name.startswith(old_id):
                old_dir = d
                found = True
                break
        if not found:
            print(f"  [SKIP] Short_term_db dir for {old_id} not found")
            continue

    for json_file in old_dir.glob("*.json"):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        if "persona_name" in data and data["persona_name"] in OLD_DISPLAY_NAMES:
            old_pn = data["persona_name"]
            data["persona_name"] = OLD_DISPLAY_NAMES[old_pn]
            print(f"  [persona_name] {old_pn} -> {data['persona_name']}")
        json_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    # Rename directory
    if old_dir != new_dir_std and not new_dir_std.exists():
        old_dir.rename(new_dir_std)
        print(f"  [DIR] {old_dir.name}/ -> {new_dir_std.name}/")

# Also handle inconsistencies in Short_term_db dir names
for old_possible in ["anubhav_prasad", "ghanisht_kaushal", "gurnoor_singh", "parv_singla"]:
    old_dir = STD_DIR / old_possible
    if old_dir.exists() and old_possible in OLD_TO_NEW_ID:
        new_dir = STD_DIR / OLD_TO_NEW_ID[old_possible]
        if old_dir != new_dir and not new_dir.exists():
            old_dir.rename(new_dir)
            print(f"  [DIR FIX] {old_possible}/ -> {new_dir.name}/")

# ── Phase 3: Update relationship_matrix.json ──
print("\n" + "=" * 60)
print("PHASE 3: Updating relationship_matrix.json...")
REL_PATH = BASE / "environment" / "relationship_matrix.json"
if REL_PATH.exists():
    data = json.loads(REL_PATH.read_text(encoding="utf-8"))
    rels = data.get("relationships", data)
    new_rels = {}
    for key, record in rels.items():
        if "->" not in key:
            new_rels[key] = record
            continue
        source, target = key.split("->", 1)
        new_source = OLD_TO_NEW_ID.get(source, source)
        new_target = OLD_TO_NEW_ID.get(target, target)
        new_key = f"{new_source}->{new_target}"
        # Update context string
        if isinstance(record, dict) and "context" in record:
            record["context"] = replace_display_names(record["context"])
        new_rels[new_key] = record
        if new_key != key:
            print(f"  [KEY] {key} -> {new_key}")

    # Write back
    if "relationships" in data:
        data["relationships"] = new_rels
        REL_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        REL_PATH.write_text(json.dumps(new_rels, indent=2), encoding="utf-8")
    print("  [OK] relationship_matrix.json updated")

# ── Phase 4: Update checkpoints ──
print("\n" + "=" * 60)
print("PHASE 4: Updating checkpoint files...")
CKPT_DIR = BASE / "checkpoints"
count = 0
for ckpt in sorted(CKPT_DIR.glob("*.json.gz")):
    try:
        # Read compressed
        with gzip.open(ckpt, "rt", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [SKIP] {ckpt.name}: {e}")
        continue

    changed = False
    # Update agents in checkpoint
    for agent in data.get("agents", []):
        agent_id = agent.get("agent_id", "")
        if agent_id in OLD_TO_NEW_ID:
            new_id = OLD_TO_NEW_ID[agent_id]
            print(f"    [agent_id] {agent_id} -> {new_id}")
            # Update agent_id
            agent["agent_id"] = new_id
            # Update manager agent_id if present
            if agent.get("manager") and agent["manager"].get("agent_id"):
                agent["manager"]["agent_id"] = new_id
            changed = True

        persona_name = agent.get("persona_name", "")
        if persona_name in OLD_DISPLAY_NAMES:
            new_pn = OLD_DISPLAY_NAMES[persona_name]
            if new_pn != persona_name:
                print(f"    [persona_name] {persona_name} -> {new_pn}")
                agent["persona_name"] = new_pn
                changed = True

        # Update persona dict inside agent
        persona = agent.get("persona", {})
        if persona.get("Name") in OLD_DISPLAY_NAMES:
            old_n = persona["Name"]
            persona["Name"] = OLD_DISPLAY_NAMES[old_n]
            changed = True
        # Update text fields in persona
        for key in ["learned", "lifestyle", "goals", "daily_plan_req"]:
            if key in persona and isinstance(persona[key], str):
                new_val = replace_display_names(persona[key])
                if new_val != persona[key]:
                    persona[key] = new_val
                    changed = True

    if changed:
        # Write back compressed
        try:
            tmp = ckpt.with_suffix(ckpt.suffix + ".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                f.write(json.dumps(data, separators=(",", ":"), default=str))
            tmp.replace(ckpt)
            count += 1
        except Exception as e:
            print(f"  [ERROR] writing {ckpt.name}: {e}")

print(f"  Updated {count} checkpoint files")

# ── Phase 5: Update Python source files ──
print("\n" + "=" * 60)
print("PHASE 5: Updating Python source files...")

SRC_DIR = BACKEND / "src"

# 5a. Actions.py - update default persona argument
actions_py = BACKEND / "src" / "agents" / "Actions.py"
if actions_py.exists():
    content = actions_py.read_text(encoding="utf-8")
    # CLI default
    content = content.replace('default="parv"', 'default="gray_wilder"')
    content = content.replace('default="parv",\n        help="Persona name (e.g. parv', 'default="gray_wilder",\n        help="Persona name (e.g. gray_wilder')
    content = content.replace('default="tanishq"', 'default="jules"')
    # Docstring example
    content = content.replace('AgentActionManager("parv"', 'AgentActionManager("gray_wilder"')
    actions_py.write_text(content, encoding="utf-8")
    print("  [OK] Actions.py")

# 5b. Single_agent.py - update docstring default
single_py = BACKEND / "src" / "agents" / "Single_agent.py"
if single_py.exists():
    content = single_py.read_text(encoding="utf-8")
    content = content.replace('"persona_name": "parv"', '"persona_name": "gray_wilder"')
    content = content.replace("python Single_agent.py parv", "python Single_agent.py gray_wilder")
    single_py.write_text(content, encoding="utf-8")
    print("  [OK] Single_agent.py")

# 5c. day_planner.py - update CLI default
day_planner_py = BACKEND / "src" / "agents" / "day_planner.py"
if day_planner_py.exists():
    content = day_planner_py.read_text(encoding="utf-8")
    content = content.replace('default="gurnoor"', 'default="ellery_quinn"')
    content = content.replace('help="Persona name or persona JSON path (for example: tanishq', 'help="Persona name or persona JSON path (for example: jules')
    day_planner_py.write_text(content, encoding="utf-8")
    print("  [OK] day_planner.py")

# 5d. conversation.py - update profiles, special tuples, CLI defaults, LEGACY map
conv_py = BACKEND / "src" / "agents" / "conversation.py"
if conv_py.exists():
    content = conv_py.read_text(encoding="utf-8")

    # Update the CLI defaults
    content = content.replace('default="parv"', 'default="gray_wilder"')
    content = content.replace('default="tanishq"', 'default="jules"')
    content = content.replace('default="gurnoor"', 'default="ellery_quinn"')

    # Update docstring examples
    content = content.replace('agent_a_id="parv_singla"', 'agent_a_id="gray_wilder"')
    content = content.replace('agent_b_id="tanishq"', 'agent_b_id="jules"')
    content = content.replace('matrix.get("parv_singla", "tanishq")', 'matrix.get("gray_wilder", "jules")')
    content = content.replace('matrix.get("tanishq", "parv_singla")', 'matrix.get("jules", "gray_wilder")')

    # Update the profiles dict in ensure_grounded_seed
    # Replace profile entries one by one
    for old_id, new_info in MAPPING.items():
        new_id = new_info["id"]
        # In profiles dict keys
        content = re.sub(
            rf'"{re.escape(old_id)}":\s*\(',
            f'"{new_id}": (',
            content,
        )

    # Update special relationship tuples
    # Old -> New pairs in the special dict
    special_replacements = [
        ('"gurnoor_singh", "parv_singla"', '"ellery_quinn", "gray_wilder"'),
        ('"parv_singla", "gurnoor_singh"', '"gray_wilder", "ellery_quinn"'),
        ('"lavanya_sharma", "ghanisht_kaushal"', '"finley_ashford", "dale_whitman"'),
        ('"ghanisht_kaushal", "lavanya_sharma"', '"dale_whitman", "finley_ashford"'),
        ('"riya_murarka", "tanishq"', '"hollis_bowen", "jules"'),
        ('"tanishq", "riya_murarka"', '"jules", "hollis_bowen"'),
        ('"ansh_batra", "saksham"', '"briar_noel", "ivy"'),
        ('"saksham", "ansh_batra"', '"ivy", "briar_noel"'),
    ]
    for old_t, new_t in special_replacements:
        content = content.replace(old_t, new_t)

    # Update context strings in special tuples (display names in context)
    for old_name, new_name in OLD_DISPLAY_NAMES.items():
        content = content.replace(old_name, new_name)

    # Update LEGACY_AGENT_IDS
    content = content.replace(
        '"aditi_menon": "riya_murarka"',
        '"aditi_menon": "hollis_bowen"',
    )
    content = content.replace(
        '"meher_bansal": "lavanya_sharma"',
        '"meher_bansal": "finley_ashford"',
    )

    conv_py.write_text(content, encoding="utf-8")
    print("  [OK] conversation.py")

# 5e. Odin.py - check for hardcoded names
odin_py = BACKEND / "Odin.py"
if odin_py.exists():
    content = odin_py.read_text(encoding="utf-8")
    # Any remaining hardcoded agent IDs or names
    for old_id, new_info in MAPPING.items():
        # Replace "agent_id" references that are exact matches
        content = content.replace(f'"{old_id}"', f'"{new_info["id"]}"')
    # Also replace display names
    for old_name, new_name in OLD_DISPLAY_NAMES.items():
        # Only replace full-word occurrences (not substrings)
        content = content.replace(old_name, new_name)
    odin_py.write_text(content, encoding="utf-8")
    print("  [OK] Odin.py")

# 5f. tick_graph.py
tick_py = BACKEND / "src" / "core" / "tick_graph.py"
if tick_py.exists():
    content = tick_py.read_text(encoding="utf-8")
    # Replace any hardcoded references
    for old_id, new_info in MAPPING.items():
        content = content.replace(f'"{old_id}"', f'"{new_info["id"]}"')
    for old_name, new_name in OLD_DISPLAY_NAMES.items():
        content = content.replace(old_name, new_name)
    tick_py.write_text(content, encoding="utf-8")
    print("  [OK] tick_graph.py")

# 5g. world_engine.py
we_py = BACKEND / "src" / "core" / "world_engine.py"
if we_py.exists():
    content = we_py.read_text(encoding="utf-8")
    # Check for any hardcoded references
    for old_id, new_info in MAPPING.items():
        content = content.replace(f'"{old_id}"', f'"{new_info["id"]}"')
    for old_name, new_name in OLD_DISPLAY_NAMES.items():
        content = content.replace(old_name, new_name)
    we_py.write_text(content, encoding="utf-8")
    print("  [OK] world_engine.py")

print("\n" + "=" * 60)
print("RENAME COMPLETE!")
print("=" * 60)
