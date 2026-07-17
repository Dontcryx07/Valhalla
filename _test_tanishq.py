import time, sys, json
sys.path.insert(0, 'backend')
from src.agents.day_planner import run
from types import SimpleNamespace

t0 = time.perf_counter()
persona = json.loads(open('backend/data/personalities/tanishq/tanishq.json', encoding='utf-8').read())
proxy = SimpleNamespace(persona=persona, relevant_memories=[], yesterday_summary=None)
result = run(proxy, {
    'current_time': '2026-07-03 00:00',
    'places': None,
    'persona_name': 'tanishq',
    'mode': 'full_day',
    'current_location_id': persona.get('Hostel', ''),
})
elapsed = time.perf_counter() - t0
plan = result.get('day_plan', [])
err = result.get('error')
print(f'Time: {elapsed:.1f}s, Actions: {len(plan)}, Error: {err}')
for a in plan[:8]:
    print(f'  {a["start"]}-{a["end"]} {a["action"]} @ {a.get("location_id","?")}')
