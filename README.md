# Valhalla — A Living Simulation of IIT Ropar

Valhalla is a small, living model of campus life. A handful of AI students —
each with their own personality, hostel, hobbies, and goals — wake up on a
pixel map of IIT Ropar, plan their day, walk around, notice the people near
them, stop to talk, remember what happened, and carry those memories into the
next day. You watch it all unfold on a map in your browser.

This document explains, in plain language, how the whole thing works and how
to run it — no prior knowledge of the code required.

---

## The Big Picture

Think of each student as having two parts, like a person:

- **A brain** that senses the world, remembers, and decides.
- **A body** that actually walks around and does things.

The brain never moves the student directly. It *decides* ("keep studying",
"go to the mess", "say hi to a friend") and then tells the body to carry it
out. This mirrors how people work, and it keeps the thinking cleanly separated
from the doing.

A central clock ("the world") advances by a configurable number of simulated
minutes on each tick. All students act together, then the world checks who is
near whom. The wall-clock length of a simulated day is:

```text
1440 × real-seconds-per-simulated-minute ÷ speed multiplier
```

The code default is 4 seconds per simulated minute, but `.env` and command-line
overrides take precedence. Check the status bar or run the settings command
below for the active duration.

---

## What Happens On Every Tick

1. **The body moves.** Each student advances along their plan — a step down a
   path, finishing an activity, or starting the next one. This is pure
   movement; it costs nothing and never calls the AI.

2. **They perceive.** Each student looks around a configurable circle (50
   pixels by default) and notices nearby students and their activities. This
   spatial check is simple geometry and does not itself call the AI.

3. **They may talk.** If two students' circles overlap and the moment is right
   (they're both awake and free, they haven't just spoken, and it's a clean
   one‑on‑one — not a crowd), they strike up a conversation. The conversation
   is written once by the AI, in a single request, and both students remember
   it afterwards.

4. **They decide when something changes.** A new nearby situation can prompt a
   student to either continue their plan or request a remaining-day replan.
   Energy, emotion, the daily replan limit, and the shared AI budget can all
   suppress that request.

5. **The day is archived and turns over.** Reaching the final planned activity
   starts that student's archive early. At midnight, the running simulation
   lets active conversations finish for a bounded time, performs a final
   archive so late conversations are included, and installs the next-day plans
   without replacing the students or their physical state.

6. **A checkpoint is saved and broadcast.** The world writes recovery state and
   sends the latest state to the browser so the map stays live.

---

## Memory — How Students Remember

Students have two layers of memory, like short‑term and long‑term recall:

- **The day so far** — everything that happened today: the plan, the things
  they did, who they saw, and every conversation.
- **The past** — at the end of each day, that day is summarized and archived.
  Those archives build up over time into a personal history.

When a student plans their day or sits down to talk, they don't start from a
blank slate. They **recall** the most relevant bits of their past — matched by
topic and weighted toward what happened recently — plus a short summary of the
last couple of days. That recall is fed into their planning and their
conversations, so the simulation has genuine continuity: a project discussed
yesterday can carry into today.

There are two ways memory recall can work, and you choose which:

- **Keyword recall (default).** Fast, fully offline, no extra setup. It matches
  memories by shared words and recency. This is what the demo uses.
- **Meaning‑based recall (optional).** A smarter mode that understands the
  *meaning* of a query, so it can surface a related memory even when the exact
  words differ. It uses Qdrant and Gemini embeddings, consumes budget, and is
  off unless selected. If it is unavailable, the system falls back to keyword
  recall without stopping the simulation.

---

## Being Careful With the AI Budget

The AI runs on a small pool of free‑tier keys that are easily exhausted, so the
simulation is designed to spend as little as possible.

- **Movement and spatial checks cost nothing.** AI is used for initial plans,
  conversations, and eligible decisions after a novel observation.
- **Conversation is a single request.** After a chat, the current build resumes
  the existing plan; it does not automatically generate a replacement plan.
- **A budget watchdog** keeps a running count of AI usage. If you set a limit
  and it's reached, the extra thinking is skipped and the simulation keeps
  running on its free path — it never crashes for lack of budget.
- **A slower clock helps too.** Increasing the configured wall-clock duration
  spreads the same requests over more real time, which helps avoid per-minute
  rate limits.

The result: on a quiet stretch, the whole campus makes **zero** AI requests.

---

## What You See In the Browser

- **The campus map** with each student as a coloured dot that glides between
  places. The map gently shifts from night to day and back as the clock moves.
- **Name labels** on every student, and — when you focus one — a live line of
  what they're doing right now.
- **Click any student** (on the map, in the legend, or on their window) to
  focus them: the view smoothly centres on them and they're highlighted.
- **A legend** listing everyone by colour, with a marker when someone's
  chatting.
- **A conversation feed** on the right that fills in with each new chat — who
  spoke, a one‑line summary, and where.
- **Draggable student windows** showing each person's current activity and
  their conversation history.
- **A status bar** with the clock, day, pace, and how many students are active
  or chatting.

---

## How To Run It

### 1. One‑time setup

```powershell
# From the project root
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Then open the `.env` file and add your Google Gemini API keys (`GEMINI_API_KEY_1`,
`GEMINI_API_KEY_2`, …). At least one is required; more keys mean better
resilience against rate limits.

### 2. Run the web experience (recommended for the demo)

```powershell
python backend/server.py
# then open http://127.0.0.1:8000
```

This starts a fresh simulation and the live map together. It removes prior
short-term memory and checkpoints before preparing new day plans.

To continue from the latest checkpoint instead, run:

```powershell
python backend/server.py --resume-checkpoint
```

The resume command never clears runtime data. If no checkpoint exists, it
reports the issue instead of silently starting a new simulation.

### 3. Build the browser interface (first time, or after UI changes)

```powershell
cd frontend
npm install
npm run build
```

For live UI development you can instead run `npm run dev`.

### 4. Run the simulation on its own (no browser)

```powershell
# One full day
$env:PYTHONPATH="backend"; python backend/src/core/world_engine.py --days 1

# Resume from where a previous run left off
$env:PYTHONPATH="backend"; python backend/src/core/world_engine.py --resume
```

### 5. See the current settings at a glance

```powershell
$env:PYTHONPATH="backend"; python backend/src/config.py
```

---

## Settings You Can Change

Every setting has a sensible default. You can change it in the `.env` file, or
override it for a single run with a command‑line flag. A flag always wins over
`.env`, which wins over the built‑in default.

| What it controls | `.env` setting | Command‑line flag | Default |
|---|---|---|---|
| Overall speed multiplier | `SIM_TICK_SPEED` | `--tick-speed` | 1.0 |
| Real seconds per simulated minute | `SIM_REAL_SECONDS_PER_SIM_MINUTE` | `--real-seconds-per-sim-minute` | 4.0 code default; use the formula above for day length |
| How close counts as "near" (pixels) | `SIM_PERCEPTION_RADIUS_PX` | `--perception-radius-px` | 50 |
| Memory recall mode | `SIM_MEMORY_BACKEND` | `--memory-backend` | keyword |
| Max conversations per student per day | `SIM_MAX_CONVERSATIONS_PER_AGENT` | `--max-conversations` | 5 |
| Max mid‑day replans per student | `SIM_MAX_REPLANS_PER_AGENT_PER_DAY` | `--max-replans` | 3 |
| Conversation wait at day handoff | `SIM_DAY_HANDOFF_CONVERSATION_TIMEOUT_SECONDS` | `--day-handoff-conversation-timeout` | 45 seconds |
| AI requests allowed per real hour (0 = no limit) | `SIM_LLM_HOURLY_CEILING` | `--llm-hourly-ceiling` | 0 |

Examples:

```powershell
# Faster day, tighter budget
$env:PYTHONPATH="backend"; python backend/src/core/world_engine.py --tick-speed 4 --llm-hourly-ceiling 30

# Turn on meaning-based memory
$env:PYTHONPATH="backend"; python backend/src/core/world_engine.py --memory-backend vector
```

---

## Where Things Live

| Folder | What's inside |
|---|---|
| `backend/` | The simulation and the web server |
| `backend/data/personalities/` | The student profiles |
| `backend/data/Short_term_db/` | Each student's active-day plan and runtime history |
| `backend/data/Long_term_db/` | The archived history each student builds up |
| `backend/data/checkpoints/` | Save files for crash recovery |
| `backend/data/environment/` | The campus places, entry points, and relationships |
| `frontend/` | The browser map and panels |
| `changes.md` | A detailed log of everything added in this project |

---

## A Note On Cost

The number of AI requests in a simulated day depends on how many students there
are and how often they meet — not on how fast the clock runs. Slowing the clock
doesn't reduce the total; it just spreads it out so you stay under the free‑tier
rate limits. For an accurate estimate of charges, use Google's own pricing
tools for the Gemini API.
