# Career Re-entry Learning Plan
## Applied AI & Optimisation — Energy & Industrial Systems

**Last updated:** 25th June 2026  
**Status:** Week 2

## Next Session
- Complete Kaggle Time Series course
- Produce time series forecast for electrictity demand and compare against naive forecast

---

## Context & Goals

**Background:** PhD in Applied Mathematics. Previously led a large modelling team in global electricity markets. Career break; rebuilding technical skills after ~8 years of minimal coding (was only ever light scripting at entry level). Mathematically strong, Python/software-engineering rusty.

**Target roles:** Technical ecosystems — industrial AI, applied AI, energy/grid intelligence, optimisation, simulation, decision intelligence, quantum-adjacent. Goal is to build or lead technical systems where modelling, AI, or optimisation are core. Not interested in generic AI adoption or prompt-engineering roles.

**Constraints:** A few hours per week. Not full-time retraining.

**Coaching principle:** Python fluency and AI/ML are one integrated track, not two separate ones. Learn Python by doing AI/ML work on real domain problems — not as a separate prerequisite phase.

---

## Completed

- [x] Kaggle Python intro course
- [x] Kaggle Pandas micro-course
- [x] Google Prompting Essentials course *(sufficient — no further prompt engineering courses needed)*
- [x] Toolchain setup (uv, VS Code, Jupyter, Git)
- [x] Project Structure setup (energy-forecast repo)
In 01_exploration.ipynb
  - [x] Loaded and indexed NESO historic demand data (2024)
  - [x] Built datetime index from SETTLEMENT_DATE and SETTLEMENT_PERIOD
  - [x] Exploratory plots: full year, single week, hourly average, monthly average, day-of-week average
  - [x] Engineered time features: hour, day_of_week, month, is_weekend
  - [x] Confirmed seasonal, weekly and daily demand patterns visually
In 02_forecasting.ipynb
  - [x] Produced naive forecast and generated error metrices
---

## Toolchain Setup (Week 1, do once)

| Tool | Purpose |
|------|---------|
| `uv` | Modern Python package and environment manager |
| VS Code or Cursor | Primary IDE (Cursor has strong AI pair-programming integration) |
| Jupyter | Exploratory analysis and notebook work |
| `ruff` | Linting and code style (configure to run on save in VS Code) |
| `pytest` | Testing |
| Git + GitHub | Source control — set up in Week 1, used from day one |

**Git setup steps:**
1. Install Git, create GitHub account
2. Create a `energy-forecast` repo on GitHub
3. Clone locally and set up project structure (see below)
4. Learn four commands to start: `git add`, `git commit`, `git push`, `git status`

**Use AI tools as a coding accelerator from day one.** Claude or ChatGPT as a pair-programmer: explain unfamiliar patterns, suggest idiomatic Python, help debug. Not to write code for you — to compress re-entry time.

---

## Project Structure (set up in Week 1, grow into it)

```
energy-forecast/
├── data/               # raw data — gitignored
├── notebooks/          # exploratory Jupyter work
├── src/                # reusable code as it emerges
├── tests/              # pytest tests
├── .gitignore          # always include data/, __pycache__, .env
├── README.md           # what the project does and why
├── LEARNING_PLAN.md    # this file
└── pyproject.toml      # managed by uv — single source of truth for dependencies
```

**Key habits:**
- Never commit raw data — add `data/` to `.gitignore`, reference data sources in README instead
- Notebooks for exploration; move reusable code into `src/` when you find yourself copy-pasting
- Commit regularly with meaningful messages — e.g. `add demand data loader with missing-value handling` not `fix stuff`
- Working on `main` is fine for solo work — introduce feature branches in Weeks 7–8

---

## Weeks 1–2 — Environment & Fluency Warm-up

**Goal:** Toolchain working, Git habits established, Python muscle memory returning.

### Courses
- **Kaggle Pandas micro-course** — a few hours, directly practical, foundation for everything else

### Source control this week
- Set up repo and project structure as above
- Every piece of practice code goes in the repo, committed as you go
- Don't aim for perfect commits — aim for the habit

### AI angle
Nothing formal this week. Focus is environment and fluency. Use AI tools actively as you work.

---

## Weeks 3–4 — Pandas/NumPy, Time Series, Energy Data

**Goal:** Comfortable with real messy data; first end-to-end analysis on UK electricity data; classical forecasting baseline with uncertainty.

### Courses
- **Kaggle Time Series micro-course** — introduces lag features, rolling statistics, seasonality decomposition; directly relevant to the project below

### Project: UK Electricity Demand Forecast (classical baseline)
**Data source:** Elexon API or National Grid ESO open data (both free, well-documented)

**Steps:**
1. Pull demand and generation data via API
2. Clean — handle missing values, outliers, timezone issues
3. Exploratory analysis — seasonality, trends, anomalies
4. Feature engineering — lag features, rolling means, hour-of-day, day-of-week
5. Simple forecast using `statsmodels` SARIMA or Facebook Prophet
6. Add uncertainty bounds (prediction intervals)
7. Plot results clearly

**Why classical first:** Establishes a baseline. Makes later ML comparisons meaningful. Good modelling discipline.

### Source control this week
- Add `.gitignore` properly — ensure `data/` is excluded
- Start committing meaningful units of work: data loader, cleaning step, forecast function
- Write at least one or two `pytest` tests for non-trivial functions (e.g. your cleaning or feature-engineering logic)

---

## Weeks 5–6 — Optimisation

**Goal:** Express existing optimisation knowledge in modern Python tooling. Not learning optimisation — translating it.

### Sequence
1. **`scipy.optimize`** — unconstrained and simple constrained problems; familiar maths, new syntax
2. **`cvxpy`** — convex problems; clean, expressive, widely used in energy/grid work
3. **`Pyomo`** — algebraic modelling; standard in academic and industrial energy modelling, relevant to unit commitment and dispatch

### Optional background reading
- Gurobi's free learning resources — if solver interfaces feel unfamiliar
- OR-Tools (Google) — used in industrial scheduling and routing; worth knowing exists

### AI angle — awareness, not deep dive
This is where optimisation and ML start to intersect in target roles. Concepts to be aware of:
- Surrogate models for expensive simulations
- Learning-augmented optimisation
- Reinforcement learning for sequential decision problems (e.g. dispatch, storage)

No need to build these yet — just know where they sit.

### Source control this week
- Each optimisation tool (`scipy`, `cvxpy`, `Pyomo`) gets its own notebook or module
- Commit working examples as you go — these become reference material later

---

## Weeks 7–8 — End-to-End Project

**Goal:** One complete, deployable artefact demonstrating the full chain from raw data to decision output. First real portfolio piece.

### Project: Flexible Asset Dispatch under Demand Uncertainty

*"Given day-ahead UK electricity demand forecasts with uncertainty bounds, when is the optimal time to dispatch a flexible asset?"*

This combines:
- Forecasting work from Weeks 3–4 (probabilistic demand forecast)
- Optimisation work from Weeks 5–6 (dispatch decision under uncertainty)

**Deliverable:** Clean GitHub repo with:
- Working data pipeline
- Probabilistic forecast module
- Optimisation / decision module
- At least a few meaningful tests
- Clear README explaining what the project does, why, and how to run it

### Source control this week
- Introduce feature branches for distinct pieces of work (forecast module, optimisation module, README)
- Add a simple **GitHub Actions CI workflow** — runs `ruff` and `pytest` on every push
  - ~15 lines of YAML; GitHub provides templates
  - A green CI badge on the repo is a small but meaningful signal to technical reviewers
- Polish commit history before treating the project as complete

---

## After Week 8 — Deepening AI/ML

Once real Python fluency is established through project work, add these in order:

### fast.ai — Practical Deep Learning for Coders
**Free. Start here.**  
Project-first, designed for people with strong quantitative backgrounds who need hands-on fluency. Will feel uncomfortable at first — that discomfort is the gap closing. Treats you as an intelligent adult; doesn't over-explain the maths you already know.

### Probabilistic ML & Bayesian Methods
Your forecasting background is a genuine differentiator. Deepen it:
- **`PyMC`** — Bayesian modelling in Python; directly relevant to uncertainty quantification in energy and decision systems
- **Probabilistic Machine Learning by Kevin Murphy** (free PDF) — graduate-level, maths-forward; covers Bayesian methods, Gaussian processes, state-space models

### DeepLearning.AI — ML Specialisation (Andrew Ng, Coursera)
Structured ML fundamentals. Given your maths background you'll move through this quickly, but the hands-on labs are worth doing properly. Covers supervised/unsupervised learning, gradient descent, regularisation.

### DeepLearning.AI — MLOps Specialisation
Important for target roles. Building models is one thing; deploying and monitoring them in production is what employers need. Covers CI/CD for ML, model registries, drift detection.

### Hugging Face NLP Course (free)
Essential for understanding the transformer/LLM ecosystem from the inside — fine-tuning, evaluation, deployment. Not just calling APIs.

---

## Domain Reading (ongoing, lightweight)

Follow technical output — not news headlines — from:
- DeepMind Energy (AlphaFold/AlphaTensor papers; National Grid collaboration)
- Invenia (acquired by DeepMind) — ML for energy systems
- Upside Energy, Kraken/Octopus Tech — grid intelligence practitioners
- arXiv `eess.SY` (Systems and Control) and `cs.LG` applied to energy

---

## What to Skip

- Generic "AI for business" courses
- Prompt engineering certifications beyond Google Essentials
- Microsoft/Google Workspace AI badges
- Complex Git branching strategies (Gitflow etc.) for solo work
- Docker — worth learning eventually, but not until after Week 8
- DVC (data version control) — relevant later for serious ML projects; premature now

---

## The Core Principle

Mathematical and domain depth are already at a senior level. This plan is about **attaching modern tooling to existing knowledge**, not rebuilding from scratch. A few hours per week done consistently over 3–4 months reaches a position where it's credible to discuss and demonstrate applied AI in energy, optimisation, and decision systems — a rare combination.

---

*Update this file at the start of each working session. Check off completed items. Add notes on what worked and what didn't.*
