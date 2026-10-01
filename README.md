# Autonomous Clinical Claims Integrity Graph

A stateful multi-agent claims auditing engine engineered with **LangGraph**, **Gemini**, **Jev (TypeSafe AI)**, and **Pydantic** to detect billing fraud, anatomical laterality contradictions, and statistical cost outliers.

---

## Architecture Overview

```
                      +-------------------+
                      |   Planner Node    | (Plan-and-Solve)
                      +---------+---------+
                                |
                                v
               +------->+-----------------+<-------+
               |        |  Reasoner Node  |        |
               |        +--------+--------+        |
               |                 |                 |
         [Tool Invoked]   [Adjudication Ready]     |
               |                 |                 |
               v                 v                 |
       +---------------+ +---------------+         |
       |  Tools Node   | | Adjudication  |         |
       | (Z-Score/CPT) | | (Structured)  |         |
       +-------+-------+ +-------+-------+         |
               |                 |                 |
               +-----------------+                 |
                                 |                 |
                                 v                 |
                       +-------------------+       |
                       |   Jev Guardrail   |       |
                       | (TypeSafe Runtime)|       |
                       +---------+---------+       |
                                 |                 |
                   [Pass]        |       [Fail / Retry]
                     +-----------+-----------+
                     |
                     v
                  [ END ]
```

The system replaces rigid deterministic scripts with an autonomous, resilient agentic loop:
- **Plan-and-Solve Orchestration:** The `planner` analyzes claim diagnostic codes and clinical notes to establish structured validation steps before tool execution.
- **Dynamic Tool Calling (ReAct Loop):** The agent dynamically inspects billing unit ceiling matrices and executes statistical $Z$-score anomaly computations across historical encounter costs.
- **State Persistence:** Implements LangGraph memory checkpointers (`MemorySaver`) to preserve state, track validation attempts, and maintain execution lineage across sessions.
- **Runtime Contract Invariants (Jev by TypeSafe AI):** Programmatic guardrails validate domain invariants (e.g., forbidding high-risk auto-approvals and enforcing minimum confidence floors) before finalizing decisions.
- **Self-Correction Feedback Loop:** When invariant checks or mathematical guardrails fail, the graph routes errors back to the reasoning agent for contextual reflection rather than terminating abruptly.

---

## Tech Stack & Dependencies

- **Orchestration:** LangGraph (State Graphs, Dynamic Routing, Checkpointing)
- **Runtime Contract Enforcement:** Jev (TypeSafe AI) & Pydantic v2
- **Model Provider:** Google Gemini API via `langchain-google-genai`
- **Execution & Package Management:** [uv](https://github.com/astral-sh/uv) / Python 3.12+

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/sbajaj4/healthcare-agent-poc.git
cd healthcare-agent-poc
```

### 2. Configure Environment Variables
Copy the template and supply your API credentials:
```bash
cp .env.example .env
```
Ensure your `.env` contains:
```dotenv
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run the Pipeline Live
Execute using `uv` with dependency resolution:
```bash
uv run --env-file .env audit_pipeline.py
```
Or via standard `pip`:
```bash
pip install -r requirements.txt
python audit_pipeline.py
```

---

## Production Schema Contract

All agent adjudications are deterministically checked against the strict `AuditDecision` contract:

```json
{
  "audit_status": "FLAGGED_FOR_HUMAN_REVIEW",
  "risk_classification": "HIGH",
  "anomalies_detected": [
    "Ground Truth Outlier Breach: OUTLIER_BREACH: Z-score is 3.12 (> 2.5)."
  ],
  "clinical_justification": "Encounter cost of $2400 substantially deviates from historical baseline ($85-$95). ICD-10 J45.909 and CPT 94640 align with standard nebulizer therapy, but severe cost variance warrants manual supervisor review.",
  "confidence_score": 0.95
}
```
