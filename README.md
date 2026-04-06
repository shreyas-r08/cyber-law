# 🔐 Graph-Based Network Risk Simulator
### Dwell Time & Lateral Movement Analysis for Cybersecurity Research

> **Academic / Defensive Use Only** — This tool models network risk mathematically.  
> It contains no exploit code, attack tools, or real-world payloads.

---

## Overview

This simulator models an enterprise network as a **directed weighted graph** and applies
**Dijkstra's Algorithm** to compute two competing traversal paths:

| Path | Optimised For | Implication |
|------|--------------|-------------|
| 🟢 Lowest-Risk | Minimum cumulative `risk_weight` | Maximises dwell time; avoids detection |
| 🔴 Shortest-Time | Minimum cumulative `time_cost` | Fastest traversal; higher detection risk |

The core research question: *When an adversary optimises for stealth rather than speed,
how does dwell time increase and how does detection probability change?*

---

## Detection Probability Model

Per-hop detection probability:

```
P_hop = 1 - exp( -(risk × monitoring_strength) / time )
```

Cumulative detection uses the complement product (independent hops):

```
P_total = 1 - ∏(1 - P_hop_i)
```

**Parameters:**
- `risk_weight` — edge-level detection probability (0–1)
- `time_cost`   — time units required to traverse the edge
- `monitoring_strength` — adjustable slider; models SOC/IDS sensitivity

---

## Project Structure

```
network_risk_sim/
├── app.py           # Streamlit UI (main entry point)
├── graph_utils.py   # Graph construction + Dijkstra wrappers
├── simulation.py    # Dwell time + detection probability engine
├── requirements.txt # Python dependencies
└── README.md        # This file
```

---

## Setup & Installation

### Prerequisites
- Python 3.9 or higher
- pip

### Steps

```bash
# 1. Clone / download the project folder
cd network_risk_sim

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**

---

## How to Use

### Step 1 — Generate a Graph
In the **sidebar**, choose:
- **Predefined Enterprise Network** — a realistic 10-node network with named systems
- **Random Graph Generator** — configure node count, risk/time ranges, and seed

Click **⚙ GENERATE GRAPH**.

### Step 2 — Select Paths
Choose a **Source Node** and **Target Node** from the dropdowns.

Adjust the **Monitoring Strength** slider (0.1 = minimal; 3.0 = highly monitored).

Click **▶ RUN SIMULATION**.

### Step 3 — Explore Results

| Tab | Content |
|-----|---------|
| 📡 Graph View | Network topology with highlighted paths + node risk heatmap |
| 🔍 Path Analysis | Side-by-side path comparison, metrics, detection probability chart |
| 📊 Simulation Detail | Per-hop breakdown + monitoring sensitivity curve |
| 🗂 Data | Full edge/node tables + CSV export |

---

## Graph Model

### Predefined Network Nodes

| Node | Role | Sensitivity |
|------|------|-------------|
| UserPC_1 | Endpoint | 1 |
| UserPC_2 | Endpoint | 1 |
| PrintServer | Server | 2 |
| FileServer | Server | 3 |
| WebServer | DMZ | 2 |
| AppServer | Server | 3 |
| Database | Database | 5 |
| DomainController | Critical | 5 |
| BackupServer | Server | 4 |
| SIEM | Security | 4 |

### Edge Attributes

- **risk_weight** (0–1): probability this hop triggers a security alert
- **time_cost** (integer units): time required to complete the hop

---

## Research Notes

### Why Dwell Time Increases with Risk Optimisation

When traversing via low-risk edges, an adversary:
1. Takes longer routes (more hops, each with small risk weights)
2. Spends more time on each hop (higher `time_cost` per unit risk)
3. Keeps per-hop detection probability extremely low

This is the **stealth-vs-speed tradeoff**: the fastest path is the most detectable.

### Monitoring Strength Sensitivity

The sensitivity chart (Tab 3) shows how detection probability scales with SOC capabilities.
At monitoring_strength = 1.0, a low-risk path may evade detection.
At monitoring_strength = 3.0, even low-risk paths become more exposed.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `networkx` | Graph modeling + Dijkstra's algorithm |
| `matplotlib` | Graph and chart visualisation |
| `numpy` | Numerical computations |
| `pandas` | Tabular data + CSV export |

---

## Disclaimer

This tool is intended solely for **academic, educational, and defensive cybersecurity research**.
It does not contain, facilitate, or demonstrate any real attack capabilities.
All graphs and weights are synthetic and do not represent any real infrastructure.
