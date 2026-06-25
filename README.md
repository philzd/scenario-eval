# Scenario Evaluation & Data Value Platform

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-portfolio-lightgrey)

A data and evaluation platform that transforms operational AI logs into curated, evaluation-ready datasets through signal extraction, scenario prioritization, human review, and reproducible workflows.

This project demonstrates how large volumes of operational telemetry can be converted into structured scenarios, evaluation signals, ranked review queues, and curated datasets. The current implementation uses autonomous driving telemetry, but the architecture is intended to generalize to robotics, embodied AI, simulation, teleoperation, and other operational AI systems.

The current implementation uses synthetic telemetry to validate the evaluation workflow. Future work includes applying the platform to real-world operational datasets.

---

## TL;DR

This repository implements a simplified data and evaluation workflow for operational AI systems.

```text
Raw logs
↓
Signal extraction
↓
Scenario discovery
↓
Scenario metrics
↓
Scenario ranking
↓
Review queue
↓
Human review
↓
Curated dataset
↓
Dataset manifest
↓
Coverage analysis
↓
Failure discovery
↓
Engineering insights
↓
Engineering decisions
```

The goal is to help engineers identify, prioritize, review, and curate the scenarios that matter most from large volumes of operational data.

---

## Example Evaluation Outputs

Below are representative artifacts produced by the platform.

### Extracted Scenarios

Structured scenario windows extracted from raw telemetry.

Artifact:

`outputs/scenarios.jsonl`

Each scenario contains:

- `scenario_id`
- `start_time`
- `end_time`
- `event_time`
- `trigger_types`
- `source_log`

### Scenario Metrics

Scenario-level evaluation signals.

Artifact:

`outputs/scenario_metrics.jsonl`

Examples:

-  `min_ttc`
- `min_distance_to_object`
- `max_abs_jerk`
- `max_deceleration`
- `mean_speed`
- `duration_s`

### Scenario Scores

Interpretable prioritization signals.

Artifact:

`outputs/scenario_scores.jsonl`

Examples:

- `severity_score`
- `rarity_score`
- `value_score`
- `rank`

### Ranked Scenarios

Prioritized scenarios ordered by overall value.

Artifact:

`outputs/ranked_scenarios.csv`

### Observability Visualizations

Visualizations used to inspect extraction, scoring, and prioritization behavior.

Artifact:

`outputs/plots/`

Includes:

- `speed_timeline.png`
- `ttc_timeline.png`
- `acceleration_jerk_timeline.png`
- `ranked_scenario_scores.png`

These visualizations help explain:

- where scenarios occurred
- why scenarios were extracted
- why scenarios were prioritized

---

## Key Features

- Scenario extraction from operational telemetry
- Interpretable evaluation metrics
- Severity and rarity scoring
- Scenario prioritization and ranking
- Visualization and observability tooling
- Reproducible evaluation artifacts
- Human-in-the-loop dataset curation

---

## Current Status

The current implementation covers scenario prioritization, human review, dataset curation, coverage analysis, and observability workflows.

### Implemented

- ✓ Automated ranking pipeline
- ✓ Review queue generation
- ✓ Human review workflow
- ✓ Curated dataset generation
- ✓ Dataset manifest generation
- ✓ Coverage analysis
- ✓ Visualization
- ✓ Failure summary analysis

---

## Table of Contents

- [Quick Demo](#quick-demo)
- [Getting Started](#getting-started)
- [Why This Project Exists](#why-this-project-exists)
- [What This Project Is and Is Not](#what-this-project-is--is-not)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Pipeline Stages](#pipeline-stages)
- [Core Guarantees](#core-guarantees)
- [Tech Stack](#tech-stack)
- [Future Extensions](#future-extensions)
- [Project Scope](#project-scope)
- [Usage Notice](#usage-notice)
- [Author](#author)

---

## Quick Demo

1. Run the automated ranking pipeline:

```bash
python -m src.run_pipeline --rows 2000
```

Example output:

```bash
Generating synthetic telemetry...
Extracting scenarios...
Computing scenario metrics...
Scoring and ranking scenarios...

Pipeline complete.

Telemetry rows generated:   2000
Scenarios extracted:        8
Metric records written:     8
Scores written:             8
Ranked CSV:                 outputs/ranked_scenarios.csv
```

2. Generate visualizations:

```bash
python -m src.visualize_results
```

3. Generate a review queue:

```bash
python -m src.create_review_queue
```

4. Edit `outputs/review_queue.csv` to update:

- `review_status`
- `label`
- `review_notes`

5. Build the curated dataset and manifest:

```bash
python -m src.build_curated_dataset
python -m src.build_dataset_manifest
python -m src.build_coverage_report
python -m src.build_coverage_gaps
python -m src.build_failure_summary
```

---

## Getting Started

The following steps demonstrate how to run the evaluation pipeline on synthetic telemetry data.

1. Clone the repository

```bash
git clone https://github.com/philzd/scenario-eval.git
cd scenario_eval
```

2. Create a Python environment and install dependencies:

```bash
python -m venv env
source env/bin/activate
pip install -e .
pip install -r requirements.txt
```

3. Run the pipeline:

```bash
python -m src.run_pipeline --rows 2000
```

4. Generate visualizations:
   
```bash
python -m src.visualize_results
```

---

## Why This Project Exists

Operational AI systems generate large volumes of telemetry, sensor data, and execution logs.

In practice, engineers cannot manually inspect every event.

Instead, they need systems that help answer:

- What happened?
- What matters?
- What should we investigate next?

This project demonstrates a workflow that transforms raw telemetry into structured scenarios, evaluation signals, ranked review queues, curated datasets, and dataset-level metadata.

The architecture is inspired by evaluation and data workflows commonly found in robotics, autonomous systems, and embodied AI platforms.

---

## What This Project Is / Is Not

### Is

- A data and evaluation platform
- A scenario discovery workflow
- A scenario prioritization system
- A human-in-the-loop review workflow
- A dataset curation workflow
- A decision-support system for operational AI

### Is Not

- A full autonomous driving stack
- A model training pipeline
- A distributed compute platform
- A cloud-native deployment

The focus is **evaluation, prioritization, review, and dataset curation**.

---

## System Architecture

The platform transforms operational logs into evaluation-ready datasets and engineering insights.

```text
Raw logs
↓
Scenario extraction
↓
Scenario metrics
↓
Scenario scoring
↓
Ranked scenarios
↓
Review queue
↓
Human review
↓
Curated dataset
↓
Dataset manifest
↓
Coverage analysis
↓
Failure discovery
↓
Engineering insights
↓
Engineering decisions
```

Examples of engineering decisions include:

- What should we evaluate next?
- What should we train on next?
- Which failure modes deserve investigation?
- Which scenarios should be added to future datasets?

---

## Repository Structure

```text
scenario_eval/

├── data/
│   └── raw/
│
├── outputs/
│   ├── plots/
│   ├── scenarios.jsonl
│   ├── scenario_metrics.jsonl
│   ├── scenario_scores.jsonl
│   ├── ranked_scenarios.csv
│   ├── review_queue.csv
│   ├── curated_dataset.jsonl
│   ├── dataset_manifest.json
│   ├── coverage_report.json
│   ├── coverage_gaps.json
│   └── failure_summary.json
│
├── docs/
│
├── src/
│   ├── generate_synthetic_logs.py
│   ├── extract_scenarios.py
│   ├── compute_metrics.py
│   ├── score_scenarios.py
│   ├── visualize_results.py
│   ├── run_pipeline.py
│   ├── create_review_queue.py
│   ├── build_curated_dataset.py
│   ├── build_dataset_manifest.py
│   ├── build_coverage_report.py
│   ├── build_coverage_gaps.py
│   └── build_failure_summary.py
│
└── tests/
```

Note: `data/` is generated automatically when the pipeline runs.

---

## Pipeline Stages

### 1. Synthetic telemetry generation

Synthetic operational telemetry is generated to simulate operational logs.

Artifacts produced:

`data/raw/synthetic_telemetry.csv`

### 2. Scenario extraction

Continuous telemetry is converted into discrete scenario windows.

Current trigger types:

- `low_ttc`
- `high_jerk`
- `hard_braking`

Artifacts produced:

`outputs/scenarios.jsonl`

### 3. Scenario metrics

Metrics are computed for each extracted scenario.

Examples include:

- `min_ttc`
- `min_distance_to_object`
- `max_abs_jerk`
- `max_deceleration`
- `mean_speed`

Artifacts produced:

`outputs/scenario_metrics.jsonl`

### 4. Scenario scoring

Scenarios are scored using interpretable evaluation signals.

Current scoring components:

- severity
- rarity

Artifacts produced:

`outputs/scenario_scores.jsonl`

### 5. Scenario ranking

Scenarios are ranked according to overall value using severity and rarity scores.

Artifacts produced:

`outputs/ranked_scenarios.csv`

### 6. Visualization

Visualization tools provide observability into extraction and prioritization behavior.

Artifacts produced:

- `outputs/plots/speed_timeline.png`
- `outputs/plots/ttc_timeline.png`
- `outputs/plots/acceleration_jerk_timeline.png`
- `outputs/plots/ranked_scenario_scores.png`

### 7. Human review workflow

Ranked scenarios are converted into a review queue for human inspection.

Artifacts produced:

`outputs/review_queue.csv`

Human reviewers can update:

- `review_status`
- `label`
- `review_notes`

Only scenarios marked as accepted are included in the curated dataset.

Artifacts produced:

`outputs/curated_dataset.jsonl`

Dataset-level metadata is generated for tracking dataset composition.

Artifacts produced:

`outputs/dataset_manifest.json`

### 8. Coverage analysis

Coverage analysis summarizes review outcomes and dataset composition.

Artifacts produced:

- `outputs/coverage_report.json`
- `outputs/coverage_gaps.json`

Coverage analysis reports:

- review status counts
- acceptance rate
- label distribution
- represented labels
- missing labels

These artifacts help identify dataset coverage gaps and guide future review and data collection efforts.

### 9. Failure summary

Failure summary aggregates reviewed scenarios into engineering-oriented insights.

Artifacts produced:

- `outputs/failure_summary.json`

Examples include:

- trigger type counts
- review label counts
- accepted label counts
- highest-value scenarios
- highest-value accepted scenarios

These summaries help engineers identify recurring operational behaviors and prioritize future investigation.

---

## Core Guarantees

### Signal Traceability

Every ranked scenario can be traced back to the source log, trigger events, and scenario metrics that contributed to its prioritization.

### Reproducibility

Given the same inputs and configuration, the platform produces identical evaluation artifacts and rankings.

### Human Reviewability

All prioritization outputs are designed to be inspectable and reviewable by engineers before being incorporated into curated datasets.

### Observability

The platform provides metrics and visualizations that help explain why scenarios were extracted, scored, and prioritized.

---

## Tech Stack

- Python
- Pandas
- NumPy
- JSONL / CSV artifact pipelines
- Matplotlib

---

## Future Extensions

Possible future extensions include:

- Real-world operational dataset adapters
- Failure pattern analysis
- Review observability
- Dataset health metrics
- Coverage dashboards
- Docker
- GitHub Actions
- PySpark
- GCP
- Terraform
- Ray

---

## Project Scope

This project focuses on the workflow connecting:

- operational data
- evaluation
- prioritization
- human review
- dataset curation

It does not include:

- model training
- production deployment
- distributed serving infrastructure

---

## Usage Notice

This repository is shared for portfolio and demonstration purposes.

Please contact the author before reusing or redistributing the code.

---

## Author

Philippe Do
