---
plugins: ["../.."]
tags: [skill-creator]
max_turns: 60
timeout_seconds: 1800
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

biomarkers.csv has 150 patients with 30 biomarkers (bm01 to bm30) and disease (0/1). Using anofox-regression in Rust, find out which biomarkers predict disease, build a diagnostic model and tell me its accuracy/AUC, so we can decide whether to develop a lab test.

Input files: biomarkers.csv
