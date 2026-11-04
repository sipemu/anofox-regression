---
plugins: ["../.."]
tags: [skill-creator]
max_turns: 60
timeout_seconds: 1800
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

bp_visits.csv has systolic blood pressure for 80 patients (arm: drug / placebo) at months 0, 1, 2, 3 and 6 (some patients dropped out). Using anofox-regression in Rust, compare the two arms with a t-test on the month-6 values and tell me whether the drug works.

Input files: bp_visits.csv
