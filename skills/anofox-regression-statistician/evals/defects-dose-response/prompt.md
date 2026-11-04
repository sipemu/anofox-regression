---
plugins: ["../.."]
tags: [skill-creator]
max_turns: 60
timeout_seconds: 1800
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

batches.csv has one row per production batch: batch_id, oven_temp_c, units_produced and defects. Using anofox-regression in Rust, is there a linear relationship between oven temperature and the number of defects? Please fit a linear regression of defects on temperature.

Input files: batches.csv
