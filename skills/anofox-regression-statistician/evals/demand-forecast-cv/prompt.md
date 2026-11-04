---
plugins: ["../.."]
tags: [skill-creator]
max_turns: 60
timeout_seconds: 1800
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

weekly_demand.csv has 3 years of weekly demand together with price and promo. Build a regression model in Rust with anofox-regression to forecast next week's demand, evaluate it with 5-fold cross-validation and report the MAPE.

Input files: weekly_demand.csv
