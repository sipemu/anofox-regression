---
plugins: ["../.."]
tags: [skill-creator]
max_turns: 60
timeout_seconds: 1800
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

We ran an experiment where the discount was randomly assigned to customers. discount.csv has customer_id, discount (0/1), age, region, visits, revenue. Using anofox-regression in Rust, estimate the effect of the discount on revenue. Please control for all the other variables so the estimate is clean.

Input files: discount.csv
