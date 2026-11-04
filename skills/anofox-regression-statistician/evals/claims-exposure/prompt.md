---
plugins: ["../.."]
tags: [skill-creator]
max_turns: 60
timeout_seconds: 1800
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

claims.csv has policy_id, age_group (young/middle/senior), vehicle_type (sedan/suv/sports), exposure_years and n_claims for 3000 car insurance policies. Using anofox-regression in Rust, fit a regression of the number of claims on age group and vehicle type to see which drivers are riskiest.

Input files: claims.csv
