# Agent System Definition

## System Agents

### Quality Gate Agent
- **Type**: Validator
- **Priority**: Critical
- **Tools**:
  - `pmat_analyze_complexity`
  - `pmat_detect_satd`
  - `pmat_security_scan`

### Refactoring Agent
- **Type**: Transformer
- **Priority**: High
- **Tools**:
  - `pmat_refactor_code`
  - `pmat_apply_patterns`

### Analysis Agent
- **Type**: Analyzer
- **Priority**: Normal
- **Tools**:
  - `pmat_analyze_code`
  - `pmat_generate_metrics`

## Communication Protocol

- **Message Format**: JSON
- **Transport**: MCP
- **Discovery**: Auto

## Quality Requirements

- **Complexity Limit**: 8
- **Coverage Minimum**: 95%
- **SATD Tolerance**: 0

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
