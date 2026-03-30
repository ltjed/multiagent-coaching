---
name: run-remote
description: Run code on the Kubernetes GPU pod. Use this whenever you need to execute code, run tests, or launch training jobs. The local machine has no GPU — always prefer running on the remote pod.
argument-hint: <command to run on pod>
---

# Run Remote Command on GPU Pod

Command to execute: **$0**

## Instructions

Follow these steps exactly. Do NOT guess durations or sleep — use polling.

### Step 1: Discover the pod

```bash
kubectl get pods -n mltraining-dev --no-headers | grep tiali | awk '{print $1}'
```

Save the pod name. If multiple pods are found, pick the first one that is `Running`.
If no pod is found, tell the user and stop.

### Step 2: Launch the command in background

Run the user's command on the pod using `kubectl exec` via the **Bash tool with `run_in_background: true`**:

```bash
kubectl exec -n mltraining-dev <POD_NAME> -- bash -c 'cd /code/users/tiali/maracaibo && source .venv/bin/activate && $0'
```

Key details:
- Always `cd /code/users/tiali/maracaibo` and `source .venv/bin/activate` first
- Use `run_in_background: true` so the main agent is not blocked
- Note the **task_id** returned — you need it for polling

### Step 3: Poll for completion

Use the **Agent tool** to spawn a monitoring sub-agent. The sub-agent's ONLY job is to poll:

**Sub-agent prompt:**
> Check whether background task `<TASK_ID>` has completed. Use `TaskOutput` with `block: true` and `timeout: 30000` (30 seconds). If the task is still running, call `TaskOutput` again with the same parameters. Repeat until the task completes. When it finishes, return the full output (stdout and stderr). Do NOT do anything else — just monitor and return results.

Use `run_in_background: false` (foreground) so the main agent **waits** for the sub-agent to finish — this keeps the conversation flow intact without returning control prematurely.

### Step 4: Report results

Once the sub-agent returns with the output:
- Show the user the command output (stdout/stderr)
- If the command failed (non-zero exit), highlight the error
- If output is very long, summarize the key parts and offer to show the full output

## Important rules

- **NEVER sleep or guess duration.** Always poll via TaskOutput.
- **NEVER run GPU workloads locally.** Always use the pod.
- The `.venv` environment on the pod has the correct pinned versions. Do not use conda.
- If the pod restarted recently, you may need to run `uv python install 3.11` first — check if python works before running the main command.
- For long-running training jobs, the sub-agent polling loop will keep going until completion. This is by design.
