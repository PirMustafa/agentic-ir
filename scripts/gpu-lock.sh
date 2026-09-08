#!/usr/bin/env bash
# Mutual exclusion for GPU evaluations, shared by every executor worktree.
#
# Last round two agents launched run_eval concurrently on a machine with ~3 GB
# of RAM free and an 8 GB card. One of them died with SIGSEGV at question 6 and
# the agent spent its budget rediscovering why. An instruction to "wait for the
# other agent" was not enough, so this is a lock rather than a request.
#
# mkdir is the atomic primitive: it either creates the directory or fails, with
# no window between checking and creating. A stale lock older than 2 hours is
# reclaimed, because an agent that dies holding one must not block the other
# forever.
#
#   bash /d/ir_agent_worktrees/gpu-lock.sh acquire <your-name>   # blocks
#   bash /d/ir_agent_worktrees/gpu-lock.sh release <your-name>
#   bash /d/ir_agent_worktrees/gpu-lock.sh status

LOCK=/d/ir_agent_worktrees/.gpu.lock
STALE_SECONDS=7200

case "${1:-status}" in
  acquire)
    who="${2:-anonymous}"
    waited=0
    while true; do
      if mkdir "$LOCK" 2>/dev/null; then
        echo "$who $(date +%s)" > "$LOCK/owner"
        echo "GPU lock acquired by $who after ${waited}s"
        exit 0
      fi
      # Reclaim a lock whose holder is gone.
      if [ -f "$LOCK/owner" ]; then
        started=$(awk '{print $2}' "$LOCK/owner" 2>/dev/null || echo 0)
        now=$(date +%s)
        if [ -n "$started" ] && [ $((now - started)) -gt "$STALE_SECONDS" ]; then
          echo "reclaiming stale lock held by $(awk '{print $1}' "$LOCK/owner") for $((now - started))s"
          rm -rf "$LOCK"
          continue
        fi
      fi
      if [ $((waited % 60)) -eq 0 ]; then
        echo "waiting for GPU lock (held by $(awk '{print $1}' "$LOCK/owner" 2>/dev/null || echo '?'), ${waited}s so far)"
      fi
      sleep 15
      waited=$((waited + 15))
    done
    ;;
  release)
    if [ -d "$LOCK" ]; then rm -rf "$LOCK"; echo "GPU lock released by ${2:-anonymous}"; else echo "no lock held"; fi
    ;;
  status)
    if [ -d "$LOCK" ]; then echo "HELD by $(cat "$LOCK/owner" 2>/dev/null)"; else echo "free"; fi
    ;;
  *) echo "usage: $0 {acquire|release|status} [name]"; exit 2 ;;
esac
