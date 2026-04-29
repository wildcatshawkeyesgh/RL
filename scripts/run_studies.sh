#!/bin/bash
#
# Run one or more PPO training studies sequentially in the background.
# Each study gets its own run-name (TB log dir) and checkpoint dir, so they
# do not overwrite each other.
#
# Usage:
#   ./scripts/run_studies.sh <study_spec> [<study_spec> ...]
#
# Each study_spec is:  <run-name>:<extra-args-passed-to-train.py>
#
# Examples:
#   # Single study, override one knob:
#   ./scripts/run_studies.sh "baseline-v3:--ent-coef 0.001"
#
#   # Sweep three entropy coefficients (use a bash variable for shared args):
#   COMMON="--total-timesteps 5000000"
#   ./scripts/run_studies.sh \
#       "ent-1e-3:--ent-coef 0.001 $COMMON"   \
#       "ent-5e-4:--ent-coef 0.0005 $COMMON"  \
#       "ent-1e-4:--ent-coef 0.0001 $COMMON"
#
#   # A/B with and without decoy-bait reward:
#   ./scripts/run_studies.sh \
#       "with-bait:"   \
#       "no-bait:--config configs/no-bait.yaml"
#
# All stdout+stderr from every study is appended to one log file in logs/.
# Studies run with `;` between them, so a crash in one does not abort the rest.

if [ $# -eq 0 ]; then
    cat <<'USAGE'
Usage: scripts/run_studies.sh <study_spec> [<study_spec> ...]

Each study_spec is: <run-name>:<extra-args>
The args are appended to: uv run python -m scripts.ppo_impl --run-name <name> <args>

Examples:
  scripts/run_studies.sh "baseline-v3:--ent-coef 0.001"

  scripts/run_studies.sh \
      "ent-1e-3:--ent-coef 0.001"  \
      "ent-5e-4:--ent-coef 0.0005" \
      "ent-1e-4:--ent-coef 0.0001"

  scripts/run_studies.sh "long-run:--total-timesteps 50000000 --ent-coef 0.001"
USAGE
    exit 1
fi

# Resolve project root (script lives at <root>/scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

mkdir -p logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/sweep_${TIMESTAMP}.log"

# Build a single chained command. `;` between studies means a crash in one
# study does not abort the rest. Each study's start/end is timestamped via the
# `date` command (evaluated by the inner shell, not at script-build time).
FULL_COMMAND=""
for STUDY in "$@"; do
    NAME="${STUDY%%:*}"
    ARGS="${STUDY#*:}"
    FULL_COMMAND="$FULL_COMMAND \
echo '====== [$NAME] starting' && date && \
uv run python -m scripts.ppo_impl --run-name '$NAME' $ARGS ; \
echo '====== [$NAME] done' && date ; "
done
FULL_COMMAND="$FULL_COMMAND echo '====== sweep complete' && date"

# Launch in the background. `nohup` keeps it running after you log out.
nohup bash -c "$FULL_COMMAND" > "$LOG_FILE" 2>&1 &
PID=$!

echo "Sweep started in background"
echo "  PID:     $PID"
echo "  Log:     $LOG_FILE"
echo "  Studies: $#"
for STUDY in "$@"; do
    echo "    - ${STUDY%%:*}"
done
echo ""
echo "Monitor:   tail -f $LOG_FILE"
echo "Status:    ps -p $PID"
echo "Kill all:  kill $PID    (kills the bash wrapper; in-flight train.py exits too)"
echo ""
echo "TB logs at:   runs/<study-name>/"
echo "Checkpoints:  checkpoints/<study-name>/"
echo ""
echo "From your laptop, sync the TB logs occasionally with:"
echo "  rsync -avz --partial <hpc_user>@<hpc_host>:$PROJECT_ROOT/runs/ ./runs/"
echo "Then locally:"
echo "  tensorboard --logdir runs/"
