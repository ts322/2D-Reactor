#!/usr/bin/env bash
set -euo pipefail

# run_case_no_view.sh
# Usage:
#   ./run_case_no_view.sh MESH_SOURCE ITER [RESULTS_ROOT] [--use-host-foam]
#
# Example:
#   ./run_case_no_view.sh /abs/path/to/Mesh 7 /abs/path/to/Results
#
# Environment:
#   OPENFOAM_DOCKER_IMAGE (optional) - docker image with OpenFOAM, default "opencfd/openfoam-default:2506"

IMAGE_DEFAULT="opencfd/openfoam-default:2506"
OPENFOAM_IMAGE="${OPENFOAM_DOCKER_IMAGE:-$IMAGE_DEFAULT}"

if [ $# -lt 2 ]; then
  echo "Usage: $0 MESH_SOURCE ITER [RESULTS_ROOT] [--use-host-foam]" >&2
  exit 2
fi

MESH_SOURCE="$1"
ITER="$2"
RESULTS_ROOT="${3:-$HOME/2D-Reactor/2D_Reactor/Results}"

# parse optional flag(s)
USE_HOST_FOAM=false
shift 2
for arg in "$@"; do
  case "$arg" in
    --use-host-foam) USE_HOST_FOAM=true ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

# validate
if [ ! -d "$MESH_SOURCE" ]; then
  echo "ERROR: MESH_SOURCE not found: $MESH_SOURCE" >&2
  exit 2
fi

stamp="Results_iter_${ITER}"
RUN_DIR="${RESULTS_ROOT}/${stamp}"
mkdir -p "$RUN_DIR"
HOST_RUN_DIR="$(cd "$RUN_DIR" && pwd)"

# copy mesh contents into run dir (including hidden files)
cp -R "${MESH_SOURCE}/." "${HOST_RUN_DIR}/"

# helper to run FOAM commands: prefer host if available & requested; otherwise use docker
_run_foam_cmd() {
  cmd="$1"   # e.g. blockMesh, pimpleFoam, checkMesh
  # If user requested host and binary exists, run on host
  if [ "$USE_HOST_FOAM" = true ] && command -v "$cmd" >/dev/null 2>&1; then
    (cd "$HOST_RUN_DIR" && "$cmd" -case .)
    return $?
  fi

  # If binary present on host and user didn't force docker, use it
  if command -v "$cmd" >/dev/null 2>&1 && [ "$USE_HOST_FOAM" = false ]; then
    (cd "$HOST_RUN_DIR" && "$cmd" -case .)
    return $?
  fi

  # Docker fallback
  if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: Neither host OpenFOAM binary nor docker available to run '$cmd'." >&2
    exit 3
  fi
  if [ -z "$OPENFOAM_IMAGE" ]; then
    echo "ERROR: OPENFOAM_DOCKER_IMAGE not set and no default available." >&2
    exit 3
  fi

  uid="$(id -u)"
  gid="$(id -g)"
  docker run --rm -u "${uid}:${gid}" \
    -v "${HOST_RUN_DIR}:/home/openfoam/case" -w /home/openfoam/case \
    "${OPENFOAM_IMAGE}" \
    bash -lc "${cmd} -case /home/openfoam/case"
}

# Run the case
echo "[RUN] blockMesh for case: ${HOST_RUN_DIR}"
_run_foam_cmd blockMesh

echo "[RUN] checkMesh -allGeometry -allTopology"
# Some checkMesh variants accept -all flags; call explicitly in docker/host
if [ "$USE_HOST_FOAM" = true ] && command -v checkMesh >/dev/null 2>&1; then
  (cd "$HOST_RUN_DIR" && checkMesh -allGeometry -allTopology -case .)
else
  docker run --rm -u "$(id -u):$(id -g)" \
    -v "${HOST_RUN_DIR}:/home/openfoam/case" -w /home/openfoam/case \
    "${OPENFOAM_IMAGE}" \
    bash -lc "checkMesh -allGeometry -allTopology -case /home/openfoam/case"
fi

echo "[RUN] pimpleFoam"
_run_foam_cmd pimpleFoam

# marker file
touch "${HOST_RUN_DIR}/test.foam"

# Print run directory (caller should capture stdout)
echo "${HOST_RUN_DIR}"