#!/usr/bin/env bash
# Run swakless test case in Results_Date_Time folder (macOS + Docker)

set -e
IMAGE="opencfd/openfoam-default:2506"

# Base case (assumes you run this from .../2D_Reactor/)
baseDir="$PWD/swakless-1"

# Results directory with timestamp
timestamp=$(date +"%Y-%m-%d_%H-%M")
runDir="$PWD/Results/Results_$timestamp"
mkdir -p "$runDir"

echo "Copying from $baseDir to $runDir"
cp -R "$baseDir/"* "$runDir/"

# Run simulation INSIDE Docker, with your results dir mounted at /home/openfoam
docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v "$runDir":/home/openfoam \
  -w /home/openfoam \
  "$IMAGE" \
  bash -lc 'blockMesh && pimpleFoam'

# Ensure test.foam exists for ParaView
touch "$runDir/test.foam"

# Open results in ParaView (macOS)
  open -a "ParaView-6.0.0.app" "$runDir/test.foam"

