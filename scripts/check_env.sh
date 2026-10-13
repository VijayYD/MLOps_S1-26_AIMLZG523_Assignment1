#!/usr/bin/env bash
# Preflight environment check for the Heart Disease MLOps project.
# Must run unmodified on macOS (development) and Linux (BITS Prayogshala lab).
set -uo pipefail

PASS=0
FAIL=0
WARN=0

ok()   { printf "  [OK]   %s\n" "$1"; PASS=$((PASS+1)); }
bad()  { printf "  [FAIL] %s\n" "$1"; FAIL=$((FAIL+1)); }
warn() { printf "  [WARN] %s\n" "$1"; WARN=$((WARN+1)); }

section() { printf "\n=== %s ===\n" "$1"; }

OS="$(uname -s)"
ARCH="$(uname -m)"

section "Operating system"
printf "  OS: %s (%s)\n" "$OS" "$ARCH"

# ---------------------------------------------------------------------------
# Hard requirements: needed to do anything at all in this repo.
# ---------------------------------------------------------------------------
section "Hard requirements"

EXPECTED_PY="3.13"
[ "$OS" = "Linux" ] && EXPECTED_PY="3.12"

if command -v python3 >/dev/null 2>&1; then
  PY_VER="$(python3 --version 2>&1)"
  ok "python3 on PATH: $PY_VER ($(command -v python3))"
  if printf "%s" "$PY_VER" | grep -q " ${EXPECTED_PY}\."; then
    ok "PATH python3 matches the pinned ${EXPECTED_PY}.x for this OS"
  else
    warn "PATH python3 is not ${EXPECTED_PY}.x (pinned for this OS). This is fine if the project venv uses a pinned interpreter (e.g. via pyenv) instead of PATH python3 -- just don't rely on PATH python3 for setup_venv.sh"
  fi
else
  bad "python3 not found on PATH"
fi

if command -v pip3 >/dev/null 2>&1; then
  ok "pip3 found: $(pip3 --version 2>&1)"
elif python3 -m pip --version >/dev/null 2>&1; then
  ok "pip found via 'python3 -m pip': $(python3 -m pip --version 2>&1)"
else
  bad "pip not found (neither pip3 nor 'python3 -m pip')"
fi

if command -v git >/dev/null 2>&1; then
  ok "git found: $(git --version 2>&1)"
else
  bad "git not found on PATH"
fi

if command -v curl >/dev/null 2>&1; then
  ok "curl found: $(curl --version 2>&1 | head -n1)"
else
  bad "curl not found on PATH"
fi

# ---------------------------------------------------------------------------
# Needed later (Docker build/run, Kubernetes deploy, monitoring). Missing
# tools here are reported, not treated as fatal at Phase 0 -- the project
# plan explicitly asks before installing anything.
# ---------------------------------------------------------------------------
section "Docker"

if command -v docker >/dev/null 2>&1; then
  ok "docker CLI found: $(docker --version 2>&1)"
  if docker info >/dev/null 2>&1; then
    ok "docker daemon reachable without sudo"
  else
    warn "docker CLI present but daemon not reachable (is Docker Desktop running?)"
  fi
  if docker compose version >/dev/null 2>&1; then
    ok "docker compose plugin found: $(docker compose version 2>&1)"
  else
    warn "docker compose plugin not found (docker-compose.yml step will need it)"
  fi
else
  warn "docker not found on PATH (needed from Phase 7 onward)"
fi

section "Kubernetes tooling"

EXPECTED_MINIKUBE="v1.35.0"

if command -v minikube >/dev/null 2>&1; then
  MK_VER="$(minikube version --short 2>&1 | head -n1 | tr -d '\r')"
  ok "minikube found: $MK_VER"
  if [ "$MK_VER" = "$EXPECTED_MINIKUBE" ]; then
    ok "minikube matches the pinned version ($EXPECTED_MINIKUBE)"
  else
    warn "minikube is $MK_VER, not the pinned $EXPECTED_MINIKUBE -- install the pinned release for Mac/lab parity"
  fi
else
  warn "minikube not found on PATH (needed from Phase 7 onward)"
fi

if command -v kubectl >/dev/null 2>&1; then
  KUBECTL_VER="$(kubectl version --client -o json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["clientVersion"]["gitVersion"])' 2>/dev/null)"
  if [ -z "$KUBECTL_VER" ]; then
    KUBECTL_VER="$(kubectl version --client 2>/dev/null | head -n1)"
  fi
  ok "kubectl found: $KUBECTL_VER"
elif command -v minikube >/dev/null 2>&1 && minikube kubectl -- version --client >/dev/null 2>&1; then
  warn "kubectl not found on PATH; falling back to 'minikube kubectl --' (works)"
else
  warn "kubectl not found on PATH and 'minikube kubectl --' fallback unavailable (needed from Phase 7 onward)"
fi

if command -v make >/dev/null 2>&1; then
  ok "make found: $(make --version 2>&1 | head -n1)"
else
  warn "make not found on PATH (scripts/*.sh equivalents will be used instead)"
fi

# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------
section "CPU and memory"

if [ "$OS" = "Darwin" ]; then
  CPUS="$(sysctl -n hw.ncpu 2>/dev/null || echo unknown)"
  MEM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
  MEM_GB=$(( MEM_BYTES / 1024 / 1024 / 1024 ))
  printf "  CPUs: %s\n" "$CPUS"
  printf "  RAM:  %s GB\n" "$MEM_GB"
elif [ "$OS" = "Linux" ]; then
  CPUS="$(nproc 2>/dev/null || echo unknown)"
  MEM_KB="$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
  MEM_GB=$(( MEM_KB / 1024 / 1024 ))
  printf "  CPUs: %s\n" "$CPUS"
  printf "  RAM:  %s GB\n" "$MEM_GB"
else
  warn "Unrecognized OS '$OS'; skipping CPU/RAM report"
fi

# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------
section "Port availability (8000 API, 5001 MLflow, 9090 Prometheus, 3000 Grafana)"

port_in_use() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -i ":${port}" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  elif command -v ss >/dev/null 2>&1; then
    ss -ltn "sport = :${port}" 2>/dev/null | grep -q ":${port}"
    return $?
  elif command -v netstat >/dev/null 2>&1; then
    netstat -an 2>/dev/null | grep -q "[.:]${port}.*LISTEN"
    return $?
  else
    return 1
  fi
}

for port in 8000 5001 9090 3000; do
  if port_in_use "$port"; then
    warn "port $port is currently in USE"
  else
    ok "port $port is free"
  fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section "Summary"
printf "  Passed: %d  Warnings: %d  Failed: %d\n" "$PASS" "$WARN" "$FAIL"

if [ "$FAIL" -gt 0 ]; then
  printf "\nHard requirement(s) missing. Install the tool(s) marked [FAIL] above and re-run this script.\n"
  exit 1
fi

printf "\nAll hard requirements satisfied. Review [WARN] items above before later phases.\n"
exit 0
