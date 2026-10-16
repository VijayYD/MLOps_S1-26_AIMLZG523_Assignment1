# Screenshot checklist

Numbered, with the exact target filename `reports/build_report.py` looks
for under `screenshots/`. Take the screenshot, save it at that exact path,
re-run `make report`, and it's embedded automatically — otherwise the
report shows a clearly labelled placeholder box instead. Items without a
target filename are already generated figures/logs, not manual
screenshots.

| # | Screenshot | Target filename | Command / where to look | Machine |
|---|---|---|---|---|
| 1 | EDA figures | *(already in `reports/figures/01_*`–`06_*.png`)* | `notebooks/01_eda.ipynb` | n/a |
| 2 | Modelling figures | *(already in `reports/figures/07_*`–`13_*.png`)* | `notebooks/02_modelling.ipynb` | n/a |
| 3 | MLflow runs list | `screenshots/03_mlflow_runs_list.png` | `make mlflow-ui` → http://127.0.0.1:5001 → `heart-disease-4site` experiment | Mac |
| 4 | MLflow run detail | `screenshots/04_mlflow_run_detail.png` | Click the selected model's child run → params/metrics/artifacts, and the registered model's `champion` alias | Mac |
| 5 | GitHub Actions — all jobs green | `screenshots/05_gha_all_green.png` | Repo → Actions tab, after a push, showing `lint`/`test`(×2)/`train`/`docker` all passing | n/a (after a push) |
| 6 | `pytest` coverage summary | `screenshots/06_pytest_coverage.png` | `make test` terminal output, or `htmlcov/index.html` | Mac and/or lab |
| 7 | Docker build + container smoke test | `screenshots/07_docker_smoke.png` | `make docker-smoke` (full text log already saved to `screenshots/logs/run_container_smoke_*.log`) | Mac and/or lab |
| 8 | Non-root user + healthy `HEALTHCHECK` | `screenshots/08_docker_inspect.png` | `docker inspect --format='{{json .State.Health}}' <container>` and `docker exec <container> whoami` | Mac and/or lab |
| 9 | `kubectl get pods,svc,ingress,hpa`, plus `/predict` via Ingress and LoadBalancer | `screenshots/09_kubectl_get.png` | `make k8s-verify` (full text log already saved to `screenshots/logs/k8s_verify_*.log`) — one screenshot of this single run's output covers both the resource listing and both `/predict` paths, so there is no separate #10 | Mac and/or lab |
| 11 | Swagger / OpenAPI docs | `screenshots/11_swagger_docs.png` | `http://127.0.0.1:8000/docs` (port-forward the API Service) | Mac and/or lab |
| 12 | Prometheus targets page | `screenshots/12_prometheus_targets.png` | Port-forward `svc/prometheus` → `http://127.0.0.1:9090/targets` | Mac and/or lab |
| 13 | Grafana dashboard with live data | `screenshots/13_grafana_dashboard.png` | After `make load-test`, port-forward `svc/grafana` → `http://127.0.0.1:3000` | Mac and/or lab |
| 14 | Lab terminal runs | `screenshots/14_lab_terminal.png` | `check_env.sh`/`setup_venv.sh`/`make test`/`make k8s-deploy` run for real on the Prayogshala machine | Lab |
