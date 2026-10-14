# Screenshot checklist

**Status:** running list, growing with each phase; final numbering and any
remaining gaps are confirmed in Phase 9. `n/a` in the Machine column means
either machine works identically.

| # | Screenshot | Command / where to look | Machine |
|---|---|---|---|
| 1 | EDA figures (missingness heatmap, class balance, distributions, correlation heatmaps, site-shift boxplots) | Already rendered inline in `notebooks/01_eda.ipynb`; files in `reports/figures/01_*.png`–`06_*.png` | n/a |
| 2 | Modelling figures (ROC/PR/calibration curves, confusion matrix, feature importance, leave-one-site-out bars) | `notebooks/02_modelling.ipynb`; files in `reports/figures/07_*.png`–`13_*.png` | n/a |
| 3 | MLflow runs list | `make mlflow-ui` → http://127.0.0.1:5001 → the `heart-disease-4site` experiment, showing the parent run + 3 nested child runs | Mac (and lab, same command) |
| 4 | MLflow run detail | Click the selected model's child run (e.g. `logreg`) → params/metrics/artifacts tabs, and the registered model with the `champion` alias | Mac |
| 5 | GitHub Actions run — all jobs green | `https://github.com/VijayYD/MLOps_S1-26_AIMLZG523_Assignment1/actions` after a push, showing `lint`/`test`(×2)/`train`/`docker` all passing, plus the uploaded artifacts list | n/a (after a push) |
| 6 | `pytest` coverage summary | Terminal output of `make test` (or the `htmlcov/index.html` report) showing 90%+ coverage | Mac and/or lab |
| 7 | Docker build + container smoke test | Terminal output of `make docker-smoke` (also saved automatically to `screenshots/logs/run_container_smoke_*.log`) | Mac and/or lab |
| 8 | `docker ps` / `docker inspect` showing non-root user + healthy `HEALTHCHECK` | `docker ps` then `docker inspect --format='{{json .State.Health}}' <container>` and `docker exec <container> whoami` | Mac and/or lab |
| 9 | `kubectl get pods,svc,ingress,hpa -n heart-mlops` | Terminal output of `make k8s-verify` (also saved to `screenshots/logs/k8s_verify_*.log`) | Mac and/or lab |
| 10 | `/predict` via Ingress and via LoadBalancer | Same `k8s_verify.sh` output, both response bodies | Mac and/or lab |
| 11 | Swagger / OpenAPI docs | `http://127.0.0.1:8000/docs` (port-forward the API Service or use `minikube service heart-api-lb --url`) — show the `/predict` schema with its field examples | Mac and/or lab |
| 12 | Prometheus targets page | Port-forward `svc/prometheus` → `http://127.0.0.1:9090/targets` → both `heart-api` pods showing `UP` | Mac and/or lab |
| 13 | Grafana dashboard with live data | After `make load-test`, port-forward `svc/grafana` → `http://127.0.0.1:3000` → the auto-provisioned "Heart Disease API" dashboard | Mac and/or lab |
| 14 | Lab terminal runs | `scripts/check_env.sh`, `scripts/setup_venv.sh`, `make test`, `make k8s-deploy`/`k8s-verify` run for real on the Prayogshala lab machine | Lab |
