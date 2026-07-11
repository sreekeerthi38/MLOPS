# MLOps Assignment 01 — Report Outline (target: 10 pages)

Fill each section AFTER you've actually run the pipeline on your machine —
do not write results you haven't produced. The grader is explicitly told to
flag suspiciously identical implementations.

1. **Project Overview** (0.5 pg) — problem statement, dataset, your specific
   modeling choices and why.
2. **EDA Findings** (1.5 pg) — insert your histograms, correlation heatmap,
   class balance plot, missing-value table. Write 3-4 sentences of actual
   interpretation per plot, not captions.
3. **Feature Engineering & Model Comparison** (1.5 pg) — table: model, best
   hyperparameters, accuracy/precision/recall/F1/ROC-AUC, cross-val scores.
   State which model you selected and why.
4. **Experiment Tracking (MLflow)** (1 pg) — screenshots of your MLflow UI
   run comparison table and one run's artifact view.
5. **Model Packaging** (0.5 pg) — serialization format used, requirements.txt
   approach, how the preprocessing pipeline guarantees reproducibility.
6. **Architecture Diagram** (1 pg) — draw: data -> preprocessing -> training
   -> MLflow -> Docker image -> CI/CD -> K8s -> monitoring. (Use draw.io /
   excalidraw / Mermaid — don't skip this, it's an explicit deliverable.)
7. **CI/CD Pipeline** (1 pg) — screenshot of a green GitHub Actions run,
   explain each stage.
8. **Containerization & Deployment** (1.5 pg) — Dockerfile walkthrough,
   `docker build`/`docker run` terminal screenshot, kubectl get pods/svc
   screenshot, and a successful `curl` against the deployed /predict.
9. **Monitoring & Logging** (1 pg) — screenshot of API request logs and/or
   Prometheus/Grafana dashboard; note what you'd alert on in production.
10. **Setup Instructions** (0.5 pg) — copy from README.md.
11. **Repository Link** (0.5 pg).

**Do not submit this outline as your report.** Every screenshot must come
from a run you performed.
