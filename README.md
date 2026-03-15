# 🚀 MLOps Pipeline Starter — Production Template

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.12-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/DVC-3.x-945DD6?style=flat-square&logo=dvc&logoColor=white)](https://dvc.org)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-326CE5?style=flat-square&logo=kubernetes&logoColor=white)](https://kubernetes.io)
[![CI](https://img.shields.io/github/actions/workflow/status/harshadkhetpal/mlops-pipeline-starter/ci.yml?style=flat-square&label=CI)](https://github.com/harshadkhetpal/mlops-pipeline-starter/actions)

> A complete, battle-tested MLOps starter template. Train → Evaluate → Register → Deploy → Monitor. Works locally and on Kubernetes.

## ✨ Stack
- **Training**: scikit-learn / XGBoost with MLflow experiment tracking
- **Versioning**: DVC for data and model artifacts (S3 remote)
- **Serving**: FastAPI inference server + Docker
- **Orchestration**: Kubernetes + Helm chart
- **CI/CD**: GitHub Actions → build → test → deploy

## 🚀 Quick Start
```bash
git clone https://github.com/harshadkhetpal/mlops-pipeline-starter
cd mlops-pipeline-starter
pip install -r requirements.txt
python train.py --experiment my-first-run
python serve.py  # http://localhost:8080/predict
```
