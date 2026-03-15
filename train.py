# Author: Harshad Khetpal <harshadkhetpal@gmail.com>
"""MLflow training script for RandomForestClassifier on wine dataset."""

import argparse
import pickle
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    """Train RandomForest model and log to MLflow."""
    
    # Load dataset
    wine = load_wine()
    X, y = wine.data, wine.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    
    # Start MLflow run
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("test_size", args.test_size)
        
        # Train model
        logger.info("Training RandomForestClassifier...")
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.pkl"
        
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML model with MLflow")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=10, help="Max tree depth")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--output_dir", type=str, default="./models", help="Output directory")
    
    args = parser.parse_args()
    main(args)
