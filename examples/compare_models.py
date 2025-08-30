"""
Compare performance of different models with BorutaShap
"""

import sys
import time
sys.path.insert(0, '../src')

from BorutaShap import BorutaShap
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd

# Generate data
X, y = make_classification(n_samples=5000, n_features=50, n_informative=20, random_state=42)
X = pd.DataFrame(X, columns=[f'f{i}' for i in range(50)])

models = {
    'LightGBM': LGBMClassifier(n_estimators=30, max_depth=5, verbose=-1),
    'XGBoost': XGBClassifier(n_estimators=30, max_depth=5, eval_metric='logloss'),
    'RandomForest': RandomForestClassifier(n_estimators=30, max_depth=5)
}

print("Model Performance Comparison")
print("="*50)

for name, model in models.items():
    print(f"\nTesting {name}...")
    
    fs = BorutaShap(model=model, importance_measure='shap', classification=True)
    
    start = time.time()
    fs.fit(X=X, y=y, n_trials=5, random_state=42, verbose=False)
    elapsed = time.time() - start
    
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Accepted: {len(fs.accepted)}")
    print(f"  Rejected: {len(fs.rejected)}")
