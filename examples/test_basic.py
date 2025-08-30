"""
Basic test of BorutaShap with different models
"""

import sys
sys.path.insert(0, '../src')

from BorutaShap import BorutaShap
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
import pandas as pd

# Generate sample data
print("Generating sample data...")
X, y = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=10,
    n_redundant=5,
    random_state=42
)
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])

# Test with LightGBM
print("\nTesting with LightGBM...")
model = LGBMClassifier(n_estimators=50, max_depth=5, verbose=-1)

fs = BorutaShap(
    model=model,
    importance_measure='shap',
    classification=True
)

fs.fit(X=X, y=y, n_trials=10, random_state=42, verbose=False)

print(f"Accepted features: {fs.accepted}")
print(f"Rejected features: {fs.rejected}")
print(f"Tentative features: {fs.tentative}")
print("\n✓ Test completed successfully!")
