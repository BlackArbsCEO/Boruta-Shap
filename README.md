# BorutaShap - Modern Fork

A modernized fork of [BorutaShap](https://github.com/Ekeany/Boruta-Shap) that works with current versions of NumPy, SciPy, and scikit-learn. This fork includes performance improvements and bug fixes for SHAP-based feature selection.

## Key Improvements

### 🔧 Compatibility Fixes
- **NumPy 2.0+ support**: Fixed deprecated `np.NaN` → `np.nan`
- **SciPy 1.11+ support**: Updated `binom_test` → `binomtest` with backward compatibility
- **Python 3.12 support**: Fully tested with Python 3.9 through 3.12

### 🐛 Bug Fixes
- **RandomForest + SHAP**: Fixed 3D array handling and indexing issues
- **RandomForest + Gini**: Fixed premature feature_importances_ check
- **Missing imports**: Added required imports (inspect, defaultdict)

### ⚡ Performance Insights
Based on extensive benchmarking:
- **LightGBM**: Best overall performer (0.6s avg SHAP time, F1=0.875)
- **XGBoost**: Good balance (1.6s avg SHAP time, F1=0.868)
- **RandomForest**: Best F1 on small datasets (F1=0.935 @ 1k samples)
- **GradientBoosting**: Highest accuracy but slow (13s avg SHAP time)

## Installation

```bash
# Clone this fork
git clone https://github.com/BlackArbsCEO/Boruta-Shap.git
cd Boruta-Shap

# Install in development mode
pip install -e .

# Or install directly from GitHub
pip install git+https://github.com/BlackArbsCEO/Boruta-Shap.git
```

## Requirements

```txt
numpy>=1.24.0
pandas>=1.5.0
scipy>=1.10.0
scikit-learn>=1.2.0
shap>=0.41.0
tqdm>=4.65.0
lightgbm>=3.3.0  # Optional but recommended
xgboost>=1.7.0   # Optional
```

## Quick Start

```python
from BorutaShap import BorutaShap
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
import pandas as pd

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10)
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])

# Initialize with LightGBM (recommended for speed)
model = LGBMClassifier(n_estimators=50, max_depth=5, verbose=-1)

# Run BorutaShap
fs = BorutaShap(
    model=model,
    importance_measure='shap',  # or 'gini' for tree-based models
    classification=True
)

fs.fit(X=X, y=y, n_trials=100, random_state=42)

# Get results
print(f"Accepted features: {fs.accepted}")
print(f"Rejected features: {fs.rejected}")
print(f"Tentative features: {fs.tentative}")
```

## Performance Recommendations

### Model Selection Guide

| Use Case | Recommended Model | F1 Score | SHAP Speed |
|----------|------------------|----------|------------|
| Small data (<5k samples) | RandomForest | 0.935 | 0.15s |
| Medium data (5-50k) | LightGBM | 0.90 | 0.5-2s |
| Large data (>50k) | LightGBM | 0.89 | 2-5s |
| Best accuracy | GradientBoosting | 0.91 | 10-50s |
| Production/speed critical | LightGBM | 0.88 | <2s |

### Dataset Size Impact

- **Samples**: More samples → better F1 (all models improve 5-9%)
- **Features**: More features → worse F1 (especially RandomForest: -15% from 10→200 features)
- **Sweet spot**: 5-10k samples with ≤50 features

### Feature Importance Methods

- **SHAP**: More accurate but ~11x slower than Gini
- **Gini**: Fast but only for tree-based models (not XGBoost)
- **Recommendation**: Use SHAP for final models, Gini for exploration

## Supported Models

✅ **Fully Supported:**
- LightGBM (fastest SHAP)
- XGBoost (SHAP only)
- RandomForest (both SHAP and Gini)
- ExtraTrees (both SHAP and Gini)
- GradientBoosting (both SHAP and Gini)

❌ **Not Supported:**
- BaggingClassifier (SHAP TreeExplainer incompatible)
- SVM, Neural Networks (no tree structure)

## Testing

```bash
# Run basic test
python examples/test_basic.py

# Run performance comparison
python examples/compare_models.py

# Test with your data
python examples/test_custom.py --data your_data.csv
```

## Changes from Original

1. **Fixed NumPy 2.0 compatibility** (src/BorutaShap.py:L384-394)
2. **Fixed SciPy binomial test import** (src/BorutaShap.py:L8-13)
3. **Fixed RandomForest SHAP 3D array handling** (src/BorutaShap.py:L250-260)
4. **Fixed RandomForest Gini importance check** (src/BorutaShap.py:L150-155)
5. **Added Python 3.12 support** (setup.py)
6. **Added comprehensive benchmarks** (examples/benchmark.py)

## Citation

If you use this fork, please cite both the original and this fork:

```bibtex
# Original BorutaShap
@software{boruta_shap,
  author = {Eoghan Keany},
  title = {BorutaShap: A wrapper feature selection method using Boruta and SHAP},
  url = {https://github.com/Ekeany/Boruta-Shap},
  year = {2020}
}

# This fork
@software{boruta_shap_modern,
  author = {BlackArbsCEO},
  title = {BorutaShap Modern Fork: Compatible with NumPy 2.0+},
  url = {https://github.com/BlackArbsCEO/Boruta-Shap},
  year = {2025}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Run tests with Python 3.9+ 
4. Submit a pull request

## License

MIT License (same as original)

## Acknowledgments

- Original author: [Eoghan Keany](https://github.com/Ekeany)
- SHAP library: [lundberg/shap](https://github.com/slundberg/shap)
- Boruta algorithm: [Boruta R package](https://github.com/mbq/Boruta)
