# Changelog

## [1.1.0] - 2024

### Added
- Python 3.12 support
- Comprehensive benchmarking suite
- Performance comparison examples

### Fixed
- NumPy 2.0 compatibility (np.NaN → np.nan)
- SciPy 1.11+ compatibility (binom_test → binomtest)
- RandomForest SHAP 3D array handling
- RandomForest Gini importance check before fitting
- Missing imports (inspect, defaultdict)

### Performance
- Identified LightGBM as optimal for SHAP performance
- Documented model selection guidelines
- Added dataset size recommendations

## [1.0.0] - Original
- Initial release by Eoghan Keany
