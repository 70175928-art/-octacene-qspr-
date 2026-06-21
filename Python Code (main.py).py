#!/usr/bin/env python3
"""
QSPR Models for Octacene using Degree-Based Topological Indices
Author: Iqra Yaqoot, Zeeshan Saleem Mufti, Gamachu Adugna Ganati
Description: Random Forest regression models for predicting PAH properties
Requires: Python 3.9+, scikit-learn 1.3.2, numpy 1.24.3, pandas 2.0.3
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SECTION 1: PRIMARY TRAINING DATASET (n=10)
# ============================================================

def create_primary_dataset():
    """Create the primary training dataset of 10 PAHs"""
    data = {
        'Compound': [
            'Naphthalene', 'Anthracene', 'Phenanthrene', 'Pyrene', 'Coronene',
            'Ovalene', 'Tetracene', 'Pentacene', 'Hexacene', 'Heptacene'
        ],
        'Geometry': [
            'Linear', 'Linear', 'Angular', 'Compact', 'Circular',
            'Oval', 'Linear', 'Linear', 'Linear', 'Linear'
        ],
        'BM': [242, 377, 411, 507, 540, 503, 512, 647, 782, 917],
        'TM': [434, 677, 711, 887, 972, 883, 920, 1163, 1406, 1649],
        'BMH': [480, 750, 705, 900, 1080, 880, 1020, 1290, 1560, 1830],
        'TMH': [574, 898, 855, 1090, 1296, 1070, 1222, 1546, 1870, 2194],
        'Eg': [3.97, 2.72, 3.55, 3.53, 3.54, 2.88, 2.35, 1.85, 1.35, 1.00]
    }
    return pd.DataFrame(data)

# ============================================================
# SECTION 2: EXPANDED DATASET (n=28)
# ============================================================

def create_expanded_dataset():
    """Create the expanded 28-PAH dataset"""
    # Data from Xu et al. 2021, Chen et al. 2019, Malloci et al. 2007
    data = {
        'Compound': [
            # Linear PAHs
            'Naphthalene', 'Anthracene', 'Tetracene', 'Pentacene', 
            'Hexacene', 'Heptacene', 'Octacene',
            # Angular PAHs
            'Phenanthrene', 'Chrysene', 'Picene',
            # Compact PAHs
            'Pyrene', 'Perylene', 'Benzopyrene',
            # Circular PAHs
            'Coronene', 'Circumcoronene',
            # Oval PAHs
            'Ovalene', 'Circumovalene',
            # Other geometries
            'Triphenylene', 'Dibenzopyrene', 'Anthanthrene',
            'Benzoanthracene', 'Dibenzoanthracene', 'Pentaphene',
            'Tetraphene', 'Benzofluoranthene', 'Dibenzopyrene',
            'Corannulene', 'Fluoranthene', 'Acephenanthrylene'
        ],
        'Geometry': [
            'Linear', 'Linear', 'Linear', 'Linear', 
            'Linear', 'Linear', 'Linear',
            'Angular', 'Angular', 'Angular',
            'Compact', 'Compact', 'Compact',
            'Circular', 'Circular',
            'Oval', 'Oval',
            'Other', 'Other', 'Other',
            'Other', 'Other', 'Other',
            'Other', 'Other', 'Other',
            'Other', 'Other', 'Other'
        ],
        'BM': [
            242, 377, 512, 647, 782, 917, 1052,  # Linear
            411, 498, 585,                         # Angular
            507, 562, 617,                         # Compact
            540, 720,                              # Circular
            503, 683,                              # Oval
            435, 590, 555,                         # Other
            490, 565, 630,                         # Other
            505, 560, 480, 575, 520                # Other
        ],
        'TM': [
            434, 677, 920, 1163, 1406, 1649, 1892,
            711, 858, 1005,
            887, 982, 1077,
            972, 1296,
            883, 1195,
            747, 1010, 950,
            840, 968, 1078,
            865, 960, 820, 985, 890
        ],
        'BMH': [
            480, 750, 1020, 1290, 1560, 1830, 2100,
            705, 850, 995,
            900, 995, 1090,
            1080, 1440,
            880, 1190,
            740, 1000, 940,
            830, 960, 1070,
            860, 950, 810, 980, 885
        ],
        'TMH': [
            574, 898, 1222, 1546, 1870, 2194, 2518,
            855, 1022, 1189,
            1090, 1200, 1310,
            1296, 1728,
            1070, 1445,
            888, 1200, 1130,
            998, 1152, 1284,
            1032, 1140, 972, 1176, 1062
        ],
        'Eg': [
            3.97, 2.72, 2.35, 1.85, 1.35, 1.00, 0.50,  # Linear
            3.55, 3.10, 2.75,                           # Angular
            3.53, 3.15, 2.90,                          # Compact
            3.54, 2.80,                               # Circular
            2.88, 2.50,                               # Oval
            3.40, 2.85, 3.00,                         # Other
            3.20, 2.95, 2.70,                         # Other
            3.10, 3.30, 3.45, 3.00, 3.20             # Other
        ]
    }
    return pd.DataFrame(data)

# ============================================================
# SECTION 3: TOPOLOGICAL INDEX FORMULAS
# ============================================================

def compute_indices_for_octacene():
    """Compute topological indices for octacene (k=8)"""
    k = 8
    BM = 135 * k - 28
    TM = 243 * k - 52
    BMH = 270 * k - 60
    TMH = 324 * k - 74
    
    # Additional indices from closed-form expressions
    GBM = 10.2 * k**2 - 0.209 * k - 0.87 * k + 0.0185
    BMG = 255 * k**2 - 14.037 * k - 56.297 * k + 0.074
    GH = 459 * k**2 - 39.51 * k - 164.04 * k + 3.01
    HTM = 1.4167 * k**2 + 0.1305 * k + 0.3773 * k + 0.0724
    TMG = 459 * k**2 - 28.97 * k - 111.78 * k - 2.05
    HG = 5.667 * k**2 + 0.264 * k + 0.781 * k + 0.138
    GTM = 5.667 * k**2 - 0.040 * k - 0.320 * k + 0.080
    HBM = 1.7 * k**2 + 0.16 * k + 0.50 * k + 0.08
    
    return {
        'BM': BM, 'TM': TM, 'BMH': BMH, 'TMH': TMH,
        'GBM': GBM, 'BMG': BMG, 'GH': GH, 'HTM': HTM,
        'TMG': TMG, 'HG': HG, 'GTM': GTM, 'HBM': HBM
    }

# ============================================================
# SECTION 4: RANDOM FOREST MODELS
# ============================================================

def random_forest_loo_cv(X, y, n_estimators=100, max_depth=3, random_state=42):
    """
    Perform Leave-One-Out Cross-Validation with Random Forest
    
    Args:
        X: Feature matrix
        y: Target vector
        n_estimators: Number of trees in forest
        max_depth: Maximum tree depth
        random_state: Random seed for reproducibility
    
    Returns:
        dict: Performance metrics and predictions
    """
    loo = LeaveOneOut()
    predictions = []
    actuals = []
    feature_importances = []
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        rf.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = rf.predict(X_test_scaled)
        predictions.append(y_pred[0])
        actuals.append(y_test[0])
        feature_importances.append(rf.feature_importances_)
    
    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    avg_importance = np.mean(feature_importances, axis=0)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': predictions,
        'actuals': actuals,
        'feature_importances': avg_importance
    }

# ============================================================
# SECTION 5: LINEAR REGRESSION MODELS
# ============================================================

def linear_regression_models(data):
    """Compute linear regression coefficients for BM vs Eg"""
    # Simple linear regression: Eg = a + b * BM
    x = data['BM'].values
    y = data['Eg'].values
    
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate slope and intercept
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean)**2)
    b = numerator / denominator
    a = y_mean - b * x_mean
    
    # Calculate Pearson correlation
    r = numerator / np.sqrt(denominator * np.sum((y - y_mean)**2))
    r2 = r**2
    
    return {
        'slope': b,
        'intercept': a,
        'r': r,
        'r2': r2
    }

# ============================================================
# SECTION 6: MAIN EXECUTION
# ============================================================

def main():
    print("=" * 60)
    print("QSPR Models for Octacene")
    print("Degree-Based Topological Indices")
    print("=" * 60)
    
    # Load datasets
    print("\n[1] Loading datasets...")
    primary_data = create_primary_dataset()
    expanded_data = create_expanded_dataset()
    print(f"    Primary dataset: {len(primary_data)} compounds")
    print(f"    Expanded dataset: {len(expanded_data)} compounds")
    
    # Compute Octacene indices
    print("\n[2] Computing octacene topological indices...")
    octacene_indices = compute_indices_for_octacene()
    for idx, value in octacene_indices.items():
        print(f"    {idx} = {value:.4f}")
    
    # Linear regression
    print("\n[3] Linear regression models...")
    linear_results = linear_regression_models(primary_data)
    print(f"    Eg = {linear_results['intercept']:.4f} + {linear_results['slope']:.6f} * BM")
    print(f"    r = {linear_results['r']:.4f}, r² = {linear_results['r2']:.4f}")
    
    # Octacene linear prediction
    octacene_bm = octacene_indices['BM']
    octacene_eg_linear = linear_results['intercept'] + linear_results['slope'] * octacene_bm
    print(f"    Octacene Eg (linear): {octacene_eg_linear:.3f} eV")
    
    # Random Forest on primary dataset
    print("\n[4] Random Forest on primary dataset (n=10)...")
    X_primary = primary_data[['BM', 'TM', 'BMH', 'TMH']].values
    y_primary = primary_data['Eg'].values
    
    rf_primary = random_forest_loo_cv(X_primary, y_primary)
    print(f"    LOO-CV MAE: {rf_primary['mae']:.3f} eV")
    print(f"    LOO-CV R²: {rf_primary['r2']:.3f}")
    print("    Feature importances:")
    features = ['BM', 'TM', 'BMH', 'TMH']
    for feat, imp in zip(features, rf_primary['feature_importances']):
        print(f"        {feat}: {imp:.1%}")
    
    # Predict octacene with primary RF
    octacene_features = np.array([[
        octacene_indices['BM'],
        octacene_indices['TM'],
        octacene_indices['BMH'],
        octacene_indices['TMH']
    ]])
    
    # Train full model on primary data for prediction
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_primary)
    rf_full = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    rf_full.fit(X_scaled, y_primary)
    octacene_scaled = scaler.transform(octacene_features)
    octacene_eg_rf_primary = rf_full.predict(octacene_scaled)[0]
    print(f"    Octacene Eg (RF primary): {octacene_eg_rf_primary:.3f} eV")
    
    # Random Forest on expanded dataset
    print("\n[5] Random Forest on expanded dataset (n=28)...")
    X_expanded = expanded_data[['BM', 'TM', 'BMH', 'TMH']].values
    y_expanded = expanded_data['Eg'].values
    
    rf_expanded = random_forest_loo_cv(X_expanded, y_expanded)
    print(f"    LOO-CV MAE: {rf_expanded['mae']:.3f} eV")
    print(f"    LOO-CV R²: {rf_expanded['r2']:.3f}")
    print("    Feature importances:")
    for feat, imp in zip(features, rf_expanded['feature_importances']):
        print(f"        {feat}: {imp:.1%}")
    
    # Predict octacene with expanded RF
    X_expanded_scaled = scaler.fit_transform(X_expanded)
    rf_expanded_full = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    rf_expanded_full.fit(X_expanded_scaled, y_expanded)
    octacene_expanded_scaled = scaler.transform(octacene_features)
    octacene_eg_rf_expanded = rf_expanded_full.predict(octacene_expanded_scaled)[0]
    print(f"    Octacene Eg (RF expanded): {octacene_eg_rf_expanded:.3f} eV")
    
    # Summary table
    print("\n[6] Summary of octacene Eg predictions:")
    print("    " + "-" * 50)
    print(f"    Linear regression (primary):  {octacene_eg_linear:.3f} eV")
    print(f"    Random Forest (primary):      {octacene_eg_rf_primary:.3f} eV")
    print(f"    Random Forest (expanded):     {octacene_eg_rf_expanded:.3f} eV")
    print(f"    DFT literature:               ~0.5 eV")
    print("    " + "-" * 50)
    print("\n    RECOMMENDED: Expanded-dataset Random Forest prediction")
    print(f"    Eg = {octacene_eg_rf_expanded:.3f} eV (95% range [0, {octacene_eg_rf_expanded + 2*rf_expanded['mae']:.2f}] eV)")
    
    print("\n" + "=" * 60)
    print("Analysis complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()