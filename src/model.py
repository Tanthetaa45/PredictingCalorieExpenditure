import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping
import lightgbm as lgbm

# Assuming X, y, test_df are already defined
FOLDS = 2
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

cat_preds = np.zeros((FOLDS, len(test_df)))
xgb_preds = np.zeros((FOLDS, len(test_df)))
lgbm_preds = np.zeros((FOLDS, len(test_df)))

oof_cat = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_lgbm = np.zeros(len(X))

models = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\n{'#'*15} Fold {fold+1} {'#'*15}")
    
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y[valid_idx]

    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=3500,
        learning_rate=0.02,
        depth=12,
        l2_leaf_reg=3,
        verbose=1000,
        early_stopping_rounds=200,
        loss_function='RMSE',
        eval_metric='RMSE',
        task_type='CPU'
    )
    cat_model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True, cat_features=[0])  # Update cat_features as needed

    # XGBoost
    xgb_model = XGBRegressor(
        max_depth=10,
        n_estimators=2000,
        learning_rate=0.01,
        subsample=0.9,
        colsample_bytree=0.75,
        gamma=0.01,
        enable_categorical=True,
        tree_method="hist",
        device="cpu"
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=100, verbose=0)

    # LightGBM
    lgbm_model = LGBMRegressor(
        num_leaves=50,
        max_depth=10,
        learning_rate=0.01,
        n_estimators=3000,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=1,
        reg_lambda=1,
        verbose=-1  # Set -1 to suppress logs, or 100 for interval logs
    )
    lgbm_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[early_stopping(100)])

    # OOF Predictions
    oof_cat[valid_idx] = cat_model.predict(X_valid)
    oof_xgb[valid_idx] = xgb_model.predict(X_valid)
    oof_lgbm[valid_idx] = lgbm_model.predict(X_valid)

    # Test Predictions
    cat_preds[fold] = cat_model.predict(test_df)
    xgb_preds[fold] = xgb_model.predict(test_df)
    lgbm_preds[fold] = lgbm_model.predict(test_df)

    # Fold-wise RMSE
    cat_rmse = mean_squared_error(y_valid, oof_cat[valid_idx], squared=False)
    xgb_rmse = mean_squared_error(y_valid, oof_xgb[valid_idx], squared=False)
    lgbm_rmse = mean_squared_error(y_valid, oof_lgbm[valid_idx], squared=False)

    print(f'CAT_RMSE: {cat_rmse:.4f}, XGB_RMSE: {xgb_rmse:.4f}, LGBM_RMSE: {lgbm_rmse:.4f}')

# Average predictions across folds
pred_cat = np.expm1(np.mean(cat_preds, axis=0))
pred_xgb = np.expm1(np.mean(xgb_preds, axis=0))
pred_lgbm = np.expm1(np.mean(lgbm_preds, axis=0))

# Weighted Ensemble
final_pred = pred_cat * 0.3 + pred_xgb * 0.3 + pred_lgbm * 0.4
final_pred = np.clip(final_pred, 1, 314)