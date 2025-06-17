import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
# Import EarlyStopping from xgboost.callback
from xgboost.callback import EarlyStopping
from lightgbm import LGBMRegressor, early_stopping

# ------------------ Assumed Setup ------------------
# X = train_df.drop(['Calories'], axis=1)
# y = np.log1p(train_df['Calories'].values)
# test_df = <your test data, preprocessed similarly to X>

FOLDS = 2
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

cat_preds = np.zeros((FOLDS, len(test_df)))
xgb_preds = np.zeros((FOLDS, len(test_df)))
lgbm_preds = np.zeros((FOLDS, len(test_df)))

oof_cat = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_lgbm = np.zeros(len(X))

# Optional: convert categorical columns
cat_cols = X.select_dtypes(include='object').columns.tolist()
cat_idx = [X.columns.get_loc(col) for col in cat_cols]

for col in cat_cols:
    X[col] = X[col].astype('category')
    test_df[col] = test_df[col].astype('category')

# ---------------------- CV Loop ----------------------
for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y), 1):
    print(f"\n{'#'*15} Fold {fold} {'#'*15}")

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y[valid_idx]

    # ------------------ CatBoost (GPU) ------------------
    cat_model = CatBoostRegressor(
        iterations=3500,
        learning_rate=0.02,
        depth=12,
        l2_leaf_reg=3,
        verbose=1000,
        early_stopping_rounds=200,
        loss_function='RMSE',
        eval_metric='RMSE',
        task_type='GPU',
        devices='0'  # You can specify multiple GPUs like '0,1' if needed
    )
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        cat_features=cat_idx
    )

    # ------------------ XGBoost (GPU) ------------------
    xgb_model = XGBRegressor(
        max_depth=10,
        n_estimators=2000,
        learning_rate=0.01,
        subsample=0.9,
        colsample_bytree=0.75,
        gamma=0.01,
        enable_categorical=True,
        tree_method="gpu_hist",  # <--- GPU acceleration
        predictor="gpu_predictor",
        eval_metric="rmse",
        verbosity=0,
        random_state=42,
        # Move early_stopping_rounds to the constructor
        # Pass a list containing the EarlyStopping callback instance
        # Removed the 'monitor' argument as it's not valid for the constructor
        callbacks=[EarlyStopping(rounds=100)]
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)]
        # Remove callbacks from fit method
    )

    # ------------------ LightGBM (GPU optional) ------------------
    lgbm_model = LGBMRegressor(
        num_leaves=50,
        max_depth=10,
        learning_rate=0.01,
        n_estimators=3000,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=1,
        reg_lambda=1,
        verbose=-1,
        device="gpu",  # Enable GPU for LightGBM (if compiled with GPU support)
        gpu_platform_id=0,
        gpu_device_id=0
    )
    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[early_stopping(100)]
    )

    # ------------------ OOF Predictions ------------------
    oof_cat[valid_idx] = cat_model.predict(X_valid)
    oof_xgb[valid_idx] = xgb_model.predict(X_valid)
    oof_lgbm[valid_idx] = lgbm_model.predict(X_valid)

    # ------------------ Test Predictions ------------------
    cat_preds[fold - 1] = cat_model.predict(test_df)
    xgb_preds[fold - 1] = xgb_model.predict(test_df)
    lgbm_preds[fold - 1] = lgbm_model.predict(test_df)

    # ------------------ Fold RMSE ------------------
    cat_rmse = np.sqrt(mean_squared_error(y_valid, oof_cat[valid_idx]))
    xgb_rmse = np.sqrt(mean_squared_error(y_valid, oof_xgb[valid_idx]))
    lgbm_rmse = np.sqrt(mean_squared_error(y_valid, oof_lgbm[valid_idx]))

    print(f'CAT_RMSE: {cat_rmse:.4f}, XGB_RMSE: {xgb_rmse:.4f}, LGBM_RMSE: {lgbm_rmse:.4f}')

# ---------------------- Final Predictions ----------------------
pred_cat = np.expm1(np.mean(cat_preds, axis=0))
pred_xgb = np.expm1(np.mean(xgb_preds, axis=0))
pred_lgbm = np.expm1(np.mean(lgbm_preds, axis=0))

# Weighted Ensemble
final_pred = pred_cat * 0.3 + pred_xgb * 0.3 + pred_lgbm * 0.4
final_pred = np.clip(final_pred, 1, 314)