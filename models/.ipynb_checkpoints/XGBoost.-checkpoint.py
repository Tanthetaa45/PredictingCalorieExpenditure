import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

FOLDS = 2
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
xgb_preds = np.zeros((FOLDS, len(test_df)))
oof_xgb = np.zeros(len(X))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y), 1):
    print(f"\n{'#'*15} Fold {fold} {'#'*15}")

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y[valid_idx]
    xgb_model = XGBRegressor(
        max_depth=10,
        n_estimators=2000,
        learning_rate=0.01,
        subsample=0.9,
        colsample_bytree=0.75,
        gamma=0.01,
        enable_categorical=True,
        tree_method="hist",
        eval_metric="rmse",
        verbosity=0,
        early_stopping_rounds=100,
        random_state=42
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)]
    )
    oof_xgb[valid_idx] = xgb_model.predict(X_valid)
    xgb_preds[fold - 1] = xgb_model.predict(test_df)
    xgb_rmse = np.sqrt(mean_squared_error(y_valid, oof_xgb[valid_idx]))

print("XGBoost  :", np.sqrt(mean_squared_error(y, oof_xgb)))
pred_xgb = np.expm1(xgb_preds.mean(axis=0))

