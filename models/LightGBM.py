import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor, early_stopping


FOLDS = 2
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
lgbm_preds = np.zeros((FOLDS, len(test_df)))
oof_lgbm = np.zeros(len(X))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y), 1):
    print(f"\n{'#'*15} Fold {fold} {'#'*15}")

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y[valid_idx]
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
        random_state=42
    )
    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[early_stopping(100)]
    )
    oof_lgbm[valid_idx] = lgbm_model.predict(X_valid)
    lgbm_preds[fold - 1] = lgbm_model.predict(test_df)
    lgbm_rmse = np.sqrt(mean_squared_error(y_valid, oof_lgbm[valid_idx]))

print("LightGBM :", np.sqrt(mean_squared_error(y, oof_lgbm)))
pred_lgbm = np.expm1(lgbm_preds.mean(axis=0))