import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

cat_model = CatBoostRegressor(
        iterations=3500,
        learning_rate=0.02,
        depth=12,
        l2_leaf_reg=3,
        verbose=1000,
        early_stopping_rounds=200,
        loss_function='RMSE',
        eval_metric='RMSE',
        task_type='CPU',
        random_state=42
    )
cat_model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        cat_features=cat_idx
    )
oof_cat[valid_idx] = cat_model.predict(X_valid)
cat_rmse = np.sqrt(mean_squared_error(y_valid, oof_cat[valid_idx]))
print("CatBoost :", np.sqrt(mean_squared_error(y, oof_cat)))
pred_cat = np.expm1(cat_preds.mean(axis=0))