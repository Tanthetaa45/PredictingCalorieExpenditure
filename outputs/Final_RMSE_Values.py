print("\n========== Final Model Scores ==========")

cat_rmse = np.sqrt(mean_squared_error(y, oof_cat))
xgb_rmse = np.sqrt(mean_squared_error(y, oof_xgb))
lgbm_rmse = np.sqrt(mean_squared_error(y, oof_lgbm))

print(f"CatBoost  RMSE: {cat_rmse:.4f}")
print(f"XGBoost   RMSE: {xgb_rmse:.4f}")
print(f"LightGBM  RMSE: {lgbm_rmse:.4f}")