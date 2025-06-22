import matpotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.kdeplot(np.expm1(oof_cat), label='CatBoost OOF', fill=True, alpha=0.3)
sns.kdeplot(np.expm1(oof_xgb), label='XGBoost OOF', fill=True, alpha=0.3)
sns.kdeplot(np.expm1(oof_lgbm), label='LightGBM OOF', fill=True, alpha=0.3)
sns.kdeplot(train_df['Calories'], label='True Calories', fill=True, alpha=0.3)

plt.legend()
plt.title("OOF Prediction vs True Values")
plt.xlabel("Calories")
plt.ylabel("Density")
plt.show()