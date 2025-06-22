from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(
    train_df['Age'],
    train_df['Weight'],
    train_df['Heart_Rate'],
    c=np.expm1(oof_cat),
    cmap='viridis',
    s=50,
    alpha=0.8
)

cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('Predicted Calories')

ax.set_xlabel("Age")
ax.set_ylabel("Weight")
ax.set_zlabel("Heart Rate")
ax.set_title("3D Plot of Predicted Calories (CatBoost)")

plt.tight_layout()
plt.show()