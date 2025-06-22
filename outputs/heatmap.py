#EDA PLOTS AND GRAPHS FOR STATISTICS 
fig,axis=plt.subplots(nrows=7,ncols=1,figsize=(8,20))
for i,col in enumerate(train_df.select_dtypes(include=[np.number]).columns):
    sns.histplot(train_df[col],bins=30,kde=True,ax=axis[i])
    axis[i].set_title(f'Histogram of {col}')
plt.tight_layout()
plt.show()

fig, axis = plt.subplots(nrows=7, ncols=2, figsize=(12, 10))


axis = axis.reshape(7, 2)


numeric_cols = train_df.select_dtypes(include=[np.number]).drop('Calories', axis=1).columns

for i, feature in enumerate(numeric_cols):
    row = i // 2
    col_idx = i % 2
    sns.violinplot(x=train_df['Sex'], y=train_df[feature], ax=axis[row, col_idx])
    axis[row, col_idx].set_title(f'Sex vs {feature}')

plt.tight_layout()
plt.show()

sns.heatmap(train_df.corr(numeric_only=True),annot=True,cmap='coolwarm')
plt.title("Correlation HeatMap")
plt.show()

sns.pairplot(train_df)
plt.suptitle("Pairplot of All Features",y=1.02)
plt.show()