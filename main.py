from src.data_loader import load_data

train_df, test_df = load_data()
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)