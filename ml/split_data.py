from sklearn.model_selection import train_test_split

def split_dataset(rfm_scaled, rfm_raw):
    print("STEP 3: Split Train Validation Test")

    X_train_scaled, X_temp_scaled, X_train_raw, X_temp_raw = train_test_split(
        rfm_scaled,
        rfm_raw,
        test_size=0.30,
        random_state=42
    )

    X_val_scaled, X_test_scaled, X_val_raw, X_test_raw = train_test_split(
        X_temp_scaled,
        X_temp_raw,
        test_size=0.50,
        random_state=42
    )

    X_train_scaled.to_csv("data/processed/train.csv", index=False)
    X_val_scaled.to_csv("data/processed/validation.csv", index=False)
    X_test_scaled.to_csv("data/processed/test.csv", index=False)

    X_train_raw.to_csv("data/processed/train_raw.csv", index=False)
    X_val_raw.to_csv("data/processed/validation_raw.csv", index=False)
    X_test_raw.to_csv("data/processed/test_raw.csv", index=False)

    print("Train :", len(X_train_scaled))
    print("Validation :", len(X_val_scaled))
    print("Test :", len(X_test_scaled))