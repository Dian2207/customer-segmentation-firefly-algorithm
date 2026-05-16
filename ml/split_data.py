from sklearn.model_selection import train_test_split

def split_dataset(rfm_scaled):
    print("STEP 3: Split Train Validation Test")

    X_train, X_temp = train_test_split(
        rfm_scaled,
        test_size=0.30,
        random_state=42
    )

    X_val, X_test = train_test_split(
        X_temp,
        test_size=0.50,
        random_state=42
    )

    X_train.to_csv("data/processed/train.csv", index=False)
    X_val.to_csv("data/processed/validation.csv", index=False)
    X_test.to_csv("data/processed/test.csv", index=False)

    print("Train :", len(X_train))
    print("Validation :", len(X_val))
    print("Test :", len(X_test))