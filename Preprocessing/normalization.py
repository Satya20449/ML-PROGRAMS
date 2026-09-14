from sklearn.preprocessing import MinMaxScaler


def normalize_data(data):
    """
    Normalize numerical columns using Min-Max Scaling.
    """

    print("\n" + "=" * 60)
    print("                  NORMALIZATION")
    print("=" * 60)

    numeric_columns = data.select_dtypes(include="number").columns

    print("\nNormalization Method: Min-Max Scaling")

    # Create scaler
    scaler = MinMaxScaler()

    # Apply normalization
    data[numeric_columns] = scaler.fit_transform(
        data[numeric_columns]
    )

    print("\nNormalized Columns:")
    for column in numeric_columns:
        print("-", column)

    return data