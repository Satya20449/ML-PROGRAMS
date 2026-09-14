import pandas as pd


def remove_outliers(data):
    """
    Detect and remove outliers using the IQR method.
    """

    print("\n" + "=" * 60)
    print("                 OUTLIER DETECTION")
    print("=" * 60)

    numeric_columns = data.select_dtypes(include="number").columns

    for column in numeric_columns:

        # Calculate Q1 and Q3
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)

        # Calculate IQR
        IQR = Q3 - Q1

        # Calculate limits
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        # Find outliers
        outliers = data[
            (data[column] < lower_limit) |
            (data[column] > upper_limit)
        ]

        print(f"\nColumn: {column}")
        print(f"Q1: {Q1:.2f}")
        print(f"Q3: {Q3:.2f}")
        print(f"IQR: {IQR:.2f}")
        print(f"Lower Limit: {lower_limit:.2f}")
        print(f"Upper Limit: {upper_limit:.2f}")
        print(f"Outliers Found: {len(outliers)}")

        # Remove outliers
        data = data[
            (data[column] >= lower_limit) &
            (data[column] <= upper_limit)
        ]

    print("\nOutlier handling completed.")
    print("Rows after outlier removal:", len(data))

    return data.reset_index(drop=True)