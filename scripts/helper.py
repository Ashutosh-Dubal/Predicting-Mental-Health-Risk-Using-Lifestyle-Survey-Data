

def data_audit(df):
    print("\nData Shape:", df.shape)
    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nNumeric Summary:")
    print(df.describe())

    print("\nDuplicates:", df.duplicated().sum())

    for col in df.select_dtypes(include='object'):
        print(f"\n📌 Unique values in {col}:", df[col].dropna().unique())
