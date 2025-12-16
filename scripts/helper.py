from sklearn.pipeline import Pipeline

def data_audit(df):
    print("\nData Shape:", df.shape)
    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nNumeric Summary:")
    print(df.describe())

    print("\nDuplicates:", df.duplicated().sum())

    for col in df.select_dtypes(include='object'):
        print(f"\nUnique values in {col}:", df[col].dropna().unique())

def extract_estimator_with_attr(model, attr_name):
    if hasattr(model, attr_name):
        return model
    if isinstance(model, Pipeline):
        for _, step in model.named_steps.items():
            if hasattr(step, attr_name):
                return step
    raise AttributeError(f"No step with {attr_name} found.")

def build_pretty_name_mapping(preprocessor):
    """
    Map encoded feature names (e.g. 'cat__col_1') to human-readable labels
    (e.g. 'col = 1'). Handles:
      - 'cat__' (OneHotEncoder with drop='first')
      - 'remainder__' (passthrough numeric columns)
    """
    mapping = {}

    ohe = preprocessor.named_transformers_['cat']
    cat_cols = ohe.feature_names_in_          # original categorical col names
    cat_categories = ohe.categories_          # list of category arrays per feature

    for col, cats in zip(cat_cols, cat_categories):
        for cat_val in cats:
            if ohe.drop == 'first' and cat_val == cats[0]:
                continue

            full_name = f"cat__{col}_{cat_val}"
            pretty = f"{col} = {cat_val}"
            mapping[full_name] = pretty

    all_feature_names = preprocessor.get_feature_names_out()
    for name in all_feature_names:
        if name.startswith("remainder__"):
            orig = name.split("__", 1)[1]
            mapping[name] = orig

    return mapping