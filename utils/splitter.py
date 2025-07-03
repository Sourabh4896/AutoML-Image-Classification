# utils/splitter.py

from sklearn.model_selection import train_test_split

def split_dataset(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Splits dataset into training, validation, and test sets.
    
    Parameters:
        - X: numpy array of images
        - y: numpy array of labels
        - test_size: proportion of the dataset to include in the test split
        - val_size: proportion of the remaining train set to use as validation
        - random_state: seed for reproducibility

    Returns:
        - X_train, X_val, X_test
        - y_train, y_val, y_test
    """

    # First split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Then split part of train into validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train)

    return X_train, X_val, X_test, y_train, y_val, y_test
