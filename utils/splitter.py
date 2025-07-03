# utils/splitter.py

from sklearn.model_selection import train_test_split

def split_dataset(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        X (np.array): Image data
        y (np.array): Labels
        test_size (float): Proportion of data to reserve for test
        val_size (float): Proportion of training data to reserve for validation
        random_state (int): Seed for reproducibility

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First, split into train and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Then, split remaining into train and validation
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative_size, random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
