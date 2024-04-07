from sklearn.model_selection import train_test_split

def stratified_split(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
