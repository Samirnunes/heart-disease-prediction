from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

random_state = 100

selected_model = LogisticRegression()

models = [
        RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=3, max_leaf_nodes=10),
        DecisionTreeClassifier(random_state=random_state, max_depth=3, max_leaf_nodes=10),
        LogisticRegression(),
        SVC(probability=True, random_state=random_state),
        XGBClassifier(random_state=random_state),
        KNeighborsClassifier(n_neighbors=5)
    ]

names = [
    "RandomForestMaxDepth3MaxLeafNodes10",
    "DecisionTreeMaxDepth3MaxLeafNodes10",
    "LogisticRegressionBase",
    "SVCBase",
    "XGBClassifierBase",
    "KNeighborsClassifier"
]