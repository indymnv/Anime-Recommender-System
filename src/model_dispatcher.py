from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn import ensemble

models = {
    "dt_5": tree.DecisionTreeRegressor(
        max_depth=5, random_state=42
    ),
    "dt_5c": tree.DecisionTreeClassifier(
        max_depth=5, random_state=42
    ),
    "dt_10c": tree.DecisionTreeClassifier(
        max_depth=10, random_state=42
    ),
    "dt_10": tree.DecisionTreeRegressor(
        max_depth=10, random_state=42
    ),
    "dt_15": tree.DecisionTreeRegressor(
        max_depth=15, random_state=42
    ),
    "dt_15c": tree.DecisionTreeClassifier(
        max_depth=15, random_state=42
    ),
    "rf_5": ensemble.RandomForestRegressor(n_estimators = 10, 
                           random_state = 42, 
                           max_depth=5,
                          n_jobs=-1
    ),
    "rf_10": ensemble.RandomForestRegressor(n_estimators = 100, 
                           random_state = 42, 
                           max_depth=10,
                          n_jobs=-1
    ),
    "rf_15": ensemble.RandomForestRegressor(n_estimators = 10, 
                           random_state = 42, 
                           max_depth=15,
                          n_jobs=-1
    ),
    "rf_5c": ensemble.RandomForestClassifier(n_estimators = 10, 
                           random_state = 42, 
                           max_depth=5,
                          n_jobs=-1
    ),
    "rf_10c": ensemble.RandomForestClassifier(n_estimators = 100, 
                           random_state = 42, 
                           max_depth=10,
                          n_jobs=-1
    ),
    "rf_15c": ensemble.RandomForestClassifier(n_estimators = 10, 
                           random_state = 42, 
                           max_depth=15,
                          n_jobs=-1
    ),
    "gb_10c": ensemble.GradientBoostingClassifier(n_estimators = 10, 
                            learning_rate=1.0,
                           random_state = 42, 
                           criterion='mae',
                           max_depth=10
    ),
    "lr": LinearRegression(),
}