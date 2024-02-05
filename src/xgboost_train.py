import xgboost as xgb
from dataloader import read_data, canonicalize_np
from config import MASK_IDX, MASK_VAR
import numpy as np
import matplotlib.pyplot as plt

def xgboost_train(args, file_path):
    DO_CANONICALIZE = 1
    model = xgb.XGBRegressor(objective ='reg:squarederror', seed=44, verbosity=1, n_estimators=1000)
    # the header row is already removed
    features, var_dict = read_data(file_path)
    # if MASK_VAR is not None, then use the index of the variable
    if MASK_VAR is not None:
        MASK_IDX = var_dict[MASK_VAR]
    # split the data into input and output
    # MASK_IDX is the index of the output
    X, y = np.delete(features, MASK_IDX, axis=1), features[:, MASK_IDX]
    # determine the feature name for each column in X
    feature_names = list(var_dict.keys())
    feature_names.pop(MASK_IDX)
    # canonicalize the input data
    if DO_CANONICALIZE:
        X = canonicalize_np(X, 0)
    # canonicalize the output data
    if DO_CANONICALIZE:
        y = canonicalize_np(y.reshape(-1, 1), 0)
        y = y.reshape(-1)

    dtrain = xgb.DMatrix(X, label=y)
    param = {
        'max_depth': 3,  # the maximum depth of each tree
        'eta': 0.3,  # the training step for each iteration
        'objective': 'reg:squarederror',  # error evaluation for regression tasks
        'alpha': 0.1,  # L1 regularization term on weights
        'lambda': 1,  # L2 regularization term on weights
        'gamma': 0.1  # Minimum loss reduction required to make a further partition
    }
    num_round = 1000  # the number of training iterations
    model = xgb.train(param, dtrain, num_round)
    #model.fit(X, y, eval_set=[(X, y)], eval_metric='mae', verbose=True)
    # Get feature importance
    feature_importance = model.get_score(importance_type='cover')

    # Sort features based on importance
    sorted_feature_importance = sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)

    # Print sorted features names and their importance
    print("Feature Importance:")
    idx = 0
    for feature, importance in sorted_feature_importance:
        print(f"{feature_names[idx]}: {importance}")
        idx += 1

    xgb.plot_tree(model, num_trees=2)
    plt.show()