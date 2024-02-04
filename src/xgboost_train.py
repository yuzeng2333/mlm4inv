import xgboost as xgb
from dataloader import read_data, canonicalize_np
from config import MASK_IDX, MASK_VAR
import numpy as np
import matplotlib.pyplot as plt

def xgboost_train(args, file_path):
    model = xgb.XGBRegressor(objective ='reg:squarederror', seed=42)
    # the header row is already removed
    features, var_dict = read_data(file_path)
    # if MASK_VAR is not None, then use the index of the variable
    if MASK_VAR is not None:
        MASK_IDX = var_dict[MASK_VAR]
    # split the data into input and output
    # MASK_IDX is the index of the output
    X, y = np.delete(features, MASK_IDX, axis=1), features[:, MASK_IDX]
    # canonicalize the input data
    X = canonicalize_np(X, 0)
    # canonicalize the output data
    y = canonicalize_np(y.reshape(-1, 1), 0)
    y = y.reshape(-1)
    model.fit(X, y)
    # Get feature importance
    feature_importance = model.get_booster().get_score(importance_type='weight')  # or 'gain' or 'cover'

    # Sort features based on importance
    sorted_feature_importance = sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)

    # Print sorted features and their importance
    print("Feature Importance:")
    for feature, importance in sorted_feature_importance:
        print(f"{feature}: {importance}")