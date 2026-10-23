# Given a set of data, find out which parameters are most important to include: optimize the model by selecting the most relevant features
# Minimize features while maximizing model performance using feature selection techniques

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import optuna

path = 'path_to_data_file.csv' # Update this path with the actual local/remote path to the data you wish to analyze
data = pd.read_csv(path)

# Penalty factor dictates how much each additional feature should be penalized
# It is raised to the penalty factor (e.g. if pf = 1.5, then 2 features would be penalized by 2^1.5 = 2.828, so the
# hybrid score would be 0.3535 of the original R2)
penalty_factor = 1.5

# separate out target var and drop columns
tgt = data['target']
data = data.drop(columns=['target', 'other_column_to_drop', ...])

# split data into trainng, validation/optimization, and test sets
# splits initial dataset into 50% training data, 25% validation data and 25% test data
train, valid, tgt_train, tgt_valid = train_test_split(data, tgt, test_size=0.5, random_state=42)
valid, test, tgt_valid, tgt_test = train_test_split(valid, tgt_valid, test_size=0.5, random_state=42)

# defines R2 metric for model evaluation
def r2(model, X, y):
    preds = model.predict(X)
    return 1 - (np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2))

# sets up baseline LightGBM model and evaluates R2 on full dataset
# this code can be replaced with any other model you wish to use for feature optimization
model = LGBMRegressor(verbosity=-1)
model.fit(train, tgt_train)
r2_score = r2(model, test, tgt_test)
print("R2 on full dataset:", r2_score)

hybrid_score = r2_score/(train.shape[1] ** penalty_factor) # R2/(num_features^pf)
print("Hybrid Score:", hybrid_score)
print("Categories Used:", train.shape[1])

# sets up objective function for optuna to optimize
def objective(trial):
    # Suggest a subset of features to include in the model
    selected_features = []
    for feature in data.columns:
        if trial.suggest_categorical(feature, [True, False]):
            selected_features.append(feature) # adds BO-selected features to list of features to include in training/prediction

    if not selected_features:
        return 0.0  # Return a score of 0 if no features are selected, as it implies that features have no meaningful predictive power

    # Train the model with the selected features
    X_train = train[selected_features]
    X_valid = valid[selected_features]

    # train model on subset of features and get R2 score
    model = LGBMRegressor(verbosity=-1)
    model.fit(X_train, tgt_train)
    r2_score = r2(model, X_valid, tgt_valid)

    # Calculate hybrid score to optimize R2 and minimize features used
    hybrid_score = r2_score/(len(selected_features) ** penalty_factor)
    return hybrid_score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)

# extracts columns that were selected by the best trial
best_feats = [key for key,value in study.best_trial.params.items() if value]

# trains a model on the best features of the training set, scores on test set
model = LGBMRegressor(verbosity=-1)
model.fit(train[best_feats], tgt_train)
opz_r2=float(r2(model,test[best_feats],tgt_test))

# Disp results
print("Most Optimal R2 Score:", opz_r2)
print("Best params:", best_feats)
print("Optimal Number of Categories:", len(best_feats))
print("Percent Diff in R2:", (opz_r2-r2_score)/r2_score*100, "%")
