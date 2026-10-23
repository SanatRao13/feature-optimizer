# Given a set of data, find out which parameters are most important to include: optimize the model by selecting the most relevant features
# Minimize features while maximizing model performance using feature selection techniques


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import optuna

path = 'C:/Users/rt2sa/Downloads/parkinsons+telemonitoring/parkinsons_updrs.data' # Update this path with the actual local/remote path to the data you wish to analyze
data = pd.read_csv(path)

# Penalty factor dictates how much each additional feature should be penalized
# It is raised to the penalty factor (e.g. if pf = 1.5, then 2 features would be penalized by 2^1.5 = 2.828, so the
# hybrid score would be 0.3535 of the original R2)
penalty_factor = 2

# separate out target var and drop columns
tgt = data['HNR']
data = data.drop(columns=['HNR', 'subject#', 'test_time'])
data = data.rename(columns={
    'Jitter:RAP':'Jitter_RAP',
    'Jitter:PPQ5':'Jitter_PPQ5',
    'Jitter:DDP':'Jitter_DDP',
    'Shimmer:APQ3':'Shimmer_APQ3',
    'Shimmer:APQ5':'Shimmer_APQ5',
    'Shimmer:APQ11':'Shimmer_APQ11',
    'Jitter(%)':'Jitter_pct',
    'Shimmer:DDA':'Shimmer_DDA',
    'Jitter(Abs)':'Jitter_Abs',
    'Shimmer(dB)':'Shimmer_dB'
})

print(data.columns)
# split data into trainng, validation/optimization, and test sets
train, valid, tgt_train, tgt_valid = train_test_split(data, tgt, test_size=0.5, random_state=42)
valid, test, tgt_valid, tgt_test = train_test_split(valid, tgt_valid, test_size=0.5, random_state=42)

# defines R2 metric for model evaluation
def r2(model, X, y):
    preds = model.predict(X)
    return 1 - (np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2))

def h_score(r2, feats_len, max_num_feats, pf):
    return (1-r2)**2+((feats_len**pf)/max_num_feats)**2

# sets up baseline LightGBM model and evaluates R2 on full dataset
# this code can be replaced with any other model you wish to use for feature optimization
model = LGBMRegressor(verbosity=-1)
model.fit(train, tgt_train)
r2_score = r2(model, test, tgt_test)


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
    hybrid_score = h_score(r2_score, len(selected_features), train.shape[1],penalty_factor)
    return hybrid_score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)

# extracts columns that were selected by the best trial
best_feats = [key for key,value in study.best_trial.params.items() if value]

# trains a model on the best features of the training set, scores on test set
model = LGBMRegressor(verbosity=-1)
model.fit(train[best_feats], tgt_train)
opz_r2=float(r2(model,test[best_feats],tgt_test))

# Disp results
print("------------------------------------------------------------------------------------")
print("Results")
print("------------------------------------------------------------------------------------")
print("Original R2:", r2_score)
print("Most Optimal R2 Score:", opz_r2)
print("Percent Diff in R2:", (opz_r2-r2_score)/r2_score*100, "%")
print("------------------------------------------------------------------------------------")
hybrid_score = h_score(r2_score, train.shape[1], train.shape[1], penalty_factor)
print("Original Hybrid Score:", hybrid_score)
print("Original Categories Used:", train.shape[1])
opz_hybrid_score = h_score(opz_r2, len(best_feats), train.shape[1], penalty_factor)
print("Optimized h-score:", opz_hybrid_score)
print("Percent Reduction in h-score:", (hybrid_score-opz_hybrid_score)/hybrid_score * 100, "%")
print("Best params:", best_feats)
print("Optimal Number of Categories:", len(best_feats))
