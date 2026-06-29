#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 16:51:21 2025

@author: ag0406
"""

import optuna
from sklearn.model_selection import KFold
import time
import json
import optuna.visualization as vis
import numpy as np

# Htype = "Heating"
# Htype = "Quenching"
Htype = "AM"
# Htype = "Iso" 

# Type = "Uniform"
# Type = "Normal"
Type = "NormalBounded"
# Type = "NormalSigmoid"
# Type = "StandardScaler"
# Type = "Power"

from ML_utilities import load_data_Kfold, Network, train

data = load_data_Kfold(Htype,Type)
input_dimension = data["X_all"].shape[1]         
output_dimension = data["y_all"].shape[1]

def objective(trial):
    # --- Suggest hyperparameters ---
    n_layers = trial.suggest_int("n_layers", 2, 4)
    hidden_layers = [trial.suggest_int(f"n_units_layer{i}", 16, 32) for i in range(n_layers)]
    # hidden_layers = [trial.suggest_int(f"n_units_layer{i}", 16, 64) for i in range(n_layers)]
    # activation = trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid", "leaky_relu","silu","gelu"])
    # activation = trial.suggest_categorical("activation", ["tanh", "sigmoid", "leaky_relu","silu","gelu"])
    activation = trial.suggest_categorical("activation", ["leaky_relu"])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    lr_adam  = trial.suggest_float("lr_adam", 1e-5, 1e-3, log=True)

    epochs_adam  = trial.suggest_int("epochs_adam", 1000, 10000)  # match your ranges

    lam = trial.suggest_float("lambda", 0.0, 1e-2)  # modest L2 range
    batch_size = trial.suggest_int("batch_size", 32, 256)  # used only with Adam
    patience = trial.suggest_int("patience", 50, 300)      # early stop for Adam phase
    
    
    # Get optimization data
    X_opt = data["X_opt"]
    y_opt = data["y_opt"]

    kf = KFold(n_splits=data["nfolds"], shuffle=True, random_state=42)

    fold_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_opt)):
        X_train = X_opt[train_idx]
        y_train = y_opt[train_idx]

        X_val   = X_opt[val_idx]
        y_val   = y_opt[val_idx]


        # --- Build model ---
        model = Network(input_dimension, hidden_layers, output_dimension,
                        activation=activation, dropout=dropout)
    
    
        # Package fold data in your expected structure
        fold_data = {
            "train": {"X": X_train, "y": y_train},
            "val":   {"X": X_val,   "y": y_val}
        }
    
        # Train with early stopping
        Cost_train, Cost_val = train(
            model,
            epochs_adam,
            "Adam",
            fold_data,
            lr=lr_adam,
            lam=lam,
            batch_size=batch_size,
            patience=patience,
            print_every=None
        )        

        fold_best = float(min(Cost_val))
        fold_losses.append(fold_best)

        # ---- Pruning after each fold ----
        trial.report(fold_best, step=fold)   # step = fold index
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_losses))


simt = time.time()


study = optuna.create_study(
    direction="minimize",
    storage=f"sqlite:///optuna_study_{Htype}.db",  # file-based storage
    load_if_exists=True
)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=500, n_jobs = 3)  #200 for final

print('-------------------------')
simtime=str(round((time.time()-simt)/60,3))
print(f'Computation time: {simtime} min')

print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params:")
for k, v in trial.params.items():
    print(f"    {k}: {v}")
    

############## save best parameters
DirPathOut = f"/Users/ag0406/Documents/Mattias/PhD/CNT/FeBSi/MPI_Sampling/{Htype}/"
    
best_params = study.best_trial.params
best_value = study.best_trial.value

with open(f"{DirPathOut}best_params.json", "w") as f:
    json.dump({"best_params": best_params, "best_value": best_value}, f, indent=4)

print("Best parameters saved to best_params.json")


# Generate figures
fig1 = vis.plot_optimization_history(study)
fig2 = vis.plot_param_importances(study)
fig3 = vis.plot_parallel_coordinate(study)

# Or save as static PNG (requires kaleido)
fig1.write_image(f"{DirPathOut}optimization_history.png")
fig2.write_image(f"{DirPathOut}param_importances.png")
fig3.write_image(f"{DirPathOut}parallel_coordinate.png")




