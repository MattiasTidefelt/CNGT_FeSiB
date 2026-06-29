# CNGT_FeSiB
CNGT model used for generating mapping and training surrogate models

These scripts are to be used as a three-step rocket.

***1)*** Evaluate your energies using TernaryGibbs.py, and once satisfied, play with dcTernaryAny.py until you get a sense for if things work as they should. After this, or alongside this, run instances for NucMulti_FeSiB_NoSolve.py, or IsoTherm_FeSiB.py, or Amxxx.

***2)*** Sample compositions as functions of heating/cooling rates using SampleComp.py. Then adjust the MPI_Driver_NucFeSiB.py to the heat treatment that is to be mapped and set the number of threads to use during sampling in Run_MPI_FeSiB.sh. After this, plotting can be made using Plot_MPI_Sampling.py

***3)*** Perform a hyperparameter search using Train_ML_Optuna_kFold.py. After this, visualize your result with the best parameters from the hyperparameter search using ML_interpolation_FeSiB_kFold.py. Finally, Apply_ML_model.py can be used to draw enormous amounts of samples from the trained surrogate model. To note is that the hyperparameter search is time consuming, and can be skipped if manual adjustments of the parameters are of interest with som minor adjustments to the ML_interpolation_kFold.py script.

All parameters are found in paramFeSiB.py, but the heat treatment should be keept track of in the above mentioned scripts.

***Following the functions from the above core scripts explains how they're connected***
