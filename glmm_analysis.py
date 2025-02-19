#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:44:30 2025

@author: elianalousada
"""

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load the dataset
file_path = "GLMM.xlsx"
df = pd.read_excel(file_path, sheet_name="Feuil4")

# Convert categorical variables to categorical data type
df["Learning_stage"] = df["Learning_stage"].astype("category")
df["Sex"] = df["Sex"].astype("category")
df["Genotype"] = df["Genotype"].astype("category")
df["Batch"] = df["Batch"].astype("category")
df["Cage"] = df["Cage"].astype("category")

# Fit GLMM model for Performance
model_perf = smf.mixedlm(
    "Performance ~ Learning_stage + Sex + Genotype", 
    df, 
    groups=df["Mouse"]
)
result_perf = model_perf.fit()

print(result_perf.summary())

# Fit GLMM model for Number of Trials
model_trials = smf.mixedlm(
    "Number_of_Trials ~ Learning_stage + Sex + Genotype", 
    df, 
    groups=df["Mouse"]
)
result_trials = model_trials.fit(method = ["lbfgs"])
print(result_trials.summary())

# Fit GLMM model for PRT
model_prt = smf.mixedlm(
    "PRT ~ Learning_stage + Sex + Genotype", 
    df, 
    groups=df["Mouse"]
)
result_prt = model_prt.fit()
print(result_prt.summary())


"""
Post-hoc analysis
"""

# Post hoc test for Performance
posthoc_perf = pairwise_tukeyhsd(df["Performance"], df["Learning_stage"])
print(posthoc_perf)

# Post hoc test for Number of Trials
posthoc_trials = pairwise_tukeyhsd(df["Number_of_Trials"], df["Learning_stage"])
print(posthoc_trials)

# Post hoc test for PRT
posthoc_prt = pairwise_tukeyhsd(df["PRT"], df["Learning_stage"])
print(posthoc_prt)


# Save results
with open("glmm_results_final.txt", "w") as f:
    f.write("========== Performance Model ==========\n")
    f.write(result_perf.summary().as_text() + "\n\n")
    
    f.write("========== Number of Trials Model ==========\n")
    f.write(result_trials.summary().as_text() + "\n\n")
    
    f.write("========== PRT Model ==========\n")
    f.write(result_prt.summary().as_text() + "\n\n")

    f.write("========== Post-hoc Performance ==========\n")
    f.write(str(posthoc_perf) + "\n\n")

    f.write("========== Post-hoc Number of Trials ==========\n")
    f.write(str(posthoc_trials) + "\n\n")

    f.write("========== Post-hoc PRT ==========\n")
    f.write(str(posthoc_prt) + "\n\n")
