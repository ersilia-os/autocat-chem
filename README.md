# Automated surrogate model building with catboost for chemistry datasets.

**WORK IN PROGRESS**

This package provides automated machine learning catboost surrogate models for multi-output chemistry datasets.

Works on regression, single task or multitask.

NB - input .csv files expected to have a header row

## Fit/Save

Python:

Morgan fingerprints:
auto = AutoCat()
auto.fit("chembl_100k_predictions.csv")

Molbert fingerprint reference library and optional training params:
auto = AutoCat(reference_lib="reference.h5")
auto.fit("chembl_100k_predictions.csv", optimise_time=1200, weight=True)

auto.save("chembl_45k_molbert_predictions.cbm")

## Predict

Python:

auto = AutoCat() #Or with molbert library: auto = AutoCat(reference_lib="reference.h5")

preds = auto.predict(smiles.csv)
