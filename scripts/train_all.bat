@echo off
:: Classic Stern-Volmer Model (1919)
python -m src.ml.training.train_sternvolmer
:: Modified Stern-Volmer Model
python -m src.ml.training.train_msv
:: Multi-site Quenching Model
python -m src.ml.training.train_multisite
:: Joint Regressor
python -m src.ml.training.train_pinn_extrap
:: Two-Stage Regressor
python -m src.ml.training.train_two_stage --stage 2 --lambda_phys 50 --unfreeze_pct 0.9
