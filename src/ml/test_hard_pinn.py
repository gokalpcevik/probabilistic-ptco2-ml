import torch
import numpy as np
import matplotlib.pyplot as plt
from src.ml.training.train_sternvolmer import SternVolmerMLP, stern_volmer_predict
from src.data_processing.o2_dataset import OxygenDataset
import sys

def load_model_and_data():
    stats = np.load("models/normalization_stats_physics.npz")
    min_vals = stats["min"]
    max_vals = stats["max"]
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0  # avoid divide-by-zero

    test_dataset = OxygenDataset(
        "data/processed/", split="test", normalize=True
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = SternVolmerMLP(input_dim=2)
    model.load_state_dict(torch.load("models/physics_only_mlp.pth"))
    model.eval()

    return model, test_loader, min_vals, max_vals, ranges


def test_simulated_input(features, lifetime, true_po2):
    model, _, min_vals, max_vals, ranges = load_model_and_data()
    with torch.no_grad():
        # We need to take log1p of the pulses to match the training data, the second feature will be pulses (in the arguments)
        features[1] = np.log1p(features[1])
        # We need to normalize the input features (temperature and pulses)
        features = (features - min_vals[1:]) / ranges[1:]
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        # No need to normalize lifetime, it's already in the correct range
        lifetime = torch.tensor(lifetime, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        # Predict tau0, log_ksv, and offset using the model
        print(f"Input features (normalized): {features}")
        model_out = model(features)
        tau0 = model_out[:, 0]
        log_ksv = model_out[:, 1]
        offset = model_out[:, 2]
        # Calculate pO2 using the Stern-Volmer equation
        pred_po2 = stern_volmer_predict(tau0, log_ksv, lifetime, offset)
        # Calculate the error
        error = np.abs((pred_po2.item() - true_po2) / true_po2) * 100  # percentage error
        # Return the predicted pO2 and error
        return pred_po2.item(), error


def test_physics_only():
    model, test_loader, min_vals, max_vals, ranges = load_model_and_data()

    all_true_po2 = []
    all_pred_po2 = []
    all_pred_tau0 = []
    all_pred_ksv = []
    all_measured_lifetime = []

    with torch.no_grad():
        for features, true_po2 in test_loader:
            lifetime_norm = features[:, 0]
            temperature_norm = features[:, 1]
            pulses_norm = features[:, 2]

            # --- De-normalize lifetime ---
            lifetime = lifetime_norm * (max_vals[0] - min_vals[0]) + min_vals[0]

            # Prepare physics inputs (normalized temperature and pulses)
            physics_inputs = torch.stack([temperature_norm, pulses_norm], dim=1)

            model_out = model(physics_inputs)
            tau0 = model_out[:, 0]
            log_ksv = model_out[:, 1]
            offset = model_out[:, 2]

            pred_po2 = stern_volmer_predict(tau0, log_ksv, lifetime, offset) 
            all_true_po2.append(true_po2.item()) 
            all_pred_po2.append(pred_po2.item())
            all_pred_tau0.append(tau0.item())
            all_pred_ksv.append(torch.exp(log_ksv).item())  # remember: Ksv is exp(logKsv)
            all_measured_lifetime.append(lifetime.item())

    all_true_po2 = np.array(all_true_po2)
    all_pred_po2 = np.array(all_pred_po2)
    all_pred_tau0 = np.array(all_pred_tau0)
    all_pred_ksv = np.array(all_pred_ksv)
    all_measured_lifetime = np.array(all_measured_lifetime)

    mse = np.mean((all_true_po2 - all_pred_po2) ** 2)
    r2 = 1.0 - np.sum((all_true_po2 - all_pred_po2) ** 2) / np.sum((all_true_po2 - np.mean(all_true_po2)) ** 2)

    print(f"Test MSE: {mse:.4f}")
    print(f"Test R2:  {r2:.4f}")

    plt.figure(figsize=(7,7))
    plt.scatter(all_true_po2, all_pred_po2, alpha=0.7)
    plt.plot([0, max(all_true_po2)], [0, max(all_true_po2)], '--', color='gray')
    plt.xlabel("True pO2 (cmHg)")
    plt.ylabel("Predicted pO2 (cmHg)")
    plt.title("True vs Predicted pO2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return all_true_po2, all_pred_po2, all_pred_tau0, all_pred_ksv, all_measured_lifetime

if __name__ == "__main__":
    # test_physics_only()
    # Get the simulated input (not normalized) for the test from the CLI arguments
    args = sys.argv[1:]
    if len(args) != 3:
        print("Usage: python test_hard_pinn.py <features> <obs_lifetime> <true_po2>")
        sys.exit(1)
    features = np.array([float(x) for x in args[0].split(",")])
    true_po2 = float(args[2])
    # Get the lifetime from the CLI arguments
    lifetime = float(args[1])
    # Call the test function with the simulated input
    pred_po2, error = test_simulated_input(features, lifetime, true_po2)
    print(f"Predicted pO2: {pred_po2:.4f} cmHg")
    print(f"True pO2: {true_po2:.4f} cmHg")
    print(f"Error: {error:.4f}%")
    print(f"Input features: {features}")
    print(f"Input lifetime: {lifetime:.4f} us")
    print(f"Input true pO2: {true_po2:.4f} cmHg")
    # Script usage example:
    # python test_hard_pinn.py "0.5,0.6" 0.7 0.8
    
