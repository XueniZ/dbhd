import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

# Define a range of values for beta and rho
beta_values = [0.1, 0.2, 0.3]
rho_values = [0.01, 0.02, 0.03]

# Create arrays to store AMI and NMI scores
ami_scores = []
nmi_scores = []

# Generate ground truth and estimated labels (replace this with your actual labels)
ground_truth_labels = [0, 1, 0, 1, 0]

# Initialize subplots for AMI and NMI
fig, axs = plt.subplots(len(beta_values), 1, figsize=(8, 12), sharex=True)

# Plot AMI and NMI scores for different beta and rho values
for i, beta in enumerate(beta_values):
    ami_scores_beta = []
    nmi_scores_beta = []

    for j, rho in enumerate(rho_values):
        # Replace this with your actual estimation and labels
        estimated_labels = [0, 1, 0, 2, 0]

        # Calculate AMI and NMI
        ami = adjusted_mutual_info_score(ground_truth_labels, estimated_labels)
        nmi = normalized_mutual_info_score(ground_truth_labels, estimated_labels)

        ami_scores_beta.append(ami)
        nmi_scores_beta.append(nmi)

    ami_scores.append(ami_scores_beta)
    nmi_scores.append(nmi_scores_beta)

    axs[i].plot(rho_values, ami_scores_beta, label='AMI')
    axs[i].plot(rho_values, nmi_scores_beta, label='NMI')
    axs[i].set_title(f'Beta = {beta_values[i]}')
    axs[i].set_ylabel('Scores')
    axs[i].legend()

axs[-1].set_xlabel('Rho Values')
plt.tight_layout()
plt.show()
