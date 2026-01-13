import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
import os
from torch.utils.data import random_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# ================================================================
# HARD-CODED KNOT LABELS
# ================================================================
KNOT_TYPES = [
    "0_1",
    "3_1",
    "4_1",
    "5_1",
]
NUM_CLASSES = len(KNOT_TYPES)

# ================================================================
# LOGGING
# ================================================================
def setup_logger(logfile):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler()
        ],
    )

# ================================================================
# DATASET
# ================================================================
class PolymerDataset(Dataset):
    def __init__(self, coord_file, label_file):
        self.coords = np.loadtxt(coord_file)
        self.labels = np.loadtxt(label_file).astype(int)

        self.coords = torch.tensor(self.coords, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        self.N = self.coords.shape[1] // 3

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x = self.coords[idx].view(self.N, 3)
        y = self.labels[idx]
        return x, y


# ================================================================
# ENCODER
# ================================================================
class Encoder(nn.Module):
    def __init__(self, N, d_model=128, nhead=8, latent_dim=32):
        super().__init__()

        self.embed = nn.Linear(3, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)

        self.conv1 = nn.Conv1d(d_model, d_model, 4, stride=2, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, 4, stride=2, padding=1)

        reduced_len = N // 4
        self.fc_mu = nn.Linear(d_model * reduced_len, latent_dim)
        self.fc_logvar = nn.Linear(d_model * reduced_len, latent_dim)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        return self.fc_mu(x), self.fc_logvar(x)


# ================================================================
# DECODER
# ================================================================
class Decoder(nn.Module):
    def __init__(self, N, d_model=128, latent_dim=32):
        super().__init__()

        reduced_len = N // 4
        self.fc = nn.Linear(latent_dim, d_model * reduced_len)

        self.deconv1 = nn.ConvTranspose1d(d_model, d_model, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose1d(d_model, d_model, 4, 2, 1)
        self.out = nn.Conv1d(d_model, 3, 1)

        self.reduced_len = reduced_len
        self.d_model = d_model

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), self.d_model, self.reduced_len)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.out(x)
        return x.transpose(1, 2)


# ================================================================
# CLASSIFIER
# ================================================================
class KnotClassifier(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, z):
        return self.net(z)


# ================================================================
# FULL MODEL
# ================================================================
class KnotVAE(nn.Module):
    def __init__(self, N, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(N, latent_dim=latent_dim)
        self.decoder = Decoder(N, latent_dim=latent_dim)
        self.classifier = KnotClassifier(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, self.classifier(z), z


# ================================================================
# LOSS
# ================================================================
def compute_loss(x, x_rec, mu, logvar, logits, labels, beta=1.0, gamma=1.0):
    recon = F.mse_loss(x_rec, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    ce = F.cross_entropy(logits, labels)
    return recon + beta * kl + gamma * ce, recon, kl, ce


# ================================================================
# TRAINING WITH SCHEDULER + EARLY STOPPING
# ================================================================
def train(coord_file, label_file,
          epochs=100, batch_size=32, lr=1e-3,
          patience=10, val_fraction=0.2):

    setup_logger("training.log")

    full_dataset = PolymerDataset(coord_file, label_file)

    # ------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # ------------------------------------------------
    n_total = len(full_dataset)
    n_val = int(val_fraction * n_total)
    n_train = n_total - n_val

    train_set, val_set = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KnotVAE(full_dataset.N).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=5
    )

    best_val_recon = float("inf")
    patience_counter = 0

    # ------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------
    for epoch in range(epochs):

        # --------------------
        # TRAIN
        # --------------------
        model.train()
        train_totals = np.zeros(4)

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            x_rec, mu, logvar, logits, _ = model(x)
            loss, r, kl, ce = compute_loss(x, x_rec, mu, logvar, logits, y)

            loss.backward()
            optimizer.step()

            train_totals += np.array([loss.item(), r.item(), kl.item(), ce.item()])

        train_totals /= len(train_loader)

        # --------------------
        # VALIDATION
        # --------------------
        model.eval()
        val_totals = np.zeros(4)

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                x_rec, mu, logvar, logits, _ = model(x)
                loss, r, kl, ce = compute_loss(x, x_rec, mu, logvar, logits, y)
                val_totals += np.array([loss.item(), r.item(), kl.item(), ce.item()])

        val_totals /= len(val_loader)

        # Scheduler uses total validation loss
        scheduler.step(val_totals[1]+val_totals[3])

        lr_current = optimizer.param_groups[0]["lr"]

        logging.info(
            f"Epoch {epoch+1:03d} | "
            f"TRAIN: Total {train_totals[0]:.4f} | "
            f"Recon {train_totals[1]:.4f} | "
            f"KL {train_totals[2]:.4f} | "
            f"CE {train_totals[3]:.4f} || "
            f"VAL: Total {val_totals[0]:.4f} | "
            f"Recon {val_totals[1]:.4f} | "
            f"KL {val_totals[2]:.4f} | "
            f"CE {val_totals[3]:.4f} | "
            f"LR {lr_current:.2e}"
        )

        # ------------------------------------------------
        # EARLY STOPPING BASED ON VAL RECONSTRUCTION LOSS
        # ------------------------------------------------
        if ( val_totals[1]  + val_totals[3] ) < best_val_recon:
            best_val_recon = val_totals[1] + val_totals[3]
            patience_counter = 0

            torch.save(
                model.state_dict(),
                "best_model.pt"
            )

            logging.info(
                f"  ✓ New best validation reconstruction loss: "
                f"{best_val_recon:.6f} — model saved."
            )

        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info(
                "Early stopping triggered "
                f"(no improvement in val reconstruction loss for {patience} epochs)."
            )
            break


# ================================================================
# EVALUATION MODE
# ================================================================
def evaluate(coord_file, label_file,
             model_path="best_model.pt",
             output_prefix="eval"):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ================================================================
    # LOAD DATA
    # ================================================================
    dataset = PolymerDataset(coord_file, label_file)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False
    )

    # ================================================================
    # LOAD MODEL
    # ================================================================
    model = KnotVAE(dataset.N).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_latents = []
    all_preds = []
    all_coords = []

    # ================================================================
    # FORWARD PASS
    # ================================================================
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            x_rec, mu, logvar, logits, z = model(x)

            all_latents.append(z.cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_coords.append(x_rec.cpu().numpy())

    all_latents = np.concatenate(all_latents, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_coords = np.concatenate(all_coords, axis=0)

    # ================================================================
    # t-SNE PROJECTION
    # ================================================================
    tsne = TSNE(n_components=2, random_state=0)
    Z_tsne = tsne.fit_transform(all_latents)

    np.savetxt(
        f"{output_prefix}_tsne.txt",
        np.column_stack([Z_tsne, all_preds]),
        header="tsne_x tsne_y predicted_knot"
    )

    # Scatter plot (t-SNE)
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(KNOT_TYPES):
        mask = all_preds == i
        plt.scatter(
            Z_tsne[mask, 0], Z_tsne[mask, 1],
            s=8, alpha=0.7, label=name
        )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE latent space")
    plt.legend()
    plt.axis("equal")
    plt.savefig(f"{output_prefix}_tsne_scatter.png", dpi=300)
    plt.close()


    # ================================================================
    # PCA PROJECTION
    # ================================================================
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(all_latents)

    np.savetxt(
        f"{output_prefix}_pca.txt",
        np.column_stack([Z_pca, all_preds]),
        header="pca_1 pca_2 predicted_knot"
    )

    # Scatter plot (PCA)
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(KNOT_TYPES):
        mask = all_preds == i
        plt.scatter(
            Z_pca[mask, 0], Z_pca[mask, 1],
            s=8, alpha=0.7, label=name
        )

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("PCA latent space")
    plt.legend()
    plt.axis("equal")
    plt.savefig(f"{output_prefix}_pca_scatter.png", dpi=300)
    plt.close()


    # ================================================================
    # SAVE DECODED CONFIGURATIONS (.xyz)
    # ================================================================
    if all_coords.ndim == 2:
        num_samples, dim = all_coords.shape
        N = dim // 3
        coords_list = [all_coords[i].reshape(N, 3) for i in range(num_samples)]
    elif all_coords.ndim == 3:
        num_samples, N, _ = all_coords.shape
        coords_list = [all_coords[i] for i in range(num_samples)]
    else:
        raise ValueError(f"Unexpected decoded shape: {all_coords.shape}")

    with open(f"{output_prefix}_decoded.xyz", "w") as f:
        for i in range(num_samples):
            f.write(f"{N}\n")
            f.write(f"Frame {i+1}, predicted knot: {KNOT_TYPES[all_preds[i]]}\n")
            for r in coords_list[i]:
                f.write(f"C {r[0]:.6f} {r[1]:.6f} {r[2]:.6f}\n")

    print("Evaluation completed:")
    print(" - t-SNE scatter + density")
    print(" - PCA scatter + density")
    print(" - decoded .xyz trajectory")


# ================================================================
# CLI
# ================================================================
if __name__ == "__main__":
    import sys
    mode = sys.argv[1]

    if mode == "train":
        train(sys.argv[2], sys.argv[3])
    elif mode == "eval":
        evaluate(sys.argv[2], sys.argv[3])

