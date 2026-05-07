import matplotlib.pyplot as plt
from dataset import get_dataloaders

def show_noisy_samples(n=5):
    _, test_loader = get_dataloaders(batch_size=n,noise_factor=0.5)
    noisy, clean = next(iter(test_loader))

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    fig.suptitle("Avant / Après bruitage (MNIST)", fontsize=13)

    for i in range(n):
        clean_img = clean[i].squeeze()
        noisy_img = noisy[i].squeeze()

        axes[0, i].imshow(clean_img, cmap="gray", vmin=-1, vmax=1)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Propre", loc="left")

        axes[1, i].imshow(noisy_img, cmap="gray", vmin=-1, vmax=1)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Bruité", loc="left")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_noisy_samples()
