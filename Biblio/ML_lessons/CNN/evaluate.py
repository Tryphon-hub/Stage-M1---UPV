# evaluate.py
import torch
import matplotlib.pyplot as plt

def evaluate(model, test_loader):
    model.eval()  # désactive le dropout, batchnorm, etc.

    correct = 0
    total = 0

    with torch.no_grad():  # désactive le calcul des gradients
        for images, labels in test_loader:
            output = model(images)                    # forward pass → [64, 10]
            _, predicted = torch.max(output, dim=1)   # classe avec le score max
            total += labels.size(0)                   # +64 à chaque batch
            correct += (predicted == labels).sum().item()  # compte les bonnes

    accuracy = 100 * correct / total
    print(f"Accuracy : {accuracy:.2f}%")


def visualize(model, test_loader):
    model.eval()

    images, labels = next(iter(test_loader))  # prend le premier batch

    with torch.no_grad():
        output = model(images)
        _, predicted = torch.max(output, dim=1)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap='gray')
        color = 'green' if predicted[i] == labels[i] else 'red'
        ax.set_title(f"Pred: {predicted[i]} | Real: {labels[i]}", color=color)
        ax.axis('off')
    plt.tight_layout()
    plt.show()