import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_data(x, y):
    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap="Spectral")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Data Distribution")
    plt.show()


def plot_decision_boundary(model, x, y):
    x_span = np.linspace(min(x[:, 0]) - 0.25, max(x[:, 0]) + 0.25, 50)
    y_span = np.linspace(min(x[:, 1]) - 0.25, max(x[:, 1]) + 0.25, 50)
    xx, yy = np.meshgrid(x_span, y_span)
    grid = np.c_[xx.ravel(), yy.ravel()]

    if hasattr(model, "predict_proba"):
        pred_func = model.predict_proba(grid)[:, 1]
    elif hasattr(model, "forward"):
        pred_func = model(torch.tensor(grid, dtype=torch.double)).detach().numpy()
    else:
        pred_func = model(grid).numpy()

    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap="RdYlGn", alpha=0.7)
    plt.colorbar()
    plt.scatter(x[:, 0], x[:, 1], c=y, marker="x", cmap="Spectral")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.show()
