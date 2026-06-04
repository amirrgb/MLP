import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import simplejson as json
from pathlib import Path
import random

BASE_DIR = Path(__file__).parent.parent
RESULT_DIR = BASE_DIR / "results"

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_data(X, y, title="Dataset"):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    plt.title(title)
    plt.show()

def save_table(results_list, filename):
    ensure_dir(RESULT_DIR / "tables")
    df = pd.DataFrame(results_list)
    
    path = RESULT_DIR / "tables" / f"{filename}.csv"
    df.to_csv(path, index=False)
    
    return df


def save_figure(fig, relativePath):
    full_path = RESULT_DIR / "figures" / f"{relativePath}.png"
    ensure_dir(full_path.parent)
    
    fig.savefig(full_path, bbox_inches='tight', dpi=100)
    plt.close(fig)


def plot_and_save_loss(train_hist, test_hist, title, relativePath):
    fig = plt.figure(figsize=(8, 5))
    plt.plot(train_hist, label="Train Loss")
    plt.plot(test_hist, label="Test Loss")
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
    save_figure(fig, relativePath)
    
def plot_and_save_accuracy(train_hist, test_hist, title, relativePath):
    fig = plt.figure(figsize=(8, 5))
    plt.plot(train_hist, label="Train Accuracy")
    plt.plot(test_hist, label="Test Accuracy")
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.ylim(0, 105)
    plt.show()
    save_figure(fig, relativePath)
    


def plot_and_save_decision_boundary(model, X, y, relativePath):

    device = next(model.parameters()).device
    

    if torch.is_tensor(X):
        X_np = X.cpu().numpy()
    else:
        X_np = X
    
    if torch.is_tensor(y):
        y_np = y.cpu().numpy()
    else:
        y_np = y
    
    x_min, x_max = X_np[:, 0].min() - 1, X_np[:, 0].max() + 1
    y_min, y_max = X_np[:, 1].min() - 1, X_np[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()],
                        dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(grid)).cpu().numpy()
        preds = preds.reshape(xx.shape)

    fig = plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, preds, alpha=0.3, cmap='bwr')
    plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap='bwr', edgecolors='k')
    plt.title(relativePath)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
    save_figure(fig, relativePath)
    


def save_experiment_log(config, results, filename):
    ensure_dir(RESULT_DIR / "logs")
    
    log = {
        "config": config,
        "results": results
    }
    
    path = RESULT_DIR / "logs" / f"{filename}.json"
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=4, ensure_ascii=False)

def save_model(model, filename):
    ensure_dir(RESULT_DIR / "models")
    
    path = RESULT_DIR / "models" / f"{filename}.pt"
    torch.save(model.state_dict(), path)

def show_random_image(X, y):
    idx = random.randint(0, len(X) - 1)
    
    image = X[idx]
    label = y[idx]
    
    if image.shape[0] == 1:
        image = image.squeeze() 
    
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {int(label)}")
    plt.axis('off')
    plt.show()

def plot_multi_training_curves(histories, title, save_path):
    fig = plt.figure(figsize=(8, 5))

    for label, hist in histories.items():
        plt.plot(hist, label=label)

    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.show()
    save_figure(fig, save_path)

def plot_multi_accuracy_curves(histories, title, save_path):
    fig = plt.figure(figsize=(8, 5))

    for label, hist in histories.items():
        plt.plot(hist, label=label)

    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.grid(True)

    plt.show()
    save_figure(fig, save_path)