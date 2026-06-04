import torch
import torch.nn as nn
import torch.optim as optim
from plotting import plot_and_save_loss, plot_and_save_accuracy, plot_and_save_decision_boundary, save_model, save_experiment_log
import time
from datetime import datetime

import pandas as pd
import os
import uuid

def save_results_to_excel(new_row, file_path="results.xlsx"):
    new_df = pd.DataFrame([new_row])
    if os.path.exists(file_path):
        old_df = pd.read_excel(file_path)
        combined_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    combined_df.to_excel(file_path, index=False)

def find_convergence_epoch(loss_hist, threshold=0.001):
    for i in range(1, len(loss_hist)):
        if abs(loss_hist[i] - loss_hist[i-1]) < threshold:
            return i + 1
    return len(loss_hist)

def train_model(model, train_loader, test_loader, criterion, optimizer, criterion_name="BCEWithLogitsLoss", epochs=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_loss_hist, train_acc_hist = [], []
    test_loss_hist, test_acc_hist = [], []

    for epoch in range(epochs):
        # ==================== بخش آموزش (TRAIN) ====================
        model.train()
        total_loss, correct, total = 0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            if len(X.shape) > 2: 
                X = X.view(X.size(0), -1)

            optimizer.zero_grad()
            outputs = model(X)

            # مدیریت Loss بر اساس نوع معیار
            if "BCE" in criterion_name:
                target = y.float().view_as(outputs)
                loss = criterion(outputs, target)
            else:
                target = y.long()
                loss = criterion(outputs, target)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # --- محاسبه هوشمند دقت (Accuracy) ---
            if outputs.shape[1] == 1:
                # مخصوص Perceptron یا MLP تک خروجی
                preds = (torch.sigmoid(outputs).view(-1) > 0.5).long()
            else:
                # مخصوص MLP چند خروجی
                preds = torch.argmax(outputs, dim=1).long()

            y_true = y.view(-1).long()
            correct += (preds == y_true).sum().item()
            total += y_true.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = (correct / total) * 100
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)

        # ==================== بخش ارزیابی (TEST) ====================
        model.eval()
        t_loss_val, t_correct, t_total = 0, 0, 0
        
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                if len(X.shape) > 2: 
                    X = X.view(X.size(0), -1)
                
                outputs = model(X)

                # محاسبه Loss تست
                if "BCE" in criterion_name:
                    t_loss_val += criterion(outputs, y.float().view_as(outputs)).item()
                else:
                    t_loss_val += criterion(outputs, y.long()).item()

                # --- محاسبه هوشمند دقت در تست ---
                if outputs.shape[1] == 1:
                    ps = (torch.sigmoid(outputs).view(-1) > 0.5).long()
                else:
                    ps = torch.argmax(outputs, dim=1).long()

                y_true_t = y.view(-1).long()
                t_correct += (ps == y_true_t).sum().item()
                t_total += y_true_t.size(0)

        test_loss = t_loss_val / len(test_loader)
        test_acc = (t_correct / t_total) * 100
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)

        # نمایش وضعیت آموزش
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")

    return train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist

# def train_model(model, train_loader, test_loader, criterion, optimizer, criterion_name="BCEWithLogitsLoss", epochs=20):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device)

#     train_loss_hist, train_acc_hist = [], []
#     test_loss_hist, test_acc_hist = [], []

#     for epoch in range(epochs):
#         model.train()
#         total_loss, correct, total = 0, 0, 0

#         for X, y in train_loader:
#             X, y = X.to(device), y.to(device)
#             if len(X.shape) > 2: X = X.view(X.size(0), -1)

#             # 🔴 اصلاح خودکار برچسب‌ها (اگر 0 و 1 نباشند)
#             # این خط تضمین می‌کند که کوچکترین کلاس بشود 0 و بزرگترین بشود 1
#             if y.max() > 1:
#                 unique_y = torch.unique(y)
#                 if len(unique_y) == 2:
#                     low, high = unique_y.min(), unique_y.max()
#                     y = torch.where(y == low, 0, 1).to(y.dtype)

#             optimizer.zero_grad()
#             outputs = model(X)

#             if "BCE" in criterion_name:
#                 target = y.float().view_as(outputs)
#                 loss = criterion(outputs, target)
#             else:
#                 target = y.long()
#                 loss = criterion(outputs, target)

#             loss.backward()
#             # جلوگیری از انفجار گرادیان (Gradient Clipping)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
#             optimizer.step()
#             total_loss += loss.item()

#             # محاسبه Accuracy
#             if outputs.shape[1] == 1:
#                 preds = (torch.sigmoid(outputs).view(-1) > 0.5).long()
#             else:
#                 preds = torch.argmax(outputs, dim=1).long()

#             y_true = y.view(-1).long()
#             correct += (preds == y_true).sum().item()
#             total += y_true.size(0)

#         train_loss = total_loss / len(train_loader)
#         train_acc = (correct / total) * 100
        
#         # --- بخش تست (با همان منطق اصلاح برچسب) ---
#         model.eval()
#         t_loss_val, t_correct, t_total = 0, 0, 0
#         with torch.no_grad():
#             for X, y in test_loader:
#                 X, y = X.to(device), y.to(device)
#                 if len(X.shape) > 2: X = X.view(X.size(0), -1)
                
#                 # اصلاح برچسب در تست
#                 if y.max() > 1:
#                     unique_y = torch.unique(y)
#                     if len(unique_y) == 2:
#                         low, high = unique_y.min(), unique_y.max()
#                         y = torch.where(y == low, 0, 1).to(y.dtype)

#                 outputs = model(X)
#                 if "BCE" in criterion_name:
#                     t_loss_val += criterion(outputs, y.float().view_as(outputs)).item()
#                     ps = (torch.sigmoid(outputs).view(-1) > 0.5).long()
#                 else:
#                     t_loss_val += criterion(outputs, y.long()).item()
#                     ps = torch.argmax(outputs, dim=1).long()
                
#                 y_true_t = y.view(-1).long()
#                 t_correct += (ps == y_true_t).sum().item()
#                 t_total += y_true_t.size(0)

#         test_loss = t_loss_val / len(test_loader)
#         test_acc = (t_correct / t_total) * 100

#         if (epoch + 1) % 10 == 0 or epoch == 0:
#             print(f"Epoch {epoch+1:02d} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

#         train_loss_hist.append(train_loss)
#         train_acc_hist.append(train_acc)
#         test_loss_hist.append(test_loss)
#         test_acc_hist.append(test_acc)

#     return train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist

def run_experiment(
    model_class,
    train_loader,
    test_loader,
    lr,
    epochs,
    optimizer_name,
    criterion_name,
    dataset_name,
    plot_decision=False,
    plot_loss=False,
    plot_accuracy=False,
    save_model=False,
    X_vis=None,
    y_vis=None,
    model_kwargs={}
):
    start_time = time.time()
    model_name = model_class.__name__
    experiment_name =f"{model_name}_{dataset_name}_lr({lr})"

    sample_x, _ = next(iter(train_loader))
    actual_input_dim = sample_x[0].numel()

    model = model_class(input_dim=actual_input_dim, **model_kwargs)
    
    if criterion_name == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    elif criterion_name == "BCELoss":
        criterion = nn.BCELoss()
    elif criterion_name == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()

    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    

# Train
    train_loss, train_acc, test_loss, test_acc = train_model(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            criterion_name = criterion_name, # حتما ویرگول بگذارید
            epochs = epochs
        )
    

    if plot_loss:
        plot_and_save_loss(
            train_loss,
            test_loss,
            f"{experiment_name} - Loss",
            f"{experiment_name}_loss"
        )
    
    if plot_accuracy:
        plot_and_save_accuracy(
            train_acc,
            test_acc,
            f"{experiment_name} - Accuracy",
            f"{experiment_name}_accuracy"
        )

    if plot_decision and X_vis is not None and X_vis.shape[1] == 2:
        plot_and_save_decision_boundary(
            model,
            X_vis,
            y_vis,
            f"{experiment_name}_boundary"
        )
    
    if save_model:
        save_model(model, experiment_name)

    end_time = time.time()
    elapsed_time = end_time - start_time
    converged_epoch = find_convergence_epoch(test_loss)

    row = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_name": experiment_name,
        "dataset": dataset_name,
        "model": model_name,
        "optimizer": optimizer_name,
        "criterion": criterion_name,
        "learning_rate": lr,
        "epochs": epochs,
        "train_acc": train_acc[-1],
        "test_acc": test_acc[-1],
        "train_loss": train_loss[-1],
        "test_loss": test_loss[-1],
        "converged_epoch": converged_epoch,
        "elapsed_time": f"{elapsed_time * 1000} ms",
        "notes": ""
    }

    save_results_to_excel(row, "all_results.xlsx")    

    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }



