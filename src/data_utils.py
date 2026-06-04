import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def get_linear_data(test_size=0.3, noise_rate=0.0):
    X, y = make_blobs(n_samples=200, n_features=2,
                      centers=2, cluster_std=1.0, random_state=42)
    y = y.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if noise_rate > 0:
        y_train = y_train.copy()
        
        n_samples = len(y_train)
        n_noise = int(noise_rate * n_samples)
        
        noise_indices = np.random.choice(n_samples, n_noise, replace=False)
        
        y_train[noise_indices] = 1.0 - y_train[noise_indices]
        
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    print("Train shape:", X_train.shape)

    return X_train, X_test, y_train, y_test, train_loader, test_loader


def get_moons_data(test_size=0.3, noise=0.1):
    X, y = make_moons(n_samples=200, noise=noise, random_state=42)
    y = y.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # 2) Convert to torch tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # 3) Create dataset
    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    # 4) Create dataloader
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    print("Train shape:", X_train.shape)

    return X_train, X_test, y_train, y_test, train_loader, test_loader


from torch.utils.data import DataLoader, Subset

def get_mnist(classes=[0, 1], batch_size=32):
    transform = transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).float())

    trainset = torchvision.datasets.MNIST(
        root='../data', train=True, download=True, transform=transform)

    testset = torchvision.datasets.MNIST(
        root='../data', train=False, download=True, transform=transform)

    # همیشه فیلتر را بر اساس لیست کلاس‌های ورودی انجام بده
    if classes is not None:
        # پیدا کردن ایندکس‌هایی که برچسب آن‌ها در لیست مورد نظر ماست
        train_idx = [i for i, y in enumerate(trainset.targets) if y.item() in classes]
        test_idx = [i for i, y in enumerate(testset.targets) if y.item() in classes]
        
        trainset = Subset(trainset, train_idx)
        testset = Subset(testset, test_idx)

    def extract_data(subset_ds):
        loader = DataLoader(subset_ds, batch_size=len(subset_ds), shuffle=False, num_workers=0)
        X, y = next(iter(loader))
        return X.numpy(), y.numpy().astype(np.float32)

    X_train, y_train = extract_data(trainset)
    X_test, y_test = extract_data(testset)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    print(f"Loaded classes {classes}. Train shape: {X_train.shape}")
    return X_train, X_test, y_train, y_test, train_loader, test_loader

from torch.utils.data import Dataset

class MapLabelsDataset(Dataset):
    def __init__(self, subset, class_map):
        self.subset = subset
        self.class_map = class_map
    def __getitem__(self, index):
        x, y = self.subset[index]
        # تبدیل برچسب (مثلا 3 به 0 و 8 به 1)
        return x, self.class_map[int(y)]
    def __len__(self):
        return len(self.subset)

# def get_normalized_mnist(classes=[3, 8], batch_size=32):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])

#     trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
#     testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)

#     # پیدا کردن ایندکس‌ها
#     train_idx = [i for i, y in enumerate(trainset.targets) if y.item() in classes]
#     test_idx = [i for i, y in enumerate(testset.targets) if y.item() in classes]

#     # ساخت نگاشت: {classes[0]: 0, classes[1]: 1}
#     class_map = {c: i for i, c in enumerate(classes)}

#     # استفاده از کلاس کمکی برای تغییر برچسب‌ها به 0 و 1
#     train_sub = Subset(trainset, train_idx)
#     test_sub = Subset(testset, test_idx)
    
#     train_final = MapLabelsDataset(train_sub, class_map)
#     test_final = MapLabelsDataset(test_sub, class_map)

#     def extract_data(ds):
#         loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
#         X, y = next(iter(loader))
#         X = X.view(X.size(0), -1)
#         return X.numpy(), y.numpy()

#     X_train, y_train = extract_data(train_final)
#     X_test, y_test = extract_data(test_final)

#     train_loader = DataLoader(train_final, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_final, batch_size=batch_size, shuffle=False)

#     print(f"✅ Success! Classes {classes} are now 0 and 1.")
#     return X_train, X_test, y_train, y_test, train_loader, test_loader

# def get_normalized_mnist(classes=[3, 8], batch_size=32):
#     # ۱. تعریف تبدیل‌ها (نرمال‌سازی پیکسل‌ها بین 0 و 1 و سپس استانداردسازی)
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])

#     # ۲. دانلود دیتاسیت
#     trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
#     testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)

#     # ۳. پیدا کردن ایندکس‌های مربوط به کلاس‌های انتخابی (مثلاً 3 و 8)
#     train_mask = (trainset.targets == classes[0]) | (trainset.targets == classes[1])
#     test_mask = (testset.targets == classes[0]) | (testset.targets == classes[1])

#     # ۴. استخراج داده‌ها و برچسب‌های فیلتر شده
#     X_train = trainset.data[train_mask]
#     y_train = trainset.targets[train_mask]
    
#     X_test = testset.data[test_mask]
#     y_test = testset.targets[test_mask]

#     # ۵. 🔴 بخش حیاتی: نگاشت برچسب‌ها به 0 و 1 🔴
#     # کلاس اول (مثلا 3) می‌شود 0 و کلاس دوم (مثلا 8) می‌شود 1
#     y_train = torch.where(y_train == classes[0], torch.tensor(0), torch.tensor(1)).float()
#     y_test = torch.where(y_test == classes[0], torch.tensor(0), torch.tensor(1)).float()

#     # ۶. ساخت دیتاسیت‌های جدید از داده‌های اصلاح شده
#     # نکته: ToTensor و Normalize را اینجا دستی اعمال می‌کنیم چون از Dataset اصلی جدا شدیم
#     X_train = X_train.float() / 255.0
#     X_test = X_test.float() / 255.0
    
#     # اعمال نرمال‌سازی استاندارد MNIST (میانگین و انحراف معیار)
#     X_train = (X_train - 0.1307) / 0.3081
#     X_test = (X_test - 0.1307) / 0.3081

#     # ۷. ساخت TensorDataset و DataLoader
#     from torch.utils.data import TensorDataset
    
#     # فلت کردن تصاویر (28x28 -> 784)
#     train_ds = TensorDataset(X_train.view(-1, 784), y_train)
#     test_ds = TensorDataset(X_test.view(-1, 784), y_test)

#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

#     print(f"✅ MNIST Data Ready! Classes {classes} mapped to [0, 1]")
#     print(f"Train samples: {len(y_train)} | Unique labels in train: {torch.unique(y_train)}")
    
#     return X_train.numpy(), X_test.numpy(), y_train.numpy(), y_test.numpy(), train_loader, test_loader

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def get_normalized_mnist(classes=[0, 1], batch_size=32):
    """
    بارگذاری و پیش‌پردازش MNIST با قابلیت انتخاب دو کلاس دلخواه
    و تبدیل برچسب‌ها به 0/1 برای طبقه‌بندی دودویی.

    Parameters:
        classes (list): لیست دو کلاس مورد نظر (مثلاً [3, 8])
        batch_size (int): اندازه‌ی batch

    Returns:
        X_train, X_test : آرایه‌های numpy از تصاویر (تعداد × 784)
        y_train, y_test : آرایه‌های numpy از برچسب‌های 0/1
        train_loader, test_loader : DataLoader های PyTorch
    """
    # نرمال‌سازی استاندارد MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # بارگذاری مجموعه‌های کامل
    full_train = torchvision.datasets.MNIST(
        root='../data', train=True, download=True, transform=transform)
    full_test = torchvision.datasets.MNIST(
        root='../data', train=False, download=True, transform=transform)

    # ---------- تبدیل برچسب‌ها به 0/1 فقط برای دو کلاس خواسته شده ----------
    if classes is not None and len(classes) == 2:
        c0, c1 = classes[0], classes[1]

        # ماسک نمونه‌های متعلق به کلاس‌های انتخاب شده
        train_mask = (full_train.targets == c0) | (full_train.targets == c1)
        test_mask  = (full_test.targets == c0) | (full_test.targets == c1)

        # درجا: تبدیل c0 -> 0 , c1 -> 1
        full_train.targets[train_mask] = (full_train.targets[train_mask] == c1).long()
        full_test.targets[test_mask]   = (full_test.targets[test_mask] == c1).long()

        # اندیس‌های همین نمونه‌ها برای ساخت Subset
        train_indices = torch.nonzero(train_mask, as_tuple=False).squeeze().tolist()
        test_indices  = torch.nonzero(test_mask, as_tuple=False).squeeze().tolist()

        trainset = Subset(full_train, train_indices)
        testset  = Subset(full_test, test_indices)
    else:
        # اگر تعداد کلاس‌ها دو نبود (مثلاً یک کلاس یا None)، بدون تبدیل نگه دار
        if classes is not None:
            train_idx = [i for i, y in enumerate(full_train.targets) if y.item() in classes]
            test_idx  = [i for i, y in enumerate(full_test.targets)  if y.item() in classes]
            trainset = Subset(full_train, train_idx)
            testset  = Subset(full_test, test_idx)
        else:
            trainset = full_train
            testset  = full_test

    # ---------- استخراج کل داده‌ها به صورت آرایه‌ی numpy (برای رسم و مصارف دیگر) ----------
    def extract_numpy(subset):
        loader = DataLoader(subset, batch_size=len(subset), shuffle=False, num_workers=0)
        X, y = next(iter(loader))
        return X.numpy(), y.numpy()

    X_train, y_train = extract_numpy(trainset)
    X_test, y_test   = extract_numpy(testset)

    # ---------- DataLoader نهایی ----------
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(testset, batch_size=batch_size, shuffle=False)

    print(f"✅ بارگذاری کلاس‌های {classes}. تعداد نمونه‌های آموزش: {X_train.shape[0]}")
    print(f"   برچسب‌های منحصربه‌فرد: {np.unique(y_train)}")
    return X_train, X_test, y_train, y_test, train_loader, test_loader


from torch.utils.data import Subset

def get_limited_mnist(classes=[0, 1], batch_size=32, n_train_samples=None, n_test_sample=None):
    # تبدیل عکس به تنسور و فلت کردن (MNIST 28x28 است)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    if classes is not None:
        train_idx = [i for i, y in enumerate(trainset.targets) if y.item() in classes]
        test_idx = [i for i, y in enumerate(testset.targets) if y.item() in classes]
        
        # محدود کردن نمونه‌های آموزش اگر n_train_samples مقداردهی شده باشد
        if n_train_samples is not None:
            train_idx = train_idx[:n_train_samples]
        
        if n_test_sample is not None:
            test_idx = test_idx[:n_test_sample]

        trainset = Subset(trainset, train_idx)
        testset = Subset(testset, test_idx)

    # استخراج داده‌ها به صورت Numpy برای سازگاری با توابع قبلی شما
    def extract_data(subset_ds):
        loader = DataLoader(subset_ds, batch_size=len(subset_ds), shuffle=False)
        X, y = next(iter(loader))
        X = X.view(X.size(0), -1)
        return X.numpy(), y.numpy().astype(np.float32)

    X_train, y_train = extract_data(trainset)
    X_test, y_test = extract_data(testset)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    print(f"Loaded MNIST classes {classes}. Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, train_loader, test_loader