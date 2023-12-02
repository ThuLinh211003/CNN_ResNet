import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms

def load_fashion_mnist_csv(batch_size, data_path='/Users/dinhbuithulinh/Desktop/VCCorp/Image_Processing/Datasets/archive/fashion-mnist_train.csv', test_size=0.2, random_state=42):
    # Load data from CSV file
    df = pd.read_csv(data_path)

    # Separate features and labels
    X = df.iloc[:, 1:].values.astype('float32')
    y = df.iloc[:, 0].values.astype('int64')

    # Normalize features to range [0, 1]
    X /= 255.0

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Convert to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).reshape(-1, 1, 28, 28)
    y_train_tensor = torch.from_numpy(y_train)
    X_test_tensor = torch.from_numpy(X_test).reshape(-1, 1, 28, 28)
    y_test_tensor = torch.from_numpy(y_test)

    # Create PyTorch datasets and loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_transform():
    return transforms.Compose([transforms.ToTensor()])
