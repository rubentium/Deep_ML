import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
import time


## MS2


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super(MLP, self).__init__()
        # the layers 
        self.mlp0 = nn.Linear(input_size, 512)
        self.mlp1 = nn.Linear(512, 128)
        self.mlp2 = nn.Linear(128, 64)
        self.mlp3 = nn.Linear(64, 32)
        self.mlp4 = nn.Linear(32, n_classes)


    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = nn.ReLU()(self.mlp0(x))
        x = nn.ReLU()(self.mlp1(x))
        x = nn.ReLU()(self.mlp2(x))
        x = nn.ReLU()(self.mlp3(x))
        preds = nn.ReLU()(self.mlp4(x))
        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super(CNN, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(input_channels, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.perceptron = nn.Sequential(
            nn.Linear(784, 300),
            nn.ReLU(),
            nn.Linear(300, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
            nn.ReLU()
        )


    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = self.convolution(x)
        x = x.flatten(1)
        preds = self.perceptron(x)
        return preds

class MSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MSA, self).__init__()
        self.d = d
        self.n_heads = n_heads
        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        d_head = d // n_heads
        self.d_head = d_head
        self.qkv = nn.Linear(d, 3 * d)  # Single linear layer for Q, K, V

    def forward(self, x):
        batch_size, seq_len, d = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, self.n_heads, 3 * self.d_head)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.permute(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, d_head)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        attention_scores = (q @ k.transpose(-2, -1)) / np.sqrt(self.d_head)
        attention = F.softmax(attention_scores, dim=-1)
        out = attention @ v
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, d)
        return out

class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out

class MyViT(nn.Module):
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        super(MyViT, self).__init__()
        self.chw = chw
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        patch_size = chw[1] // n_patches
        self.patch_size = patch_size
        self.input_d = chw[0] * patch_size * patch_size
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_d))
        self.positional_embeddings = nn.Parameter(self.get_positional_embeddings(n_patches ** 2 + 1, hidden_d))
        self.blocks = nn.ModuleList([ViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def get_positional_embeddings(self, sequence_length, d):
        result = torch.zeros(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                if j % 2 == 0:
                    result[i, j] = np.sin(i / (10000 ** ((2 * j) / d)))
                else:
                    result[i, j] = np.cos(i / (10000 ** ((2 * j) / d)))
        return result

    def patchify(self, images, n_patches):
        n, c, h, w = images.shape
        patch_size = h // n_patches
        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(n, c, n_patches * n_patches, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(n, n_patches * n_patches, -1)
        return patches

    def forward(self, images):
        n = images.size(0)
        patches = self.patchify(images, self.n_patches)
        tokens = self.linear_mapper(patches)
        tokens = torch.cat((self.class_token.expand(n, -1, -1), tokens), dim=1)
        out = tokens + self.positional_embeddings.unsqueeze(0).expand(n, -1, -1)
        for block in self.blocks:
            out = block(out)
        out = out[:, 0]
        out = self.mlp(out)
        return out

class Trainer(object):
    def __init__(self, model, lr, epochs, batch_size, device = torch.device("cpu")):
        self.lr = lr
        self.epochs = epochs
        self.model = model.to(device)
        self.batch_size = batch_size
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def train_all(self, dataloader):
        start = time.time()
        times = []
        for ep in range(self.epochs):
            elapsed = self.train_one_epoch(dataloader, ep)
            times.append(elapsed)

        minutes = int(time.time() - start) // 60
        seconds = int(time.time() - start) % 60
        print(f"Total training time: {minutes:02}:{seconds:02}")
        print(f"Average time per epoch: {np.mean(times):.1f}s")

    def train_one_epoch(self, dataloader, ep):
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        start = time.time()
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            print(f"Batch [{i+1}/{len(dataloader)}]", end="\r")
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Compute loss and update weights
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            # Compute accuracy
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

        accuracy = running_corrects / total_samples * 100
        elapsed = time.time() - start
        print(f"Epoch [{ep+1}/{self.epochs}], Loss: {running_loss/len(dataloader)}, Accuracy: {accuracy:.2f}%, Time: {elapsed:.1f}s")
        return elapsed

    def predict_torch(self, dataloader):
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs[0].to(self.device)
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds)
        pred_labels = torch.cat(all_preds)
        return pred_labels

    def fit(self, training_data, training_labels):
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                      torch.from_numpy(training_labels).long())
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.train_all(train_dataloader)
        return self.predict(training_data)

    def predict(self, test_data):
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        pred_labels = self.predict_torch(test_dataloader)
        return pred_labels.cpu().numpy()