import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.MIL.ABMIL import get_attn_module
from torch.utils.data import DataLoader, Dataset

class AttriMIL(nn.Module): 
    '''
    Multi-Branch ABMIL with Hyperparameter Tuning
    '''
    def __init__(self, n_classes=2, dim=1024):
        super().__init__()
        self.adaptor = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim)
        )

        attention = []
        classifiers = [nn.Linear(dim, 1) for _ in range(n_classes)]
        for _ in range(n_classes):
            attention.append(get_attn_module(embed_size=dim, hidden_size=dim // 2, att_branches=1))
        
        self.attention_nets = nn.ModuleList(attention)
        self.classifiers = nn.ModuleList(classifiers)
        self.n_classes = n_classes
        self.bias = nn.Parameter(torch.zeros(n_classes), requires_grad=True)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, h):
        h = h.squeeze(0)
        h = h + self.adaptor(h)

        A_raw = torch.empty(self.n_classes, h.size(0), device=h.device)  # N x 1
        instance_score = torch.empty(1, self.n_classes, h.size(0), device=h.device)

        for c in range(self.n_classes):
            A, h_attn = self.attention_nets[c](h)
            A = torch.transpose(A, 1, 0)  # 1 x N
            A_raw[c] = A
            instance_score[0, c] = self.classifiers[c](h_attn)[:, 0]

        attribute_score = torch.empty(1, self.n_classes, h.size(0), device=h.device)
        for c in range(self.n_classes):
            attribute_score[0, c] = instance_score[0, c] * torch.exp(A_raw[c])

        logits = torch.empty(1, self.n_classes, device=h.device)
        for c in range(self.n_classes):
            logits[0, c] = torch.sum(attribute_score[0, c], dim=-1) / torch.sum(torch.exp(A_raw[c]), dim=-1) + self.bias[c]

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {}
        return logits, Y_prob, Y_hat, attribute_score, results_dict

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, _, Y_hat, _, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.long().unsqueeze(0)
        logits, Y_prob, _, A, _ = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        loss = self.loss_fn(logits, Y)
        return loss, A

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Hyperparameter Tuning- not certain
# the parameters may change to better results

def train_model(model, dataloader, lr=1e-4, weight_decay=1e-5, epochs=10, gradient_clip=1.0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for X, Y in dataloader:
            X, Y = X.to(model.adaptor[0].weight.device), Y.to(model.adaptor[0].weight.device)

            optimizer.zero_grad()
            loss, _ = model.calculate_objective(X, Y)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            epoch_loss += loss.item()

        # Scheduler Step
        scheduler.step(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

# Instantiate and Train
if __name__ == "__main__":
    model = AttriMIL(n_classes=2, dim=1024)