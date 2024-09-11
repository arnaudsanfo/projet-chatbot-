import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Charger le fichier JSON contenant les intentions et les schémas de phrases associés
with open('intents.json', 'r', encoding='utf8') as f:
    intents = json.load(f)

# Initialisation des listes pour les mots, les tags et les paires xy (entrée-sortie)
all_words = []
tags = []
xy = []

# Parcourir chaque intention et ses schémas de phrases associés
for intent in intents['intents']:
    tag = intent['tag']
    # Ajouter le tag à la liste des tags
    tags.append(tag)
    for pattern in intent['patterns']:
        # Tokeniser chaque mot dans la phrase
        w = tokenize(pattern)
        # Ajouter les mots à la liste de tous les mots
        all_words.extend(w)
        # Ajouter la paire xy (mots, tag) à la liste xy
        xy.append((w, tag))

# Racinisation et mise en minuscule de chaque mot
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# Supprimer les doublons et trier
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "schémas de phrases")
print(len(tags), "tags:", tags)
print(len(all_words), "mots racinisés uniques:", all_words)

# Créer les données d'entraînement
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: sac de mots pour chaque schéma de phrases
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss a besoin uniquement des étiquettes de classe, pas one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-paramètres
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

# Définition de la classe du jeu de données
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # Permettre l'indexation pour obtenir l'échantillon i-ème
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # Permettre de récupérer la taille du jeu de données
    def __len__(self):
        return self.n_samples

# Créer l'instance du jeu de données
dataset = ChatDataset()
# Créer le DataLoader pour gérer le chargement des données en mini-lots
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

# Utilisation du GPU si disponible, sinon utiliser le CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialisation du modèle
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Entraînement du modèle
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Passage avant
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Rétropropagation et optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoque [{epoch+1}/{num_epochs}], Perte: {loss.item():.4f}')

# Sauvegarde du modèle et des informations associées
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Entraînement terminé. Fichier sauvegardé sous {FILE}')