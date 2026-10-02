"""
train.py
--------
Contiene il ciclo di training vero e proprio (forward -> loss -> backward -> step)
e le funzioni di supporto per salvare/caricare un modello allenato su disco.

Eseguibile direttamente:
    python -m src.train

Allena il modello sul corpus di default e salva i pesi in checkpoint.pt,
nella cartella principale del progetto.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .data import CORPUS, build_dataset, build_vocab
from .model import TinyLM

# Il checkpoint viene salvato nella root del progetto, non dentro src/,
# così è facile trovarlo e non finisce per sbaglio dentro il pacchetto.
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoint.pt"


def train_model(
    corpus: List[str] = CORPUS,
    embedding_dim: int = 8,
    epochs: int = 200,
    lr: float = 0.01,
    verbose: bool = True,
) -> Tuple[TinyLM, Dict[str, int], Dict[int, str], List[float]]:
    """
    Allena un TinyLM da zero sul corpus fornito.

    Ogni epoch ripete lo stesso ciclo in 5 passi:
    1. azzera i gradienti del giro precedente
    2. forward pass: calcola le predizioni del modello
    3. calcola la loss (quanto abbiamo sbagliato)
    4. backward pass: calcola la correzione per ogni peso
    5. l'optimizer applica la correzione

    Ritorna: (modello allenato, stoi, itos, storico della loss per ogni epoch)
    """
    stoi, itos = build_vocab(corpus)
    X, Y = build_dataset(corpus, stoi)

    model = TinyLM(vocab_size=len(stoi), embedding_dim=embedding_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    storico_loss: List[float] = []

    for epoch in range(epochs):
        optimizer.zero_grad()          # 1. azzera i gradienti del giro precedente
        logits = model(X)              # 2. forward pass su tutto il dataset
        loss = criterion(logits, Y)    # 3. quanto abbiamo sbagliato
        loss.backward()                # 4. calcola la correzione per ogni peso
        optimizer.step()               # 5. applica la correzione

        storico_loss.append(loss.item())

        if verbose and epoch % 50 == 0:
            print(f"epoch {epoch:4d} | loss {loss.item():.4f}")

    if verbose:
        print(f"epoch {epochs - 1:4d} | loss {storico_loss[-1]:.4f} (finale)")

    return model, stoi, itos, storico_loss


def save_checkpoint(
    model: TinyLM,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    path: Path = CHECKPOINT_PATH,
) -> None:
    """
    Salva pesi del modello + vocabolario in un unico file .pt,
    così da poterli ricaricare senza dover ri-allenare da capo.
    """
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_size": len(stoi),
            "embedding_dim": model.embedding.embedding_dim,
            "stoi": stoi,
            "itos": itos,
        },
        path,
    )
    print(f"Checkpoint salvato in: {path}")


def load_checkpoint(path: Path = CHECKPOINT_PATH) -> Tuple[TinyLM, Dict[str, int], Dict[int, str]]:
    """
    Ricarica un modello allenato in precedenza, pronto per generare testo,
    senza dover rifare il training.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Nessun checkpoint trovato in {path}. "
            f"Esegui prima 'python -m src.train' per crearne uno."
        )

    checkpoint = torch.load(path, weights_only=False)
    model = TinyLM(vocab_size=checkpoint["vocab_size"], embedding_dim=checkpoint["embedding_dim"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()  # modalità inferenza: non serve calcolare gradienti
    return model, checkpoint["stoi"], checkpoint["itos"]


if __name__ == "__main__":
    modello, stoi, itos, storico = train_model()
    save_checkpoint(modello, stoi, itos)
