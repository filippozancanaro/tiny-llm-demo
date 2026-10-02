"""
model.py
--------
Definisce l'architettura del nostro "mini large language model".

Due soli livelli:
1. Embedding: trasforma l'id di una parola in un vettore che ne rappresenta
   il significato (inizialmente casuale, imparato durante il training).
2. Linear: trasforma quel vettore in un punteggio (logit) per ciascuna
   parola del vocabolario, cioè "quanto è probabile che sia lei la parola
   successiva".

Non c'è attention, non ci sono livelli nascosti multipli: è deliberatamente
il modello più semplice possibile che comunque impara qualcosa.
"""

import torch
import torch.nn as nn


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 8):
        super().__init__()

        # Ogni parola del vocabolario ha un vettore di "significato"
        # di lunghezza embedding_dim, inizializzato a caso e imparato
        # durante il training.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Trasforma il vettore di significato in un punteggio grezzo (logit)
        # per ciascuna delle vocab_size parole possibili.
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensore di id di parole, shape (batch,)
        ritorna: logit grezzi, shape (batch, vocab_size)

        Questi logit NON sono ancora probabilità: vanno passati per un
        softmax (fatto altrove, in fase di loss o di generazione) per
        diventare percentuali che sommano al 100%.
        """
        e = self.embedding(x)      # (batch, embedding_dim) - "significato"
        logits = self.linear(e)    # (batch, vocab_size)    - punteggi grezzi
        return logits
