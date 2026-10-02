"""
data.py
--------
Gestisce tutto ciò che riguarda i dati "grezzi": il corpus di frasi,
la tokenizzazione (parola -> id numerico) e la costruzione del
dataset di training (coppie parola_attuale -> parola_successiva).

Nessuna logica di rete neurale qui: solo preparazione dei dati.
"""

import re
from typing import Dict, List, Tuple

import torch

# Corpus di partenza: volutamente piccolo e giocattolo.
# Puoi aggiungere frasi liberamente: più esempi = modello più "convinto".
# Regola semplice da rispettare: parole minuscole, separate da spazi, senza punteggiatura.
CORPUS: List[str] = [
    "il gatto dorme sul divano",
    "il cane dorme sul tappeto",
    "il gatto gioca con la palla",
    "il cane gioca con la palla",
    "il gatto mangia il pesce",
    "il cane mangia la carne",
]

# Token speciale di fine frase. Non è una parola del corpus: è un simbolo in più
# nel vocabolario che il modello impara a predire quando la frase è finita.
# Senza di lui il modello non avrebbe modo di "dire basta" e continuerebbe a
# generare parole all'infinito. È lo stesso meccanismo dei modelli veri
# (il token EOS / end-of-sequence).
# Le parentesi angolari lo rendono impossibile da confondere con una parola
# scritta dall'utente: normalizza_testo() rimuove la punteggiatura, quindi
# nessun input digitato dal vivo può mai produrre la stringa "<end>".
END: str = "<end>"


def normalizza_testo(testo: str) -> str:
    """
    Normalizza una frase prima di usarla nel corpus: minuscolo e rimozione
    della punteggiatura.

    Serve soprattutto per le frasi aggiunte a mano durante la demo
    (es. "Il gatto, dorme!") che altrimenti creerebbero token "sporchi"
    (diversi da quelli già nel vocabolario) e sporcherebbero il training.
    """
    return re.sub(r"[^\w\s]", "", testo.lower()).strip()


def build_vocab(corpus: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Costruisce il vocabolario a partire dal corpus.

    Ritorna due dizionari:
    - stoi (string to index): parola -> id numerico
    - itos (index to string): id numerico -> parola

    Oltre alle parole del corpus il vocabolario contiene sempre il token
    speciale END, che rappresenta la fine della frase.

    Il vocabolario è ordinato alfabeticamente solo per avere risultati
    riproducibili tra un run e l'altro (non ha nessun significato semantico).
    """
    parole = sorted(set(" ".join(corpus).split()) | {END})
    stoi = {parola: i for i, parola in enumerate(parole)}
    itos = {i: parola for parola, i in stoi.items()}
    return stoi, itos


def build_dataset(corpus: List[str], stoi: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trasforma il corpus in coppie di training (X, Y):
    per ogni frase, ogni parola diventa un esempio "predici la parola dopo di me".

    Ogni frase viene chiusa con il token END, che diventa la "risposta giusta"
    dell'ultima parola. Questo serve a due cose:
    1. il modello impara a segnalare quando la frase è finita
    2. anche l'ultima parola di una frase diventa un input di training
       (altrimenti parole come "divano", che compaiono solo in fondo, non
       verrebbero mai allenate e produrrebbero output casuali)

    Esempio:
        "il gatto dorme" -> (il -> gatto), (gatto -> dorme), (dorme -> <end>)

    Ritorna due tensori paralleli:
    - X: id della parola corrente (input)
    - Y: id della parola successiva (la "risposta giusta" da imparare)
    """
    coppie = []
    for frase in corpus:
        ids = [stoi[parola] for parola in frase.split()] + [stoi[END]]
        for i in range(len(ids) - 1):
            coppie.append((ids[i], ids[i + 1]))

    X = torch.tensor([c[0] for c in coppie])
    Y = torch.tensor([c[1] for c in coppie])
    return X, Y
