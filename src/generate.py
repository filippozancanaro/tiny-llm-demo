"""
generate.py
-----------
Inferenza: usa un modello già allenato per generare testo, una parola alla volta,
in modo autoregressivo (ogni parola generata diventa l'input del passo successivo).
Concettualmente è lo stesso meccanismo con cui ChatGPT genera le sue risposte.

Eseguibile direttamente:
    python -m src.generate il
    python -m src.generate gatto 8

Carica checkpoint.pt e genera una frase a partire dalla parola data.
"""

import sys
from typing import Dict

import torch

from .data import END
from .model import TinyLM
from .train import load_checkpoint

# Temperature di default, volutamente sotto 1.0: con un corpus così piccolo
# il campionamento "onesto" (1.0) pesca troppo spesso continuazioni rare e
# le frasi escono sgangherate. A 0.6 il modello segue di più i pattern che ha
# davvero imparato e l'output è più leggibile.
# È un parametro, non una verità: alzandolo (es. 1.5) il modello
# inizia a "delirare".
TEMPERATURE_DEFAULT: float = 0.6


def generate(
    model: TinyLM,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    parola_iniziale: str,
    n_parole: int = 5,
    temperature: float = TEMPERATURE_DEFAULT,
) -> str:
    """
    Genera una sequenza di parole a partire da parola_iniziale.

    n_parole è il numero MASSIMO di parole generate: se il modello predice il
    token di fine frase (END) ci fermiamo prima, così la frase si chiude da sola
    invece di continuare con parole a caso. Il token END non compare mai
    nell'output: è un segnale interno, non una parola.

    temperature controlla quanto il campionamento è "audace" (default 0.6):
    - valori bassi (es. 0.3) -> il modello sceglie quasi sempre la parola
      più probabile (output più prevedibile e ripetitivo)
    - valori alti (es. 1.5) -> più varietà, anche parole meno probabili
      (output più "creativo" ma anche più incoerente)
    - 0 (o valori negativi) -> nessun campionamento: si prende sempre e
      solo la parola più probabile ("greedy decoding", output deterministico)

    Se parola_iniziale non è nel vocabolario, viene sollevato un errore
    chiaro invece di un crash poco comprensibile.
    """
    if parola_iniziale not in stoi:
        # END è escluso dal suggerimento: è un token interno, non una parola
        # che abbia senso passare come punto di partenza.
        raise ValueError(
            f"'{parola_iniziale}' non è nel vocabolario. "
            f"Parole disponibili: {sorted(p for p in stoi if p != END)}"
        )

    model.eval()
    parola = parola_iniziale
    frase = [parola]

    with torch.no_grad():  # in inferenza non ci serve calcolare gradienti
        for _ in range(n_parole):
            x = torch.tensor([stoi[parola]])
            logits = model(x)

            if temperature <= 0:
                # temperature 0 (o negativa) non ha senso per un softmax:
                # la interpretiamo come "prendi sempre la parola più probabile".
                prossimo_id = int(torch.argmax(logits, dim=-1).item())
            else:
                probabilita = torch.softmax(logits / temperature, dim=-1)
                # campioniamo la prossima parola in base alle probabilità,
                # invece di prendere sempre e solo la più probabile:
                # è il motivo per cui l'output cambia leggermente ad ogni run.
                prossimo_id = int(torch.multinomial(probabilita, num_samples=1).item())

            parola = itos[prossimo_id]
            if parola == END:
                break

            frase.append(parola)

    return " ".join(frase)


def genera_da_frase(
    model: TinyLM,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    frase_iniziale: str,
    n_parole: int = 5,
    temperature: float = TEMPERATURE_DEFAULT,
) -> str:
    """
    Completa una frase intera (non solo una parola), es. "il gatto dorme" -> "il gatto dorme sul divano".

    ATTENZIONE - limite architetturale importante da avere chiaro:
    il nostro modello guarda solo UNA parola di contesto per predire la
    successiva (è un modello "bigram"). Questa funzione quindi:
    1. prende la frase in input così com'è
    2. usa SOLO la sua ultima parola come punto di partenza per generate()
    3. incolla il resto della frase originale davanti al risultato

    Il modello non sta "leggendo" l'intera frase: ignora tutto tranne
    l'ultima parola. Il risultato sembra plausibile perché nel nostro
    piccolo corpus le frasi condividono spesso lo stesso finale, ma è
    un effetto collaterale, non vera comprensione del contesto lungo.
    Per un modello che usa davvero tutta la frase come contesto serve
    l'attention (fuori dallo scope di questo mini-progetto).

    Se la frase è una singola parola, si comporta esattamente come generate().
    """
    parole_input = frase_iniziale.strip().split()
    if not parole_input:
        raise ValueError("La frase iniziale è vuota.")

    ultima_parola = parole_input[-1]

    continuazione = generate(
        model, stoi, itos, ultima_parola, n_parole=n_parole, temperature=temperature
    )
    # continuazione include già ultima_parola come prima parola: la scartiamo
    # per non duplicarla, e la sostituiamo con la frase originale completa.
    parole_generate = continuazione.split()[1:]

    return " ".join(parole_input + parole_generate)


if __name__ == "__main__":
    # Tutti gli argomenti dopo il nome dello script vengono uniti in una
    # singola frase, così sia "python -m src.generate il" (una parola)
    # sia "python -m src.generate il gatto dorme" (frase intera) funzionano.
    argomenti = sys.argv[1:]
    frase_o_parola = " ".join(argomenti) if argomenti else "il"
    n_parole = 5

    # Se l'ultimo argomento è un numero, lo trattiamo come n_parole
    # (es. "python -m src.generate il gatto dorme 8" -> genera 8 parole).
    if argomenti and argomenti[-1].isdigit():
        n_parole = int(argomenti[-1])
        frase_o_parola = " ".join(argomenti[:-1]) if len(argomenti) > 1 else "il"

    modello, stoi, itos = load_checkpoint()

    if " " in frase_o_parola:
        print(genera_da_frase(modello, stoi, itos, frase_o_parola, n_parole=n_parole))
    else:
        print(generate(modello, stoi, itos, frase_o_parola, n_parole=n_parole))
