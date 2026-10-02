# Tiny LLM Demo

Progetto demo del talk *"Come funziona davvero un LLM"*.
Un mini-modello di *text completion* in PyTorch: tokenizzazione → embedding →
training → inferenza, sullo stesso principio (semplificato) di un vero LLM.

---

## 1. Struttura del progetto

```
tiny-llm-demo/
├── README.md              <- questo file
├── requirements.txt        <- dipendenze
├── .gitignore
├── checkpoint.pt            <- creato dopo il training (non versionato)
├── src/
│   ├── __init__.py
│   ├── data.py             <- corpus, vocabolario, dataset
│   ├── model.py            <- architettura TinyLM (Embedding + Linear)
│   ├── train.py            <- ciclo di training + salva/carica checkpoint
│   ├── generate.py         <- inferenza / generazione testo
│   └── demo.py             <- demo interattiva a menu
├── tests/
│   ├── __init__.py
│   └── test_smoke.py       <- test di fumo sull'intera pipeline
├── docs/
│   ├── guida-al-codice.md        <- sorgente del PDF "guida al codice"
│   ├── principi-rete-neurale.md  <- sorgente del PDF "principi della rete neurale"
│   └── build_pdf.py              <- rigenera i due PDF qui sotto
├── tiny-llm-demo-guida-al-codice.pdf
└── tiny-llm-demo-principi-rete-neurale.pdf
```

**Documentazione di approfondimento:** i due PDF nella root (guida al codice
per chi non conosce Python; principi della rete neurale per il "perché
funziona") si rigenerano dai sorgenti Markdown in `docs/` con
`pip install markdown pygments` e `python docs/build_pdf.py` (serve Chrome o
Edge installato).

**Come leggere il codice, in ordine consigliato:** `data.py` → `model.py` →
`train.py` → `generate.py` → `demo.py`.

---

## 2. Prerequisiti

- Python 3.9 o superiore (`python3 --version` per controllare)
- `pip` funzionante
- Nessuna GPU richiesta: il modello è così piccolo che gira in secondi su CPU

---

## 3. Setup da zero

```bash
# 1. Entra nella cartella del progetto
cd tiny-llm-demo

# 2. Crea un virtual environment (isola le dipendenze da quelle di sistema)
python3 -m venv .venv

# 3. Attiva il virtual environment
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows (PowerShell/cmd)

# 4. Installa le dipendenze
pip install -r requirements.txt
```

Per uscire dal virtual environment in qualsiasi momento: `deactivate`.

---

## 4. Comandi principali

Tutti i comandi vanno lanciati dalla **root del progetto** (la cartella
`tiny-llm-demo/`), con il virtual environment attivo.

### Allenare il modello

```bash
python -m src.train
```

Allena il TinyLM sul corpus di default (200 epoch), stampa la loss ogni 50
epoch e salva i pesi in `checkpoint.pt`.

### Generare testo (dopo aver allenato)

```bash
python -m src.generate il
python -m src.generate gatto 8              # al massimo 8 parole invece delle 5 di default
python -m src.generate il gatto dorme       # completa una frase intera (vedi nota sotto)
python -m src.generate il gatto dorme 8     # frase intera + numero massimo di parole
```

Se non hai ancora allenato nulla, otterrai un errore chiaro che ti chiede
di lanciare prima `python -m src.train`.

**Nota sul token di fine frase:** il numero di parole è un *massimo*, non una
quantità fissa. Il vocabolario contiene un token speciale `<end>` (definito in
`src/data.py`) che chiude ogni frase del corpus durante il training: il modello
impara così a predire "qui la frase è finita", e `generate()` si ferma appena lo
incontra. È lo stesso meccanismo del token EOS dei modelli veri, ed è il motivo
per cui `il gatto dorme` produce `il gatto dorme sul divano` e non continua con
parole a caso. Senza questo token il modello non avrebbe modo di dire "basta".

**Nota su "completare una frase intera":** il modello guarda solo l'ultima
parola per predire la successiva (è un modello "bigram"). Quando dai in
input una frase con più parole, `genera_da_frase()` (in `src/generate.py`)
prende solo l'ultima parola come punto di partenza
e incolla il resto della frase davanti al risultato — non è vera "lettura"
del contesto lungo, è un trucco per rendere l'output più naturale da vedere
in demo. È un ottimo spunto per introdurre il limite che l'attention risolve.

### Demo interattiva

```bash
python -m src.demo
```

Menu a 5 opzioni: mostra il corpus, aggiungi una frase e ri-allena al volo,
genera testo, reset al corpus originale, esci (con opzione di salvataggio).

### Eseguire i test

```bash
pytest
```

oppure, per un output più dettagliato:

```bash
pytest -v
```

I test verificano che vocabolario e dataset abbiano la forma corretta, che
il training riduca davvero la loss, che la generazione produca sempre parole
del vocabolario (mai output "impossibili") e che si fermi correttamente sul
token di fine frase.

---

## 5. Reset completo del progetto

Se vuoi ripartire da zero (es. dopo aver rovinato qualcosa durante prove
ripetute della demo):

```bash
# Rimuovi il modello allenato
rm checkpoint.pt

# (opzionale) Ricrea anche il virtual environment da zero
deactivate
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Il corpus originale è definito staticamente in `src/data.py` (variabile
`CORPUS`): la demo interattiva non lo modifica mai in modo permanente, quindi
un semplice riavvio di `python -m src.demo` equivale già a un reset dei dati.

---

## 6. Estensioni facili

- **Aggiungere frasi al corpus base:** modifica la lista `CORPUS` in
  `src/data.py`.
- **Cambiare la dimensione dell'embedding:** `train_model(embedding_dim=16)`
  in `src/train.py` — più dimensioni, più "spazio" per rappresentare
  sfumature di significato (ma serve più corpus per sfruttarle davvero).
- **Cambiare velocità/durata del training:** parametri `epochs` e `lr` di
  `train_model(...)`.
- **Rendere la generazione più/meno "audace":** parametro `temperature` in
  `generate(...)`, default 0.6 (`TEMPERATURE_DEFAULT` in `src/generate.py`).
  0.3 = prevedibile, 1.5 = più creativo e incoerente. Alzarlo sullo
  stesso prompt è un buon modo per vedere il modello che "delira".

---

## 7. Problemi comuni

| Problema | Causa probabile | Soluzione |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | Stai lanciando i comandi da una cartella sbagliata, o non usi `-m` | Esegui sempre dalla root del progetto con `python -m src.nome_modulo` |
| `FileNotFoundError: Nessun checkpoint trovato` | Non hai ancora allenato il modello | Lancia `python -m src.train` prima di `generate` |
| Errore nell'installazione di `torch` | Versione Python non supportata o pip datato | Aggiorna pip (`pip install --upgrade pip`) e verifica la versione Python (>=3.9) |
| La demo genera frasi senza senso | Corpus troppo piccolo, oppure il limite bigram: il modello vede solo l'ultima parola, quindi `il cane gioca` può diventare `il cane gioca con la carne` | È normale ed è anche didattico: con 6-10 frasi il modello impara solo pattern molto grezzi, e il contesto oltre l'ultima parola è proprio ciò che l'attention risolve |
| `UserWarning: Failed to initialize NumPy` ad ogni comando | `numpy` non installato: torch lo cerca e avvisa | `pip install -r requirements.txt` (numpy è tra le dipendenze proprio per evitare questo warning) |
| `pytest` non trovato | Virtual environment non attivo, o dipendenze non installate | Attiva `.venv` e rilancia `pip install -r requirements.txt` |
