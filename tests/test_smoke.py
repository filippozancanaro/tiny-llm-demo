"""
test_smoke.py
-------------
Test "di fumo": non verificano la qualità linguistica del modello (con un
corpus così piccolo non avrebbe senso), ma che l'intera pipeline funzioni
senza errori e che il training riduca effettivamente la loss.

Esecuzione (dalla root del progetto):
    pytest
"""

from src.data import CORPUS, END, build_dataset, build_vocab
from src.generate import generate, genera_da_frase
from src.train import train_model


def test_vocab_costruito_correttamente():
    stoi, itos = build_vocab(CORPUS)
    assert len(stoi) == len(itos)
    assert all(itos[i] == parola for parola, i in stoi.items())


def test_dataset_ha_la_forma_giusta():
    stoi, _ = build_vocab(CORPUS)
    X, Y = build_dataset(CORPUS, stoi)
    assert X.shape == Y.shape
    assert X.numel() > 0


def test_training_riduce_la_loss():
    _, _, _, storico = train_model(epochs=100, verbose=False)
    # La loss dell'ultima epoch deve essere significativamente più bassa
    # di quella iniziale: prova che il modello sta davvero imparando.
    assert storico[-1] < storico[0]


def test_generate_produce_solo_parole_del_vocabolario():
    modello, stoi, itos, _ = train_model(epochs=100, verbose=False)
    risultato = generate(modello, stoi, itos, "il", n_parole=5)
    parole_generate = risultato.split()
    assert all(parola in stoi for parola in parole_generate)


def test_generate_si_ferma_al_token_di_fine_frase():
    modello, stoi, itos, _ = train_model(epochs=200, verbose=False)
    # "divano" compare solo a fine frase, quindi il suo unico target di
    # training è END: con decoding greedy il modello deve fermarsi subito
    # invece di continuare a inventare parole.
    risultato = generate(modello, stoi, itos, "divano", n_parole=5, temperature=0)
    assert risultato == "divano"


def test_il_token_di_fine_frase_non_compare_mai_nelloutput():
    modello, stoi, itos, _ = train_model(epochs=100, verbose=False)
    risultato = generate(modello, stoi, itos, "il", n_parole=20)
    assert END not in risultato.split()


def test_generate_con_parola_sconosciuta_solleva_errore():
    modello, stoi, itos, _ = train_model(epochs=10, verbose=False)
    try:
        generate(modello, stoi, itos, "parolachenonesiste")
        assert False, "doveva sollevare ValueError"
    except ValueError:
        pass


def test_genera_da_frase_mantiene_la_frase_originale_come_prefisso():
    modello, stoi, itos, _ = train_model(epochs=100, verbose=False)
    risultato = genera_da_frase(modello, stoi, itos, "il gatto dorme", n_parole=3)
    # Il completamento deve sempre iniziare esattamente con la frase data in input.
    assert risultato.startswith("il gatto dorme")


def test_genera_da_frase_con_singola_parola_si_comporta_come_generate():
    modello, stoi, itos, _ = train_model(epochs=100, verbose=False)
    risultato = genera_da_frase(modello, stoi, itos, "il", n_parole=4)
    parole_generate = risultato.split()
    assert all(parola in stoi for parola in parole_generate)
