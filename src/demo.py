"""
demo.py
-------
Demo interattiva a riga di comando: permette di

1. vedere il corpus attuale
2. aggiungere una nuova frase e ri-allenare in pochi secondi
3. generare testo con il modello aggiornato
4. tornare al corpus originale in qualsiasi momento (reset)

Avvio:
    python -m src.demo
"""

from .data import CORPUS, normalizza_testo
from .generate import generate, genera_da_frase
from .train import save_checkpoint, train_model


def stampa_menu() -> None:
    print("\n=== Demo: Tiny Language Model ===")
    print("1. Mostra il corpus attuale")
    print("2. Aggiungi una frase e ri-allena")
    print("3. Genera testo")
    print("4. Reset al corpus originale e ri-allena")
    print("5. Esci")


def main() -> None:
    # Copia modificabile del corpus: non tocchiamo mai CORPUS originale,
    # così il reset (opzione 4) è sempre affidabile.
    corpus_corrente = list(CORPUS)

    print("Alleno il modello iniziale sul corpus di base...")
    modello, stoi, itos, _ = train_model(corpus_corrente, verbose=False)
    print("Pronto.")

    try:
        while True:
            stampa_menu()
            scelta = input("Scegli un'opzione (1-5): ").strip()

            if scelta == "1":
                for frase in corpus_corrente:
                    print(f"  - {frase}")

            elif scelta == "2":
                frase_grezza = input(
                    "Scrivi una nuova frase (es. 'il gatto dorme sul cuscino'): "
                ).strip()
                nuova_frase = normalizza_testo(frase_grezza)
                if not nuova_frase:
                    print("Frase vuota, ignorata.")
                    continue

                corpus_corrente.append(nuova_frase)
                print("Ri-alleno il modello con la frase aggiunta...")
                modello, stoi, itos, storico = train_model(corpus_corrente, verbose=False)
                print(f"Fatto. Loss finale: {storico[-1]:.4f}")

            elif scelta == "3":
                testo = input("Parola o frase di partenza (es. 'gatto' o 'il gatto dorme'): ").strip()
                try:
                    # Se contiene uno spazio è una frase intera: usiamo
                    # genera_da_frase, che completa in base all'ultima parola.
                    # Con una sola parola si comporta come generate() normale.
                    if " " in testo:
                        risultato = genera_da_frase(modello, stoi, itos, testo, n_parole=6)
                    else:
                        risultato = generate(modello, stoi, itos, testo, n_parole=6)
                    print(f"-> {risultato}")
                except ValueError as e:
                    print(f"Errore: {e}")

            elif scelta == "4":
                corpus_corrente = list(CORPUS)
                print("Reset al corpus originale. Ri-alleno...")
                modello, stoi, itos, _ = train_model(corpus_corrente, verbose=False)
                print("Fatto.")

            elif scelta == "5":
                salva = input("Vuoi salvare il modello corrente su disco? (s/n): ").strip().lower()
                if salva == "s":
                    save_checkpoint(modello, stoi, itos)
                print("Ciao!")
                break

            else:
                print("Opzione non valida, scegli un numero da 1 a 5.")
    except (EOFError, KeyboardInterrupt):
        # Ctrl+D / Ctrl+C durante un input(): usciamo senza salvare invece
        # di mostrare un traceback poco elegante.
        print("\nCiao!")


if __name__ == "__main__":
    main()
