import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Permette di salvare i grafici su file anche senza finestra grafica
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import cv2
import numpy as np
from ultralytics import YOLOE
from ultralytics.data.utils import img2label_paths
from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer

# ==============================================================================
# CONFIGURAZIONE PERCORSI E PARAMETRI
# ==============================================================================
DATASET_PATH = r"./datasets/crosswalk-segv1iyolo26"      # Cartella del dataset oppure percorso diretto a data.yaml
OUTPUT_DIR   = r"./modelli/test"    # Cartella in cui salvare pesi, grafici e valutazioni
MODEL_PATH   = r"yoloe-26n-seg.pt"               # Percorso del file .pt locale

EPOCHS         = 10
IMGSZ          = 640
BATCH          = 8
PATIENCE       = 15
DEVICE         = 0  # None = automatico (GPU se disponibile), "0" = prima GPU, "cpu" = solo CPU
CONF_THRESHOLD = 0.25   # Soglia di confidenza per le immagini di anteprima
# ==============================================================================


def risolvi_data_yaml(dataset_path: str) -> Path:
    """Accetta sia la cartella del dataset sia il percorso diretto a data.yaml."""
    p = Path(dataset_path).expanduser().resolve()
    yaml_file = (p / "data.yaml") if p.is_dir() else p

    if not yaml_file.exists():
        raise FileNotFoundError(f"File data.yaml non trovato in: {yaml_file}")
    return yaml_file


def salva_grafici_training(run_dir: Path) -> None:
    """Salva i grafici come a lezione: un grafico per ogni metrica, training contro validazione."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        print(f"[AVVISO] {csv_path} non trovato, salto la creazione dei grafici.")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Loss totale = somma di tutte le loss di YOLO (box, segmentazione, classe, ...)
    colonne_training = [c for c in df.columns if c.startswith("train/") and c.endswith("_loss")]
    colonne_validazione = [c for c in df.columns if c.startswith("val/") and c.endswith("_loss")]

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df[colonne_training].sum(axis=1), label="Loss training")
    plt.plot(df["epoch"], df[colonne_validazione].sum(axis=1), label="Loss validazione")
    plt.xlabel("Epoca")
    plt.ylabel("Loss totale")
    plt.title("Andamento della loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(run_dir / "andamento_loss.png")
    plt.close()

    # mAP delle maschere (YOLO la calcola solo sulla validazione)
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["metrics/mAP50(M)"], label="mAP50 validazione")
    plt.plot(df["epoch"], df["metrics/mAP50-95(M)"], label="mAP50-95 validazione")
    plt.xlabel("Epoca")
    plt.ylabel("mAP maschere")
    plt.title("Andamento della mAP")
    plt.legend()
    plt.grid(True)
    plt.savefig(run_dir / "andamento_map.png")
    plt.close()

    print(f"[INFO] Grafici salvati in: {run_dir}")

def maschera_da_poligoni(poligoni, altezza, larghezza):
    """Disegna i poligoni pieni su un'immagine nera: 1 dove ci sono le strisce, 0 altrove."""
    maschera = np.zeros((altezza, larghezza), dtype=np.uint8)
    for poligono in poligoni:
        if len(poligono) >= 3:
            cv2.fillPoly(maschera, [np.array(poligono, dtype=np.int32)], 1)
    return maschera


def colora_maschera(immagine, maschera):
    """Colora di verde la zona della maschera lasciando vedere l'immagine sotto."""
    colorata = immagine.copy()
    colorata[maschera == 1] = (0.5 * colorata[maschera == 1] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    return colorata


def mostra_migliori_risultati(model, immagini, run_dir: Path, n: int = 6) -> None:
    """Come a lezione: tre righe con immagine, maschera predetta e maschera reale.
    I migliori risultati sono quelli con l'IoU piu' alta tra maschera predetta e maschera reale."""
    risultati = []

    for percorso in immagini:
        immagine = cv2.cvtColor(cv2.imread(str(percorso)), cv2.COLOR_BGR2RGB)
        altezza, larghezza = immagine.shape[:2]

        # Maschera reale: poligoni del file .txt delle etichette (coordinate tra 0 e 1)
        poligoni_reali = []
        file_etichette = Path(img2label_paths([str(percorso)])[0])
        if file_etichette.exists():
            for riga in file_etichette.read_text().splitlines():
                valori = [float(v) for v in riga.split()[1:]]
                punti = [[valori[i] * larghezza, valori[i + 1] * altezza] for i in range(0, len(valori) - 1, 2)]
                poligoni_reali.append(punti)

        # Maschera predetta dal modello
        r = model.predict(str(percorso), conf=CONF_THRESHOLD, imgsz=IMGSZ, verbose=False)[0]
        poligoni_predetti = r.masks.xy if r.masks is not None else []

        maschera_reale = maschera_da_poligoni(poligoni_reali, altezza, larghezza)
        maschera_predetta = maschera_da_poligoni(poligoni_predetti, altezza, larghezza)
        intersezione = np.logical_and(maschera_reale, maschera_predetta).sum()
        unione = np.logical_or(maschera_reale, maschera_predetta).sum()
        iou = intersezione / unione if unione > 0 else 1.0

        risultati.append({"iou": iou, "immagine": percorso, "reali": poligoni_reali, "predetti": poligoni_predetti})

    print(f"[INFO] IoU media sulle {len(risultati)} immagini: {np.mean([r['iou'] for r in risultati]):.3f}")

    risultati = sorted(risultati, key=lambda x: x["iou"], reverse=True)
    migliori = risultati[:n]

    plt.figure(figsize=(2.6 * n, 7))
    for i, risultato in enumerate(migliori):
        immagine = cv2.cvtColor(cv2.imread(str(risultato["immagine"])), cv2.COLOR_BGR2RGB)
        altezza, larghezza = immagine.shape[:2]

        plt.subplot(3, n, i + 1)
        plt.imshow(immagine)
        plt.title("Immagine")
        plt.axis("off")

        plt.subplot(3, n, i + 1 + n)
        plt.imshow(colora_maschera(immagine, maschera_da_poligoni(risultato["predetti"], altezza, larghezza)))
        plt.title(f"Predetta\nIoU={risultato['iou']:.3f}")
        plt.axis("off")

        plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(colora_maschera(immagine, maschera_da_poligoni(risultato["reali"], altezza, larghezza)))
        plt.title("Reale")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(run_dir / "migliori_risultati.png")
    plt.close()
    print(f"[INFO] Migliori risultati salvati in: {run_dir / 'migliori_risultati.png'}")


def main():
    # percorsi dataset e cartella di output
    data_yaml = risolvi_data_yaml(DATASET_PATH)
    output_dir = Path(OUTPUT_DIR).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    project_dir = output_dir.parent
    run_name = output_dir.name

    print(f"Dataset : {data_yaml}")
    print(f"Modello : {MODEL_PATH}")
    print(f"Output  : {output_dir}")

    # 2fine-tuning
    model = YOLOE(MODEL_PATH)
    train_kwargs = dict(
        data=str(data_yaml),
        trainer=YOLOEPESegTrainer,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        plots=False,
    )
    if DEVICE is not None:
        train_kwargs["device"] = DEVICE

    model.train(**train_kwargs)

    # 3. Genera e salva i grafici di training con Matplotlib
    salva_grafici_training(output_dir)

    # 4. Carica il modello migliore e valida
    best_weights = output_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Checkpoint migliore non trovato in: {best_weights}")

    # Copia una versione comodamente accessibile direttamente dentro OUTPUT_DIR
    shutil.copy2(best_weights, output_dir / "best.pt")

    with open(data_yaml, "r", encoding="utf-8") as f:
        yaml_cfg = yaml.safe_load(f)
    eval_split = "test" if yaml_cfg.get("test") else "val"

    print(f"\n[INFO] Avvio valutazione finale sullo split '{eval_split}'...")
    best_model = YOLOE(str(best_weights))
    metrics = best_model.val(
        data=str(data_yaml),
        split=eval_split,
        imgsz=IMGSZ,
        batch=BATCH,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        plots=False,
    )

    print("\n=== RISULTATI FINALI ===")
    print(f"Box  mAP50    : {metrics.box.map50:.4f}")
    print(f"Box  mAP50-95 : {metrics.box.map:.4f}")
    print(f"Mask mAP50    : {metrics.seg.map50:.4f}")
    print(f"Mask mAP50-95 : {metrics.seg.map:.4f}")

if __name__ == "__main__":
    main()