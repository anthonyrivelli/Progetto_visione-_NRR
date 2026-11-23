#scarica il datatset con i file temporanei #
# import asyncio
# from ultralytics.data.converter import convert_ndjson_to_yolo

# asyncio.run(convert_ndjson_to_yolo("day-detection-16v2iyolo26.ndjson", output_path="datasets"))

# scarica solo il datatset e prepara il zip #
import asyncio
import shutil
from pathlib import Path
from ultralytics.data.converter import convert_ndjson_to_yolo

yaml_path = asyncio.run(convert_ndjson_to_yolo("day-detection-solo-classe.ndjson", output_path="datasets"))
print("data.yaml:", yaml_path)

dataset_dir = Path(yaml_path).parent
zip_path = shutil.make_archive(str(dataset_dir), "zip", root_dir=dataset_dir)
print("Zip creato:", zip_path)

#scarica il dataset pronto per l'utilizzo #
# import asyncio
# import shutil
# import tempfile
# from pathlib import Path
# from ultralytics.data.converter import convert_ndjson_to_yolo

# ndjson = "day-detection-solo-classe.ndjson"
# dest = Path("datasets") / Path(ndjson).stem  # cartella finale, senza hash nel nome

# if dest.exists():
#     raise SystemExit(f"{dest} esiste già: cancellala o cambia nome")

# dest.parent.mkdir(parents=True, exist_ok=True)
# with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
#     yaml_path = asyncio.run(convert_ndjson_to_yolo(ndjson, output_path=tmp))
#     shutil.move(str(Path(yaml_path).parent), dest)

# print("Dataset pronto in:", dest.resolve())


# import json

# src = "day-detection-16v2iyolo26.ndjson"
# dst = "day-detection-solo-classe.ndjson"
# keep = "Zebra Crossing"         # nome della classe da tenere
# keep_empty = False   # True = tieni anche le immagini senza quella classe (come sfondi)

# with open(src, encoding="utf-8") as f:
#     lines = [json.loads(line) for line in f if line.strip()]

# header, records = lines[0], lines[1:]
# names = {int(k): v for k, v in header["class_names"].items()}
# if keep not in names.values():
#     raise SystemExit(f"Classe '{keep}' non trovata. Classi disponibili: {list(names.values())}")
# cls_id = next(k for k, v in names.items() if v == keep)
# header["class_names"] = {"0": keep}

# out = [header]
# for r in records:
#     ann = r.get("annotations", {})
#     new = {k: [[0, *item[1:]] for item in items if int(item[0]) == cls_id] for k, items in ann.items()}
#     new = {k: v for k, v in new.items() if v}
#     if new or keep_empty:
#         r["annotations"] = new
#         out.append(r)

# with open(dst, "w", encoding="utf-8") as f:
#     for obj in out:
#         f.write(json.dumps(obj) + "\n")

# print(f"{len(out) - 1} immagini su {len(records)} contengono '{keep}' -> {dst}")