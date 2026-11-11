import cv2
from ultralytics import YOLO
import numpy as np

# Funzioni di supporto matematico
def centro(bbox):
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

def area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def copertura(box_a, box_b):
    w = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])
    h = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return (w * h) / max(area(box_a), 1)

def calcola_istogramma(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist


class ObjectDetector:
    def __init__(self, model_path, conf_threshold):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

        # memoria per la gestione delle occlusioni
        self.oggetti = {}   
        self.alias = {}     
        self.frame_idx = 0

        self.max_frame_nascosto = 90 
        self.min_hits = 5              
        self.soglia_parziale = 0.2     
        self.soglia_occluso = 0.3      
        self.max_distanza = 1.0        
        self.max_diff_colore = 0.5 # 0->identici, 1->diversi     
        self.soglia_doppione = 0.7     

    def detect(self, frame, target_classes=None):
        results = self.model.track(
            frame, 
            conf=self.conf_threshold, 
            persist=True, 
            verbose=False, 
            tracker="tracktrack.yaml", 
            classes=target_classes, 
            device=0
        )[0]
        
        detections = []
        if results.boxes is not None and results.boxes.id is not None:
            track_ids = results.boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(results.boxes, track_ids):
                bbox = list(map(int, box.xyxy[0]))
                detections.append({
                    "bbox": bbox,
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                    "track_id": track_id,
                    "hist": calcola_istogramma(frame, bbox)
                })
        return detections

    def gestisci_occlusioni(self, frame, detections):
        self.frame_idx += 1
        h_frame, w_frame = frame.shape[:2]

        # DOPPIE DETECTION
        for det in detections:
            det["doppia"] = False
            altezza = det["bbox"][3] - det["bbox"][1]
            for altro in detections:
                if altro is det or altro["class_id"] != det["class_id"] or det["class_id"] == 0:
                    continue
                stessa_base = abs(det["bbox"][3] - altro["bbox"][3]) < 0.1 * altezza
                if stessa_base and area(det["bbox"]) < area(altro["bbox"]) and copertura(det["bbox"], altro["bbox"]) > self.soglia_doppione:
                    det["doppia"] = True
        detections = [det for det in detections if not det["doppia"]]

        # OCCLUSIONE PARZIALE 
        for det in detections:
            det["coperto"] = 0.0
            altezza = det["bbox"][3] - det["bbox"][1]
            for altro in detections:
                if altro is det or altro["bbox"][3] < det["bbox"][3] + 0.1 * altezza:
                    continue
                det["coperto"] = max(det["coperto"], copertura(det["bbox"], altro["bbox"]))

        # TRACK ORIGINALE
        track_presenti = [det["track_id"] for det in detections]
        for det in detections:
            id_vecchio = self.alias.get(det["track_id"])
            if id_vecchio is not None and id_vecchio in track_presenti:
                del self.alias[det["track_id"]]
                self.nuovo_oggetto(det["track_id"], det)

        # OGGETTI GIÀ CONOSCIUTI
        visibili = []
        nuovi = []
        for det in detections:
            id_ogg = self.alias.get(det["track_id"], det["track_id"])
            if id_ogg in self.oggetti:
                self.aggiorna_oggetto(id_ogg, det)
                visibili.append(id_ogg)
            else:
                nuovi.append(det)

        # OGGETTI NUOVI
        for det in nuovi:
            id_vecchio = self.cerca_oggetto_nascosto(det, visibili)
            if id_vecchio is not None:
                self.alias[det["track_id"]] = id_vecchio
                id_ogg = id_vecchio
            else:
                id_ogg = det["track_id"]
                self.nuovo_oggetto(id_ogg, det)
                
            self.aggiorna_oggetto(id_ogg, det)
            visibili.append(id_ogg)

        # OGGETTI NASCOSTI E PREDITTIVI
        for id_ogg in list(self.oggetti.keys()):
            if id_ogg in visibili:
                continue
            
            ogg = self.oggetti[id_ogg]

            if self.frame_idx - ogg["ultimo_frame"] > self.max_frame_nascosto or ogg["hits"] < self.min_hits:
                del self.oggetti[id_ogg]
                continue

            # Predizione vettore velocità 
            vx, vy = ogg["velocita"]
            x1, y1, x2, y2 = ogg["bbox"]
            ogg["bbox"] = [x1 + vx, y1 + vy, x2 + vx, y2 + vy]

            cx, cy = centro(ogg["bbox"])
            fuori = cx < 0 or cx > w_frame or cy < 0 or cy > h_frame
            sul_bordo = (x1 <= 5 and vx < 0) or (x2 >= w_frame - 5 and vx > 0)
            if fuori or sul_bordo:
                del self.oggetti[id_ogg]
                continue

            doppione = False
            altezza = ogg["bbox"][3] - ogg["bbox"][1]
            for id_altro in visibili:
                altro = self.oggetti[id_altro]
                stessa_base = abs(ogg["bbox"][3] - altro["bbox"][3]) < 0.1 * altezza
                if altro["class_id"] == ogg["class_id"] and ogg["class_id"] != 0 and stessa_base and copertura(ogg["bbox"], altro["bbox"]) > self.soglia_doppione:
                    doppione = True
            
            if doppione:
                del self.oggetti[id_ogg]
                continue

            ogg["stato"] = "non rilevato"
            for id_altro in visibili:
                if copertura(ogg["bbox"], self.oggetti[id_altro]["bbox"]) > self.soglia_occluso:
                    ogg["stato"] = f"occluso da ID {id_altro}"
                    break

    def nuovo_oggetto(self, id_ogg, det):
        self.oggetti[id_ogg] = {
            "bbox": det["bbox"],
            "class_id": det["class_id"],
            "confidence": det["confidence"],
            "velocita": [0.0, 0.0],
            "colore": det["hist"],
            "ultimo_frame": self.frame_idx,
            "hits": 0,
            "stato": "visibile",
        }

    def aggiorna_oggetto(self, id_ogg, det):
        ogg = self.oggetti[id_ogg]
        if det["coperto"] < self.soglia_parziale:
            if self.frame_idx - ogg["ultimo_frame"] == 1 and ogg["stato"] == "visibile":
                cx_old, cy_old = centro(ogg["bbox"])
                cx_new, cy_new = centro(det["bbox"])
                vx, vy = ogg["velocita"]
                ogg["velocita"] = [0.7 * vx + 0.3 * (cx_new - cx_old), 0.7 * vy + 0.3 * (cy_new - cy_old)]

            if det["hist"] is not None:
                ogg["colore"] = det["hist"]
            ogg["stato"] = "visibile"
        else:
            ogg["stato"] = "occluso parz."

        ogg["bbox"] = det["bbox"]
        ogg["class_id"] = det["class_id"]
        ogg["confidence"] = det["confidence"]
        ogg["ultimo_frame"] = self.frame_idx
        ogg["hits"] += 1

    def cerca_oggetto_nascosto(self, det, visibili):
        hist = det["hist"]
        cx, cy = centro(det["bbox"])
        id_migliore = None
        punteggio_migliore = None

        for id_ogg, ogg in self.oggetti.items():
            if id_ogg in visibili or ogg["hits"] < self.min_hits or ogg["class_id"] != det["class_id"]:
                continue

            ox, oy = centro(ogg["bbox"])
            altezza = max(ogg["bbox"][3] - ogg["bbox"][1], 1)
            distanza = np.hypot(cx - ox, cy - oy) / altezza
            
            if distanza > self.max_distanza:
                continue

            if hist is None or ogg["colore"] is None:
                diff_colore = self.max_diff_colore
            else:
                diff_colore = cv2.compareHist(hist, ogg["colore"], cv2.HISTCMP_BHATTACHARYYA)
            
            if diff_colore > self.max_diff_colore:
                continue

            punteggio = distanza / self.max_distanza + diff_colore / self.max_diff_colore
            if punteggio_migliore is None or punteggio < punteggio_migliore:
                id_migliore = id_ogg
                punteggio_migliore = punteggio

        return id_migliore

    def conteggio(self):
        contatore = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            5: 0,
            7: 0,
        }
        for ogg in self.oggetti.values():
            # class_id = ogg["class_id"]
            # if class_id in contatore:
            #     contatore[class_id] += 1
            
            contatore[ogg["class_id"]] += 1
        return contatore

    def process_video_stream(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Impossibile aprire il flusso video")

        np.random.seed(100)
        colors = np.random.randint(0, 255, size=(100, 3), dtype="uint8").tolist()
        
        target_classes = [0, 1, 2, 3, 5, 7] 
        class_names = {
            0: "Persone",
            1: "Biciclette",
            2: "Auto",
            3: "Moto",
            5: "Autobus",
            7: "Camion"
        }

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = self.detect(frame, target_classes=target_classes)
            self.gestisci_occlusioni(frame, detections)

            for id_ogg, ogg in self.oggetti.items():
                x1, y1, x2, y2 = [int(v) for v in ogg["bbox"]]
                class_id = ogg["class_id"]
                color = tuple(colors[class_id % len(colors)])

                if ogg["stato"] == "visibile" or ogg["stato"] == "occluso parz.":
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{id_ogg} Cls:{class_id} Conf:{ogg['confidence']:.2f}"
                    if ogg["stato"] == "occluso parz.":
                        label += " (occluso parz.)"
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    label = f"ID:{id_ogg} ({ogg['stato']})"

                cv2.putText(frame, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # VISUALIZZAZIONE CONTATORE SULLO SCHERMO
            contatore = self.conteggio()
            
            # Overlay semi-trasparente per il pannello del contatore
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (220, 210), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # Titolo ed elementi del contatore
            cv2.putText(frame, "CONTATORE", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.line(frame, (20, 42), (210, 42), (255, 255, 255), 1)

            y_offset = 65
            for cls_id, count in contatore.items():
                nome_classe = class_names.get(cls_id, f"Classe {cls_id}")
                cv2.putText(frame, f"{nome_classe}: {count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += 24

            cv2.imshow("Detection & Tracking in tempo reale", frame)
            
            if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ObjectDetector(model_path="yolo26m.pt", conf_threshold=0.30)
    detector.process_video_stream("video/video_traffico.mp4")