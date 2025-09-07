## ML Challenges

- Équipe - Amine ROSTANE, Joplin WAMBA GOULA
- Majeure - IA & Data Science (5e année) @ ESIEA
- Enseignant - Matthieu Le Berre
- TD - ML Challenges

---

### Modèles entraînés (artifacts)
- ESPCN - `artifacts/espcn_30_epoch.pt`
- FSRCNN - `artifacts/fsrcnn_30_epoch.pt`
- EDSRLite - `artifacts/edsrlite_5_epoch.pt`

---

### Exécuter l’entraînement - `main.py`
- Entraîne un modèle (ESPCN, FSRCNN, EDSRLITE), sauvegarde un checkpoint avec métadonnées

```bash
# espcn
python main.py --model-class ESPCN --epochs 30 --batch-size 10 --output-model artifacts/espcn_30_epoch.pt

# fsrcnn
python main.py --model-class FSRCNN --epochs 30 --batch-size 10 --output-model artifacts/fsrcnn_30_epoch.pt

# edsrlite
python main.py --model-class EDSRLITE --epochs 5 --batch-size 8 --output-model artifacts/edsrlite_5_epoch.pt

# reprendre un entraînement
python main.py --model-class ESPCN --resume-from artifacts/espcn_30_epoch.pt --epochs 10 --output-model artifacts/espcn_40_epoch.pt
```

---

### Interface graphique - `gui_app.py`
- Ouvre une GUI pour charger un modèle `.pt`, importer une image et l’upscaler
```bash
python gui_app.py
```

---

### Temps réel webcam - `realtime.py`
- Compare côte à côte la montée bicubique vs la montée par le modèle, à partir d’une entrée basse résolution

```bash
# échelle auto d’après le checkpoint
python realtime.py --checkpoint artifacts/espcn_30_epoch.pt

# échelle explicite (ex. 4x)
python realtime.py --checkpoint artifacts/espcn_30_epoch.pt --downscale 4 --width 1280 --height 720 --display-height 480
```