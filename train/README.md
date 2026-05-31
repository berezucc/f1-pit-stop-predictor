# Training

`model.cbm` and `meta.json` at the repo root are committed, so the app runs
without this step. Use this only to reproduce or retrain.

The CSV is not included. Get the Kaggle
[Playground Series S6E5](https://www.kaggle.com/competitions/playground-series-s6e5)
data first:

```bash
kaggle competitions download -c playground-series-s6e5 -p train/data/
unzip train/data/playground-series-s6e5.zip -d train/data/
```

Then:

```bash
pip install -r train/requirements.txt
python train/train_app_model.py   # writes ../model.cbm and ../meta.json
```

What it does:

- Drops `Pre-Season Testing` and recomputes `RaceProgress` so training matches
  the single-row inference path.
- 5-fold stratified OOF AUC for a non-leaked headline number.
- Sets the decision threshold at the OOF point that maximises Youden's J.
- Refits on all rows, saves the model and the dashboard metadata.

Features: `Race`, `Compound`, `Year`, `LapNumber`, `Stint`, `TyreLife`,
`Position`, `RaceProgress`. Target: `PitNextLap`.
