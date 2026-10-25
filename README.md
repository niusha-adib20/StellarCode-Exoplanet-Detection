# 🌌 StellarCode — Exoplanet Detection

A Flask web application for exploring exoplanet candidate classification with tabular machine learning and Kepler, K2, and TESS light-curve data. It includes a saved tabular model, an optional 1D CNN for light curves, batch prediction, and an interactive web interface.

> **Research note:** Predictions are exploratory and educational. They do not confirm an exoplanet; confirmation requires scientific follow-up and validation.

## Features

- **Tabular classification:** Classifies candidate records as `CONFIRMED` or `FALSE_POSITIVE` from the features expected by the saved model.
- **Mission light curves:** Searches public Kepler, K2, or TESS light-curve products with Lightkurve, folds the signal using a supplied period and transit epoch, and runs the CNN when its model is installed.
- **Batch inference:** Accepts a list of JSON rows or a CSV upload.
- **Decision thresholds:** Supports `default`, `high_precision`, and `youden` threshold modes; thresholds come from the saved artifacts where available and otherwise use application defaults.
- **Web interface:** Provides a browser-based form and visualizations for prediction workflows.

The `/api/predict/ensemble` route is a demo fusion workflow: it combines the tabular score with a **synthetic** light curve generated from the submitted candidate's transit depth. It does not use the target's downloaded mission light curve. Use `/api/predict/astronomical` for inference on a retrieved mission light curve.

## How the models work

### Tabular model

The application loads `koi_toi_artifacts.pkl` at startup. This artifact contains the fitted imputer, feature list, and selected classifier. Feature values are aligned to that list; columns named `log_...` are computed from their base feature before imputation. The expected feature names can be read from `koi_toi_features.json` or returned by `GET /api/meta`.

The training notebook combines KOI and TOI catalog data, maps a shared feature set, and saves the inference artifact. It evaluates Random Forest and, when installed, XGBoost and CatBoost; the final saved inference model is a selected single classifier.

### Light-curve CNN

When TensorFlow and `cnn_lc_boosted.keras` are available, the application can classify either a downloaded mission light curve or user-supplied phase/flux arrays. Mission retrieval needs an internet connection and a public data product for the target. The downloaded signal is normalized, folded around the supplied epoch and period, clipped, binned, and passed to the CNN. CNN metadata and thresholds are loaded from `cnn_artifacts_boost.pkl` when present.

The CNN is optional for starting the web app. CNN-specific endpoints return an error when the model is unavailable; the demo ensemble route falls back to a simple synthetic-signal heuristic.

## Tech stack

Python, Flask, Flask-CORS, NumPy, pandas, scikit-learn, TensorFlow/Keras, Lightkurve, Astropy, XGBoost, CatBoost, Plotly, and browser-side HTML/CSS/JavaScript.

## Repository layout

```text
.
├── app.py
├── templates/
│   └── index.html
├── statics/
│   └── music/
│       └── gateway.mp3
├── koi_toi_artifacts.pkl
├── koi_toi_features.json
├── cnn_lc_boosted.keras
├── cnn_artifacts_boost.pkl
├── requirements.txt
└── train_model_code/
    ├── cnn/
    │   ├── cnn.ipynb
    │   └── tess_koi_lc_rescued_clip8.npz
    ├── machine_learning_code/
    │   └── model.ipynb
    └── data/
        ├── koi.csv
        ├── toi.csv
        └── k2.csv
```

## Getting started

### Requirements

- Python 3.11 is the version recorded in the training notebook metadata; use a compatible Python environment for the pinned packages in `requirements.txt`.
- Git
- Internet access for retrieving mission light curves

### Install and run

```bash
git clone https://github.com/niusha-adib20/StellarCode-Exoplanet-Detection.git
cd StellarCode-Exoplanet-Detection
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install the pinned dependencies and start the application:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open [http://localhost:8000](http://localhost:8000). The app listens on port `8000` by default and uses the `PORT` environment variable when set.

The tabular artifact `koi_toi_artifacts.pkl` is required for tabular prediction. The CNN model and metadata files enable the CNN workflows; if the model is absent or TensorFlow cannot load it, CNN inference is disabled. Artifact files must be compatible with the installed library versions.

## Using the application

- **Tabular:** Submit candidate measurements using the feature names listed by `GET /api/meta`.
- **Astronomical:** Select TESS, Kepler, or K2 and provide a target ID, orbital period in days, and transit epoch `t0`. The epoch may be JD, BTJD, or BKJD as supported by the application.
- **Batch:** Send JSON containing a `rows` array or upload a CSV file using the multipart field `file`.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Serves the web interface |
| GET | `/api/meta` | Returns the loaded feature list, model status, thresholds, and endpoint names |
| POST | `/api/predict/tabular` | Classifies one tabular candidate |
| POST | `/api/predict/batch` | Classifies JSON rows or an uploaded CSV |
| POST | `/api/predict/astronomical` | Retrieves a mission light curve and classifies it with the CNN |
| POST | `/api/predict/lc_cnn` | Classifies supplied phase/flux arrays with the CNN |
| POST | `/api/predict/ensemble` | Combines tabular probability with a synthetic-light-curve CNN/heuristic score |
| POST | `/api/lightcurve/synthetic` | Generates a synthetic light curve for the UI workflow |

### Tabular prediction example

Send either a nested `features` object or a flat JSON object. This example is illustrative; your model may require additional inputs. Check `/api/meta` for the complete feature list.

```bash
curl -X POST http://localhost:8000/api/predict/tabular \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "koi_period": 5.43,
      "koi_duration": 2.5,
      "koi_depth": 300.0,
      "koi_prad": 1.2,
      "koi_srad": 0.9,
      "koi_kepmag": 12.0
    },
    "threshold_mode": "default"
  }'
```

## Training

The notebooks under `train_model_code/` contain the training workflows:

- `machine_learning_code/model.ipynb` prepares and evaluates tabular classifiers and saves the tabular inference artifacts.
- `cnn/cnn.ipynb` trains and evaluates the light-curve CNN using the prepared `.npz` dataset.

Follow the data paths and setup cells in each notebook when retraining. The runtime application consumes the saved artifacts at the repository root.

## Limitations

- Classification quality depends on input quality, training data, preprocessing, artifact versions, and the selected threshold.
- Mission light-curve predictions require an available public data product and a working network connection.
- `/api/predict/ensemble` uses a generated synthetic signal, so its CNN score is not based on measured photometry for the submitted target.
- The application is a project and exploration tool, not an official NASA service or a substitute for scientific validation.
- The repository pins many dependencies. TensorFlow and other scientific packages can have platform-specific installation constraints.

## Data and references

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [NASA TESS Mission](https://science.nasa.gov/mission/tess/)
- [NASA Kepler Mission](https://science.nasa.gov/mission/kepler/)
- [Lightkurve documentation](https://lightkurve.github.io/)
- [TensorFlow documentation](https://www.tensorflow.org/)
- [Flask documentation](https://flask.palletsprojects.com/)

## License

This project is distributed under the [MIT License](LICENSE).
