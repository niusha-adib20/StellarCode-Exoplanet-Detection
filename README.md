# 🌌 StellarCode — Exoplanet Detection

An astronomy-focused machine learning project that helps identify potential exoplanets using data from NASA's Kepler, K2, and TESS missions. The project combines a tabular classifier with a 1D Convolutional Neural Network (CNN) for stellar light curves and presents predictions through an interactive Flask web application.

## ✨ Features

- **Tabular classification:** Predicts whether a candidate is likely to be a confirmed exoplanet or a false positive using measured planetary and stellar properties.
- **Light-curve analysis:** Retrieves public light-curve data for a target, preprocesses and phase-folds the signal, then evaluates it with a 1D CNN.
- **Combined prediction:** Offers an ensemble workflow that combines tabular and CNN-based predictions when the CNN model is available.
- **Batch inference:** Accepts multiple records as JSON rows or a CSV upload.
- **Decision-threshold options:** Supports default, high-precision, and Youden-style threshold modes where configured.
- **Interactive web interface:** A space-themed UI for entering candidate data, running predictions, and viewing results.

> **Important:** This project provides machine-learning predictions for exploration and educational use. Its outputs should not be treated as scientific confirmation of an exoplanet.

## 🧠 How It Works

### 1. Tabular model
The tabular pipeline uses candidate features such as orbital period, transit duration and depth, planet radius, stellar radius/temperature, and magnitude-related measurements. Missing values are handled with a fitted imputer, and selected skewed features are represented with log-transformed values. The saved model and preprocessing artifacts are loaded by the Flask application.

### 2. Light-curve CNN
For astronomical inference, the app can search for a target's light curve through [Lightkurve](https://lightkurve.github.io/), which accesses publicly available mission data. The signal is normalized, aligned to the supplied transit epoch, phase-folded around the transit, clipped for extreme values, binned to a fixed-length sequence, and passed to a 1D CNN.

Supported mission identifiers include **TESS**, **Kepler**, and **K2**. A target identifier, orbital period, and transit epoch (`t0`) are required for this workflow.

### 3. Combined workflow
The application includes an ensemble endpoint that combines the tabular prediction with a CNN-based signal when available. The final label depends on the probability and the selected decision threshold.

## 🛠️ Tech Stack

- **Language:** Python
- **Web application:** Flask, Flask-CORS, HTML, CSS, JavaScript
- **Data processing:** NumPy, pandas, scikit-learn
- **Deep learning:** TensorFlow / Keras
- **Astronomy data:** Lightkurve, Astropy
- **Additional model libraries:** XGBoost, CatBoost
- **Visualization / interface:** Plotly and custom frontend components

## 📁 Project Structure

```text
StellarCode-Exoplanet-Detection/
├── app.py                              # Flask app and prediction API
├── templates/
│   └── index.html                      # Web interface
├── statics/
│   └── music/
│       └── gateway.mp3                 # Optional UI background audio
├── koi_toi_artifacts.pkl               # Saved tabular model and preprocessing
├── koi_toi_features.json               # Tabular model feature list
├── cnn_lc_boosted.keras                # Saved light-curve CNN
├── cnn_artifacts_boost.pkl             # CNN metadata and thresholds
├── requirements.txt                    # Python dependencies
└── train_model_code/
    ├── cnn/
    │   ├── cnn.ipynb                   # CNN training workflow
    │   └── tess_koi_lc_rescued_clip8.npz
    ├── machine_learning_code/
    │   └── model.ipynb                 # Tabular model training workflow
    └── data/
        ├── koi.csv
        ├── toi.csv
        └── k2.csv
```

## 🚀 Getting Started

### Prerequisites

- Python **3.11** is recommended (the training notebook metadata uses Python 3.11).
- Git
- Internet access for fetching mission light curves
- Enough memory and disk space for the scientific Python dependencies and saved models

### 1. Clone the repository

```bash
git clone https://github.com/niusha-adib20/StellarCode-Exoplanet-Detection.git
cd StellarCode-Exoplanet-Detection
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

The Flask app uses port **8000** by default. Open [http://localhost:8000](http://localhost:8000) in your browser. If the `PORT` environment variable is set, the application uses that port instead.

The repository includes saved model artifacts, so the app expects these files to be present at the project root. If an artifact is missing or incompatible with your installed library versions, model loading or predictions may fail.

## 🔭 Using the Application

### Tabular prediction
Enter the available candidate and stellar measurements in the web form, then run a prediction. The app returns a probability, predicted label, and the decision threshold used.

### Astronomical light-curve prediction
Choose a mission (**TESS**, **Kepler**, or **K2**) and provide:
- Target identifier
- Orbital period in days
- Transit epoch `t0` (JD, BTJD, or BKJD, as appropriate)

The app attempts to retrieve available public light-curve data, preprocesses the signal, and returns the CNN probability and phase-folded light curve.

### Batch prediction
Upload a CSV file through the batch form to classify multiple candidates in one request.

## 🔌 API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Serves the web interface |
| GET | `/api/meta` | Returns model metadata, available thresholds, and endpoint information |
| POST | `/api/predict/tabular` | Predicts from one candidate's tabular features |
| POST | `/api/predict/batch` | Predicts from JSON rows or an uploaded CSV |
| POST | `/api/predict/astronomical` | Retrieves and classifies a mission light curve |
| POST | `/api/predict/lc_cnn` | Classifies a supplied phase/flux light curve with the CNN |
| POST | `/api/predict/ensemble` | Combines tabular and CNN-based predictions when available |
| POST | `/api/lightcurve/synthetic` | Generates a synthetic light curve for the app workflow |

### Example: tabular prediction request

The API accepts either a nested `features` object or a flat JSON object with feature names expected by the saved model.

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

This is an illustrative request only: the saved model may require additional features. Check `/api/meta` for the complete `feature_list`.

## 🧪 Model Training

The training notebooks are in `train_model_code/`:

- `machine_learning_code/model.ipynb` — trains and evaluates tabular classifiers, including Random Forest, XGBoost, CatBoost when available, and a probability-averaging ensemble.
- `cnn/cnn.ipynb` — trains a 1D CNN for phase-folded light curves and evaluates performance with metrics such as precision, recall, F1, and ROC-AUC.

The training workflow saves model and preprocessing artifacts consumed by `app.py`. To retrain the models, open the corresponding notebook and follow its data paths and setup cells.

## ⚠️ Notes and Limitations

- Results depend on the supplied measurements, available mission data, preprocessing choices, saved model versions, and decision threshold.
- Light-curve retrieval requires internet access and an available public data product for the selected target.
- Some tabular features may be mission-specific; the training notebook maps a shared feature set between the KOI and TOI catalogues.
- The project is an exploration tool, not a substitute for peer-reviewed analysis or confirmation by an astronomy team.
- `requirements.txt` pins many package versions. If installation fails on your platform, use a clean Python 3.11 environment and consult the package-specific installation instructions, especially for TensorFlow.

## 📚 Data and References

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [NASA TESS Mission](https://science.nasa.gov/mission/tess/)
- [NASA Kepler Mission](https://science.nasa.gov/mission/kepler/)
- [Lightkurve documentation](https://lightkurve.github.io/)
- [TensorFlow documentation](https://www.tensorflow.org/)
- [Flask documentation](https://flask.palletsprojects.com/)

## 📄 License

See the [LICENSE](LICENSE) file for the license that applies to this repository.

---

Built with Python, machine learning, and a little cosmic curiosity. ✨
