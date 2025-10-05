import os
import io
import json
import pickle
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# -------------------------
# CONFIG
# -------------------------
TABULAR_ARTIFACTS = Path("koi_toi_artifacts.pkl")
CNN_MODEL_PATH    = Path("cnn_lc_boosted.keras")
CNN_ARTIFACTS     = Path("cnn_artifacts_boost.pkl")

DEFAULT_THRESHOLD_MODES = {
    "default": 0.5,
    "high_precision": 0.95,
    "youden": 0.5,
}

HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8000"))
DEBUG = True

# -------------------------
# Flask
# -------------------------
app = Flask(__name__, static_folder='statics')
CORS(app)

# -------------------------
# Globals (models/objects)
# -------------------------
g = {
    "imputer": None,
    "feature_list": None,
    "selected_model": None,
    "models": {},        # {"SELECTED": clf, "RF":..., "XGB":..., "CAT":...} if present
    "thresholds": {},    # {"selected_model": {...}, "ENS": {...}, ...}
    "has_cnn": False,
    "cnn_model": None,
    "cnn_input_len": 201,
    "cnn_thresholds": {"default": 0.5, "high_precision": 0.95, "youden": 0.5},
    "cnn_val_ids": [],
}

# -------------------------
# Small utils
# -------------------------
def _json_error(msg: str, code: int = 400, detail: str = ""):
    return jsonify({"ok": False, "error": msg, "detail": detail}), code

def _extract_features_payload(payload: Dict[str, Any], feature_list: List[str]) -> Dict[str, Any]:
    """Accept both {'features':{...}} and flat JSON with feature keys."""
    if isinstance(payload, dict) and "features" in payload and isinstance(payload["features"], dict):
        return payload["features"]
    return {k: payload.get(k, None) for k in feature_list}

def _build_row_with_logs(feature_list: List[str], feat_map: Dict[str, Any]) -> List[float]:
    """
    Build a row aligned to feature_list.
    If a column starts with 'log_', compute log1p(abs(base)).
    """
    row = []
    cache = {}  # cache base values parsed to float
    for col in feature_list:
        if col.startswith("log_"):
            base = col.replace("log_", "")
            # ensure base parsed once
            if base not in cache:
                v = feat_map.get(base, None)
                cache[base] = None if v is None else float(v)
            bv = cache[base]
            row.append(np.log1p(abs(bv)) if bv is not None else None)
        else:
            v = feat_map.get(col, None)
            try:
                row.append(None if v is None else float(v))
            except Exception:
                row.append(None)
    return row

def _ensure_matrix(feature_list: List[str], payload: Dict[str, Any]) -> np.ndarray:
    feat_map = _extract_features_payload(payload, feature_list)
    row = _build_row_with_logs(feature_list, feat_map)
    return np.array([row], dtype=float)

def _apply_imputer_and_log(X: np.ndarray) -> np.ndarray:
    # imputer is already fitted on final feature_list (including any log_* columns)
    return g["imputer"].transform(X)

def _get_thr_for(model_key: str, mode: str) -> float:
    if model_key in g["thresholds"] and isinstance(g["thresholds"][model_key], dict) and mode in g["thresholds"][model_key]:
        return float(g["thresholds"][model_key][mode])
    if mode in g["thresholds"]:
        try:
            return float(g["thresholds"][mode])
        except Exception:
            pass
    return float(DEFAULT_THRESHOLD_MODES.get(mode, 0.5))

def _resample_1d(arr: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(arr).reshape(-1)
    if arr.size == target_len: return arr
    x_old = np.linspace(0, 1, num=arr.size)
    x_new = np.linspace(0, 1, num=target_len)
    return np.interp(x_new, x_old, arr)

# -------------------------
# TABULAR: load artifacts
# -------------------------
def load_tabular_artifacts():
    with open(TABULAR_ARTIFACTS, "rb") as f:
        art = pickle.load(f)

    g["imputer"] = art["imputer"]
    g["feature_list"] = art["feature_list"]
    g["selected_model"] = art["model"]
    g["models"]["SELECTED"] = g["selected_model"]

    thrs = {
        "default": 0.5,
        "high_precision": DEFAULT_THRESHOLD_MODES["high_precision"],
        "youden": DEFAULT_THRESHOLD_MODES["youden"],
    }
    # Merge any thresholds stored in your artifact
    for k in ["selected_model", "ENS", "RF", "XGB", "CAT"]:
        if k in art and isinstance(art[k], dict):
            thrs[k] = {**DEFAULT_THRESHOLD_MODES, **art[k]}
    if "decision_threshold" in art:
        thrs["selected_model"] = {
            **DEFAULT_THRESHOLD_MODES,
            "high_precision": float(art["decision_threshold"])
        }
    g["thresholds"] = thrs

# -------------------------
# CNN: load model + meta
# -------------------------
def load_cnn_artifacts():
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras
    except Exception as e:
        print("TensorFlow not available, CNN disabled:", e)
        return

    if not CNN_MODEL_PATH.exists():
        print("CNN model file not found:", CNN_MODEL_PATH)
        return

    from tensorflow import keras
    g["cnn_model"] = keras.models.load_model(CNN_MODEL_PATH)
    g["has_cnn"] = True

    if CNN_ARTIFACTS.exists():
        with open(CNN_ARTIFACTS, "rb") as f:
            meta = pickle.load(f)
        g["cnn_input_len"] = int(meta.get("input_len", 201))
        g["cnn_thresholds"] = {
            "default": 0.5,
            "high_precision": float(meta.get("thr_high_precision", DEFAULT_THRESHOLD_MODES["high_precision"])),
            "youden": float(meta.get("thr_youden", DEFAULT_THRESHOLD_MODES["youden"])),
        }
        g["cnn_val_ids"] = list(meta.get("val_ids", []))
    else:
        g["cnn_input_len"] = 201
        g["cnn_thresholds"] = DEFAULT_THRESHOLD_MODES.copy()
        g["cnn_val_ids"] = []

# =========================================================
# >>>>>>>>>>>  ASTRONOMY PIPELINE (Lightkurve)  <<<<<<<<<<<
# =========================================================
PHASE_HALF_WIDTH = 0.20
N_SAMPLES        = 201
MIN_POINTS       = 20
CLIP_SIGMA       = 8.0
MAX_FILES        = 3

JD0_BTJD = 2457000.0
JD0_BKJD = 2454833.0

def _align_epoch_to_lc_time(lc, t0_raw: float) -> float:
    """Align T0 (likely in JD) to LC time base (BTJD/BKJD)."""
    t0_raw = float(t0_raw)
    fmt = getattr(lc.time, "format", None)
    tmed = float(np.nanmedian(lc.time.value))
    is_jd_like = (t0_raw > 2.45e6)

    if fmt is not None:
        f = fmt.lower()
        if "btjd" in f:
            return (t0_raw - JD0_BTJD) if is_jd_like else t0_raw
        if "bkjd" in f:
            return (t0_raw - JD0_BKJD) if is_jd_like else t0_raw

    if is_jd_like:
        cand_btjd = t0_raw - JD0_BTJD
        cand_bkjd = t0_raw - JD0_BKJD
        return cand_btjd if abs(cand_btjd - tmed) < abs(cand_bkjd - tmed) else cand_bkjd
    return t0_raw

def _mission_ident(mission_id: str, target_id: str):
    m = (mission_id or "").strip().upper()
    target_id_str = str(target_id).strip().upper()
    target_id_clean = (target_id_str
                       .replace("TIC", "").replace("KOI", "").replace("KIC", "")
                       .replace("EPIC", "").replace("TOI", ""))
    try:
        tid = str(int(float(target_id_clean)))
    except ValueError:
        raise ValueError(f"Target ID must contain a valid number: {target_id}")

    if m in {"TESS", "TOI", "TIC"}:
        return "TESS", f"TIC {tid}"
    if m in {"KEPLER", "KOI", "KIC"}:
        return "Kepler", f"KIC {tid}"
    if m in {"K2", "EPIC"}:
        return "K2", f"EPIC {tid}"
    return "TESS", f"TIC {tid}"

def fetch_lightcurve_mast(mission: str, ident: str, max_files: int = MAX_FILES):
    """Download & stitch LC via lightkurve, with robust error handling."""
    try:
        import lightkurve as lk
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 1. Search for all light curves for the identifier
            sr = lk.search_lightcurve(ident, product="lightcurve")
        
        if len(sr) == 0:
            return None # No light curve found in the archive
        
        n = min(len(sr), max_files)
        
        # 2. Try to download the collection
        try:
            lc_collection = sr[:n].download_all()
            if lc_collection is None or len(lc_collection) == 0:
                 return None
            
            # 3. Try to stitch all of them
            lc = lc_collection.stitch()
            return lc
            
        except Exception:
            # 4. Fallback: If stitching fails (e.g., due to gaps or different data formats), 
            # try to return the first individual LightCurve object found.
            print(f"Warning: Failed to stitch light curves for {ident}. Attempting to return the first sector.")
            try:
                return lc_collection[0]
            except Exception:
                return None
                
    except Exception:
        # Handle exceptions during search (e.g., if Lightkurve package is missing)
        return None

def preprocess_and_fold(lc,
                        period_days: float,
                        t0_raw: float,
                        phase_half_width: float = PHASE_HALF_WIDTH,
                        n_samples: int = N_SAMPLES,
                        min_points: int = MIN_POINTS,
                        clip_sigma: float = CLIP_SIGMA) -> np.ndarray:
    """Normalize → (optional detrend) → align T0 → fold → window → robust clip → bin → normalize(median=1)."""
    import astropy.units as u

    lc = lc.normalize()
    # Optional: lc = lc.flatten(window_length=401)

    t0_aligned = _align_epoch_to_lc_time(lc, t0_raw)
    folded = lc.fold(period=float(period_days) * u.day, epoch_time=float(t0_aligned) * u.day)

    m = (folded.phase.value >= -phase_half_width) & (folded.phase.value <= +phase_half_width)
    if m.sum() < min_points:
        return None

    phase = folded.phase.value[m]
    flux  = folded.flux.value[m]

    med = np.nanmedian(flux)
    mad = np.nanmedian(np.abs(flux - med)) * 1.4826
    if mad > 0 and np.isfinite(mad):
        good = (flux > med - clip_sigma * mad) & (flux < med + clip_sigma * mad)
        phase, flux = phase[good], flux[good]
        if len(flux) < min_points:
            return None

    idx = np.argsort(phase)
    phase, flux = phase[idx], flux[idx]

    bins = np.linspace(-phase_half_width, +phase_half_width, n_samples + 1)
    digit = np.digitize(phase, bins) - 1
    out = np.zeros(n_samples, dtype=np.float32)
    cnt = np.zeros(n_samples, dtype=np.int32)
    for i, f in zip(digit, flux):
        if 0 <= i < n_samples:
            out[i] += f; cnt[i] += 1
    cnt[cnt == 0] = 1
    out = out / cnt
    out = np.nan_to_num(out, nan=float(np.nanmedian(out)))

    med2 = np.nanmedian(out)
    if not np.isfinite(med2) or med2 == 0:
        return None
    out = out / med2
    return out  # (n_samples,)

# =========================================================
# STARTUP: load once
# =========================================================
try:
    load_tabular_artifacts()
    load_cnn_artifacts()
    print("✅ Model artifacts loaded successfully.")
    print("Thresholds:", {**g["thresholds"], "CNN": g["cnn_thresholds"]})
except Exception as e:
    print("❌ Failed to load artifacts:", e)
    traceback.print_exc()

# =========================================================
# ROUTES
# =========================================================

@app.route("/", methods=["GET"])
def root():
    return render_template("index.html")

@app.route("/api/meta", methods=["GET"])
def api_meta():
    try:
        return jsonify({
            "ok": True,
            "model_name": "Ensemble Exoplanet Classifier (RF/XGB/CAT selection)",
            "default_threshold": float(DEFAULT_THRESHOLD_MODES.get("default", 0.5)),
            "high_precision_threshold": float(DEFAULT_THRESHOLD_MODES.get("high_precision", 0.95)),
            "feature_list": g["feature_list"],
            "has_cnn": g["has_cnn"],
            "cnn_input_len": g["cnn_input_len"],
            "thresholds": {**g["thresholds"], "CNN": g["cnn_thresholds"]},
            "models": list(g["models"].keys()),
            "endpoints": {
                "tabular": "/api/predict/tabular",
                "batch": "/api/predict/batch",
                "astronomical": "/api/predict/astronomical",
                "lc_cnn": "/api/predict/lc_cnn",
                "ensemble": "/api/predict/ensemble",
                "synthetic_lc": "/api/lightcurve/synthetic",
            }
        })
    except Exception as e:
        return _json_error("meta failed", 500, str(e))

# ---------------- TABULAR (single) ----------------
@app.route("/api/predict/tabular", methods=["POST"])
def predict_tabular():
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return _json_error("Expected JSON body with either {'features':{...}} or flat feature keys.", 400)

        model_key = payload.get("model", "SELECTED")
        mode = payload.get("threshold_mode", "default")
        if model_key not in g["models"]:
            model_key = "SELECTED"

        X = _ensure_matrix(g["feature_list"], payload)
        X_imp = _apply_imputer_and_log(X)

        clf = g["models"][model_key]
        if not hasattr(clf, "predict_proba"):
            return _json_error(f"Model '{model_key}' does not support predict_proba.", 500)

        prob = float(clf.predict_proba(X_imp)[0, 1])
        thr = _get_thr_for("selected_model" if model_key == "SELECTED" else model_key, mode)
        pred = int(prob >= thr)
        label = "CONFIRMED" if pred else "FALSE_POSITIVE"

        # Return with the keys your UI expects, plus some extras for debugging
        return jsonify({
            "ok": True,
            "Ensemble_Probability": prob,
            "Predicted_Label": label,
            "Decision_Threshold_Used": thr,
            "model": model_key,
            "threshold_mode": mode,
            "threshold_used": thr,
            "prob": prob,
            "pred": pred
        })
    except Exception as e:
        return _json_error("tabular prediction failed", 500, traceback.format_exc())

# ---------------- TABULAR (batch) ----------------
@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    try:
        model_key = request.args.get("model", "SELECTED")
        mode = request.args.get("threshold_mode", "default")
        if model_key not in g["models"]:
            model_key = "SELECTED"
        clf = g["models"][model_key]
        if not hasattr(clf, "predict_proba"):
            return _json_error(f"Model '{model_key}' does not support predict_proba.", 500)

        rows_dicts = []
        # Accept CSV upload or JSON rows
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            rows = payload.get("rows", [])
            if not rows:
                return _json_error("JSON must contain 'rows' list.", 400)
            rows_dicts = rows
        else:
            if "file" not in request.files:
                return _json_error("Provide JSON with 'rows' or upload CSV file as 'file'.", 400)
            file = request.files["file"]
            df = pd.read_csv(io.BytesIO(file.read()))
            rows_dicts = df.to_dict(orient="records")

        results = []
        idx = 0
        for rd in rows_dicts:
            # rd may already be {"features":{...}} or flat
            feat_map = _extract_features_payload(rd, g["feature_list"])
            X_row = np.array([_build_row_with_logs(g["feature_list"], feat_map)], dtype=float)
            X_imp = _apply_imputer_and_log(X_row)
            prob = float(clf.predict_proba(X_imp)[0, 1])
            thr = _get_thr_for("selected_model" if model_key == "SELECTED" else model_key, mode)
            label = "CONFIRMED" if prob >= thr else "FALSE_POSITIVE"
            the_id = rd.get("kic_id") or rd.get("tic_id") or "N/A"
            results.append({
                "Index": idx,
                "ID": the_id,
                "proba_tabular": prob,
                "decision": label
            })
            idx += 1

        return jsonify({"ok": True, "batch_results": results})
    except Exception as e:
        return _json_error("batch prediction failed", 500, traceback.format_exc())

# ---------------- ASTRONOMICAL (Lightkurve → CNN) ----------------
@app.route("/api/predict/astronomical", methods=["POST"])
def predict_astronomical():
    """
    JSON:
    { "mission_id": "TESS|Kepler|K2", "target_id": 123..., "period": 5.43, "t0": 2459001.23, "threshold_mode": "default|high_precision|youden" }
    """
    try:
        if not g["has_cnn"]:
            return _json_error("CNN model not loaded on server.", 500)

        payload = request.get_json(silent=True) or {}
        mission_id = payload.get("mission_id", "")
        target_id  = payload.get("target_id", "")
        period     = float(payload.get("period", "nan"))
        t0         = float(payload.get("t0", "nan"))
        mode       = payload.get("threshold_mode", "default")

        if not mission_id or target_id == "":
            return _json_error("Provide 'mission_id' and 'target_id'.", 400)
        if not np.isfinite(period) or not np.isfinite(t0):
            return _json_error("Provide valid numeric 'period' (days) and 't0' (JD/BTJD/BKJD).", 400)

        mission, ident = _mission_ident(mission_id, str(target_id))
        lc = fetch_lightcurve_mast(mission, ident, max_files=MAX_FILES)
        if lc is None:
            return _json_error(f"No light curve found for {mission}:{ident}.", 404)

        arr = preprocess_and_fold(
            lc, period_days=period, t0_raw=t0,
            phase_half_width=PHASE_HALF_WIDTH,
            n_samples=g["cnn_input_len"], min_points=MIN_POINTS, clip_sigma=CLIP_SIGMA
        )
        if arr is None:
            return _json_error("LC pre-processing produced no valid data.", 422)

        X_lc = arr.reshape(1, -1, 1)
        p = float(g["cnn_model"].predict(X_lc, verbose=0).ravel()[0])
        thr = float(g["cnn_thresholds"].get(mode, 0.5))
        pred = int(p >= thr)

        phase = np.linspace(-PHASE_HALF_WIDTH, +PHASE_HALF_WIDTH, g["cnn_input_len"]).tolist()
        flux  = arr.tolist()

        return jsonify({
            "ok": True,
            "mission": mission,
            "identifier": ident,
            "threshold_mode": mode,
            "threshold_used": thr,
            "prob": p,
            "pred": pred,
            "lightcurve": {"phase": phase, "flux": flux}
        })
    except Exception as e:
        return _json_error("astronomical prediction failed", 500, traceback.format_exc())

# ---------------- Direct CNN on provided arrays ----------------
@app.route("/api/predict/lc_cnn", methods=["POST"])
def predict_lc_cnn():
    try:
        if not g["has_cnn"]:
            return _json_error("CNN model not loaded on server.", 500)
        js = request.get_json(silent=True) or {}
        phase = js.get("phase"); flux = js.get("flux")
        mode  = js.get("threshold_mode", "default")
        if phase is None or flux is None:
            return _json_error("Provide JSON with 'phase' and 'flux' arrays.", 400)

        phase = np.asarray(phase, dtype=float).reshape(-1)
        flux  = np.asarray(flux,  dtype=float).reshape(-1)
        if phase.size != flux.size or phase.size < 10:
            return _json_error("Phase/flux length mismatch or too short.", 400)

        target_len = int(g.get("cnn_input_len", 201))
        flux_i = _resample_1d(flux, target_len)
        med = np.nanmedian(flux_i) if np.isfinite(flux_i).any() else 1.0
        if not np.isfinite(med) or med == 0: med = 1.0
        flux_i = np.clip(flux_i/med, 0.97, 1.03).astype(np.float32)

        X = flux_i.reshape(1, -1, 1)
        p = float(g["cnn_model"].predict(X, verbose=0).ravel()[0])
        thr = float(g["cnn_thresholds"].get(mode, 0.5))
        pred = int(p >= thr)
        return jsonify({"ok": True, "cnn_probability": round(p,6), "cnn_label": "TRANSIT" if pred else "NON_TRANSIT", "cnn_threshold": thr})
    except Exception as e:
        return _json_error("cnn lightcurve prediction failed", 500, traceback.format_exc())

# ---------------- ENSEMBLE (full: tabular + synthetic LC + CNN) ----------------
@app.route("/api/predict/ensemble", methods=["POST"])
def predict_ensemble():
    """
    Front-end posts the same flat JSON as /api/predict/tabular.
    We compute:
      - tabular probability from the form
      - synthetic LC around phase 0, run CNN (if available)
      - fuse probs: final = alpha*tab + (1-alpha)*cnn
    Response matches the UI contract.
    """
    try:
        payload = request.get_json(silent=True) or {}
        alpha = float(payload.get("alpha", 0.7))  # tabular weight
        mode  = payload.get("threshold_mode", "default")

        # --- tabular ---
        X = _ensure_matrix(g["feature_list"], payload)
        X_imp = _apply_imputer_and_log(X)
        clf = g["models"]["SELECTED"]
        p_tab = float(clf.predict_proba(X_imp)[0, 1])

        # --- synthetic LC (phase/model/observed) ---
        n = g["cnn_input_len"]
        half_w = 0.10
        depth = float(payload.get("koi_depth", 500.0)) / 1_000_000.0   # ppm→fraction (~0.005 defaults to 5000 ppm if large)
        depth = float(np.clip(depth, 0.001, 0.05))
        duration = 0.06
        noise = 0.001
        rng = np.random.default_rng(42)
        phase = np.linspace(-half_w, +half_w, n)
        flux_model = np.ones(n); in_tr = np.abs(phase) <= (duration/2.0); flux_model[in_tr] -= depth
        flux_obs = flux_model + rng.normal(0.0, noise, size=n)

        # --- CNN on observed ---
        if g["has_cnn"]:
            med = np.nanmedian(flux_obs); med = 1.0 if (not np.isfinite(med) or med==0) else med
            arr = np.clip(flux_obs/med, 0.97, 1.03).astype(np.float32).reshape(1, -1, 1)
            p_cnn = float(g["cnn_model"].predict(arr, verbose=0).ravel()[0])
        else:
            # heuristic fallback if CNN not available
            dip = 1.0 - np.min(flux_obs)
            p_cnn = float(np.clip(dip / 0.02, 0, 1))

        # --- fuse ---
        p_final = float(alpha * p_tab + (1.0 - alpha) * p_cnn)
        thr = _get_thr_for("selected_model", mode)
        label = "CONFIRMED" if p_final >= thr else "FALSE_POSITIVE"

        return jsonify({
            "ok": True,
            "final_ensemble_probability": p_final,
            "final_predicted_label": label,
            "tabular_only_result": {
                "Ensemble_Probability": p_tab,
                "Predicted_Label": ("CONFIRMED" if p_tab >= thr else "FALSE_POSITIVE"),
                "Decision_Threshold_Used": thr
            },
            "light_curve_data": {
                "found": True,
                "id": payload.get("kic_id") or payload.get("tic_id") or "SYNTH",
                "phase": phase.tolist(),
                "flux_model": flux_model.tolist(),
                "flux_observed": flux_obs.tolist()
            }
        })
    except Exception as e:
        return _json_error("ensemble prediction failed", 500, traceback.format_exc())

# ---------------- Synthetic LC (for UI testing) ----------------
@app.route("/api/lightcurve/synthetic", methods=["POST"])
def lightcurve_synth():
    try:
        payload = request.get_json(silent=True) or {}
        n        = int(payload.get("n", g.get("cnn_input_len", 201)))
        half_w   = float(payload.get("phase_half_width", 0.10))
        depth    = float(payload.get("depth", 0.01))
        duration = float(payload.get("duration", 0.06))
        noise    = float(payload.get("noise", 0.001))
        seed     = int(payload.get("seed", 42))

        rng   = np.random.default_rng(seed)
        phase = np.linspace(-half_w, +half_w, n)
        model = np.ones(n, dtype=float)
        in_tr = np.abs(phase) <= (duration / 2.0)
        model[in_tr] -= depth
        observed = model + rng.normal(0.0, noise, size=n)

        return jsonify({"ok": True, "phase": phase.tolist(), "model": model.tolist(), "observed": observed.tolist()})
    except Exception as e:
        return _json_error("synthetic LC failed", 500, traceback.format_exc())

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    print("Thresholds:", {**g["thresholds"], "CNN": g["cnn_thresholds"]})
    app.run(host=HOST, port=PORT, debug=DEBUG)
