from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import joblib
import tensorflow as tf
import logging

# Minimal custom components to allow loading the saved model
@tf.keras.utils.register_keras_serializable()
class ElasticGatedResidualLayer(tf.keras.layers.Layer):
	def __init__(self, units=32, **kwargs):
		super().__init__(**kwargs)
		self.units = units

	def build(self, input_shape):
		self.dense1 = tf.keras.layers.Dense(self.units, activation='gelu')
		self.dense2 = tf.keras.layers.Dense(self.units)
		self.gate = tf.keras.layers.Dense(self.units, activation='sigmoid')
		self.residual = tf.keras.layers.Dense(self.units)
		super().build(input_shape)

	def call(self, inputs):
		x = self.dense1(inputs)
		x = self.dense2(x)
		g = self.gate(inputs)
		res = self.residual(inputs)
		return x * g + res * (1.0 - g)

	def get_config(self):
		config = super().get_config()
		config.update({"units": self.units})
		return config


@tf.keras.utils.register_keras_serializable()
class AdherenceRegLayer(tf.keras.layers.Layer):
	def __init__(self, scale=200.0, threshold=0.5, **kwargs):
		super().__init__(**kwargs)
		self.scale = scale
		self.threshold = threshold

	def call(self, inputs):
		return tf.keras.activations.sigmoid((inputs[:, 1:2] - self.threshold) * self.scale)

	def get_config(self):
		config = super().get_config()
		config.update({"scale": self.scale, "threshold": self.threshold})
		return config


@tf.keras.utils.register_keras_serializable()
class RobustSparseFocalLoss(tf.keras.losses.Loss):
	def __init__(self, gamma=2.0, class_weights=None, label_smoothing=0.0, initial_weight=1.0, **kwargs):
		super().__init__(**kwargs)
		self.gamma = gamma
		self.class_weights = class_weights
		self.label_smoothing = label_smoothing
		self.initial_weight = initial_weight
		self.dynamic_weight = tf.Variable(initial_weight, dtype=tf.float32, trainable=False)

	def call(self, y_true, y_pred):
		y_true = tf.cast(y_true, tf.int32)
		num_classes = tf.shape(y_pred)[-1]
		y_true_oh = tf.one_hot(y_true, depth=num_classes)
		y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
		cross_entropy = -y_true_oh * tf.math.log(y_pred)
		weight = tf.pow(1.0 - y_pred, self.gamma)
		focal_loss = weight * cross_entropy
		if self.class_weights is not None:
			class_w = tf.convert_to_tensor(self.class_weights, dtype=tf.float32)
			focal_loss = focal_loss * class_w
		return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1)) * self.dynamic_weight


# --- FastAPI app ---
app = FastAPI(title="Kepatuhan Model API")


class PredictRequest(BaseModel):
	all_features: List[float]
	selected_features: List[float]


class PredictResponse(BaseModel):
	perception: Dict[str, Any]
	behaviour: Dict[str, Any]
	adherence: Dict[str, Any]
	adherence_reg: float


# Try to load model and scalers on startup
MODEL_PATH = "model_kepatuhan.keras"
SCALER_PATH = "scaler.pkl"
SCALER_SEL_PATH = "scaler_select.pkl"

_model = None
_scaler = None
_scaler_sel = None


logger = logging.getLogger("uvicorn.error")


def load_assets():
	global _model, _scaler, _scaler_sel
	_model = None
	_scaler = None
	_scaler_sel = None

	# Load scaler
	try:
		_scaler = joblib.load(SCALER_PATH)
		logger.info("Loaded scaler from %s", SCALER_PATH)
	except Exception as e:
		_scaler = None
		logger.warning("Failed to load scaler: %s", e)

	# Load selected scaler
	try:
		_scaler_sel = joblib.load(SCALER_SEL_PATH)
		logger.info("Loaded selected scaler from %s", SCALER_SEL_PATH)
	except Exception as e:
		_scaler_sel = None
		logger.warning("Failed to load selected scaler: %s", e)

	# Load model (may contain custom objects)
	try:
		_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
		logger.info("Loaded model from %s", MODEL_PATH)
	except Exception as e:
		_model = None
		logger.warning("Failed to load model: %s", e)


@app.on_event("startup")
def on_startup():
	logger.info("Starting API and loading assets...")
	load_assets()


@app.get("/health")
def health():
	return {
		"status": "ok",
		"model_loaded": _model is not None,
		"scaler_loaded": _scaler is not None,
		"scaler_select_loaded": _scaler_sel is not None,
	}


@app.get("/")
def root():
	# Redirect to interactive docs for convenience
	return RedirectResponse(url="/docs")


@app.post("/reload")
def reload_assets():
	try:
		load_assets()
		return JSONResponse({
			"status": "reloaded",
			"model_loaded": _model is not None,
			"scaler_loaded": _scaler is not None,
			"scaler_select_loaded": _scaler_sel is not None,
		})
	except Exception as e:
		logger.exception("Error reloading assets")
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
	if _model is None:
		raise HTTPException(status_code=503, detail="Model not loaded")
	if _scaler is None or _scaler_sel is None:
		raise HTTPException(status_code=503, detail="Scalers not loaded")

	x_all = np.array(req.all_features, dtype=float).reshape(1, -1)
	x_sel = np.array(req.selected_features, dtype=float).reshape(1, -1)

	# Validate dimensions using scaler attributes when available
	try:
		if hasattr(_scaler, "mean_") and x_all.shape[1] != _scaler.mean_.shape[0]:
			raise HTTPException(status_code=400, detail=f"all_features length must be {_scaler.mean_.shape[0]}")
		if hasattr(_scaler_sel, "mean_") and x_sel.shape[1] != _scaler_sel.mean_.shape[0]:
			raise HTTPException(status_code=400, detail=f"selected_features length must be {_scaler_sel.mean_.shape[0]}")
	except HTTPException:
		raise
	except Exception:
		pass

	x_all_s = _scaler.transform(x_all)
	x_sel_s = np.nan_to_num(_scaler_sel.transform(x_sel))

	preds = _model.predict([x_all_s, x_sel_s])

	prob_perc = preds[0][0]
	prob_behav = preds[1][0]
	prob_adher = preds[2][0]
	reg_val = float(np.array(preds[3]).flatten()[0])

	label_perception = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
	label_behaviour = {0: 'Perilaku Buruk', 1: 'Perilaku Baik'}
	label_adherence = {0: 'Non-Adherent', 1: 'Adherent'}

	perc_class = int(np.argmax(prob_perc))
	behav_class = int(np.argmax(prob_behav))
	adher_class = int(np.argmax(prob_adher))

	return {
		"perception": {"label": label_perception.get(perc_class, str(perc_class)), "class": perc_class, "probabilities": prob_perc.tolist()},
		"behaviour": {"label": label_behaviour.get(behav_class, str(behav_class)), "class": behav_class, "probabilities": prob_behav.tolist()},
		"adherence": {"label": label_adherence.get(adher_class, str(adher_class)), "class": adher_class, "probabilities": prob_adher.tolist()},
		"adherence_reg": reg_val,
	}


if __name__ == "__main__":
	import uvicorn
	uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")

