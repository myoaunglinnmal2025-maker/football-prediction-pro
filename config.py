from pathlib import Path
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
PRED_DIR = ROOT / "predictions"
MATCHES_CSV = DATA_DIR / "matches.csv"
FIXTURES_CSV = DATA_DIR / "fixtures.csv"
MODEL_FILE = MODEL_DIR / "goal_model.joblib"
PREDICTIONS_CSV = PRED_DIR / "predictions.csv"
