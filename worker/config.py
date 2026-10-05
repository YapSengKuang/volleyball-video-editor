import os


def _int(name: str, fallback: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return fallback
    try:
        return int(raw)
    except ValueError:
        return fallback


def _float(name: str, fallback: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return fallback
    try:
        return float(raw)
    except ValueError:
        return fallback


def _r2_endpoint() -> str:
    account = os.environ.get("R2_ACCOUNT_ID", "").strip()
    if not account:
        return ""
    return f"https://{account}.r2.cloudflarestorage.com"


DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgres://volleyball:volleyball@localhost:5432/volleyball"
)
_R2_ENDPOINT = _r2_endpoint()
S3_ENDPOINT = _R2_ENDPOINT or os.environ.get("S3_ENDPOINT", "http://localhost:9000")
S3_BUCKET = os.environ.get("R2_BUCKET") or os.environ.get("S3_BUCKET", "videos")
S3_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("S3_ACCESS_KEY", "minio")
S3_SECRET_KEY = os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("S3_SECRET_KEY", "minio-secret-key")
USE_R2 = bool(_R2_ENDPOINT) or "r2.cloudflarestorage.com" in S3_ENDPOINT
S3_REGION = "auto" if USE_R2 else os.environ.get("S3_REGION", "us-east-1")
WEB_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "WEB_ORIGIN", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]
WORK_DIR = os.environ.get("WORK_DIR", "/work")
MAX_DURATION_SECONDS = _float("MAX_DURATION_SECONDS", 2.5 * 60 * 60)
FFMPEG_THREADS = _int("FFMPEG_THREADS", 0)
PROXY_FPS = _float("PROXY_FPS", 5)
PROXY_HEIGHT = _int("PROXY_HEIGHT", 320)
BALL_IMGSZ = _int("BALL_IMGSZ", 320)
BALL_THREADS = _int("BALL_THREADS", 0)
BALL_DEVICE = os.environ.get("BALL_DEVICE", "auto")
ANALYZE_TIMEOUT_SECONDS = _int("ANALYZE_TIMEOUT_SECONDS", 60 * 60)
EXPORT_TIMEOUT_SECONDS = _int("EXPORT_TIMEOUT_SECONDS", 3 * 60 * 60)
RETENTION_HOURS = _int("RETENTION_HOURS", 24)
