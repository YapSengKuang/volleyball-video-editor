"""Start the cutter against the hosted database and video store."""

import os
from pathlib import Path

env_file = Path("/env/.env.local")
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key] = value.strip().strip('"')
    unpooled = os.environ.get("DATABASE_URL_UNPOOLED")
    if unpooled:
        os.environ["DATABASE_URL"] = unpooled

os.makedirs(os.environ.get("WORK_DIR", "/work"), exist_ok=True)

from main import main

main()
