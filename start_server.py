import os
import subprocess

result = subprocess.run(
    ["uv", "run", "fastapi", "dev"],
    env={**os.environ, "ALLOWED_ORIGINS": "http://localhost:5500"},
    check=True,
)
