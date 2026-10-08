# Wigglegram API

Turns one photo into a 3D wigglegram (Depth Anything 3). Deployed with [deploy/](../deploy/).

To start locally, run:
`DEVICE=cpu PYTHONPATH=src .venv/bin/uvicorn wigglegram.main:app --port 7860 --reload`
