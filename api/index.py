from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
import os

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load data on cold start
with open("q-vercel-latency.json", "r") as f:
    telemetry = json.load(f)

@app.post("/")
async def check_latency(request: Request):
    payload = await request.json()
    regions = payload["regions"]
    threshold_ms = payload["threshold_ms"]
    
    results = {}
    for region in regions:
        recs = [r for r in telemetry if r["region"] == region]
        lat = np.array([r["latency_ms"] for r in recs])
        up = np.array([r["uptime_pct"] for r in recs])
        
        results[region] = {
            "avg_latency": float(np.mean(lat)),
            "p95_latency": float(np.percentile(lat, 95)),
            "avg_uptime": float(np.mean(up)),
            "breaches": int(np.sum(lat > threshold_ms))
        }
    return results
