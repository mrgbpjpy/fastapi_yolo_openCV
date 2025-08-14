{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "./backend/Dockerfile"
  },
  "deploy": {
    "runtime": "V2",
    "startCommand": "uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1",
    "sleepApplication": false,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10,
    "multiRegionConfig": {
      "us-east4-eqdc4a": {
        "numReplicas": 1
      }
    },
    "healthcheckPath": "/health",
    "healthcheckTimeout": 300
  }
}
