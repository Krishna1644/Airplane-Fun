@echo off
echo Starting Flight Risk Application...

echo Starting API Server...
start cmd /k "cd api && uvicorn main:app --reload"

echo Starting React Client...
start cmd /k "cd client && npm run dev"

echo Both servers are starting up in separate windows!
