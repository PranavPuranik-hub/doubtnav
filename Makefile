.PHONY: setup run

setup:
	cd frontend && npm install
	python -m pip install -r requirements.txt
	python -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')"

run:
	python -c "import os, subprocess; (not os.path.exists('frontend/dist')) and subprocess.run('npm run build', cwd='frontend', shell=True)"
	python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
