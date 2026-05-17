.PHONY: install generate train-offline serve frontend-dev evaluate demo clean

install:
	uv sync
	cd frontend && npm install

generate:
	python -m scripts.run_generator --config config/generator_config.json

train-offline:
	python -m scripts.run_offline_pipeline

serve:
	uvicorn system2_platform.api.main:app --reload --port 8000

frontend-dev:
	cd frontend && npm run dev

evaluate:
	python -m scripts.run_evaluator

demo: clean generate train-offline
	@echo "Starting API in background..."
	uvicorn system2_platform.api.main:app --port 8000 &
	sleep 5
	python scripts/replay_stream.py --rate 50
	$(MAKE) evaluate

clean:
	rm -rf data/raw/*.csv data/artifacts/* data/evaluation/*
