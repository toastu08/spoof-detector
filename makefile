.PHONY: app verify-data evaluate

app:
	@clear
	@uv run --with uvicorn uvicorn app:app --host 0.0.0.0 --port 8000

verify-data:
	uv run python scripts/verify_asvspoof2019_la.py --no-report

evaluate:
	uv run python -m spoof_detection.evaluate \
		--checkpoint Models/Itteration1/best_audio_spoof_model_unweighted.pth \
		--asvspoof dev eval --source image --output-root reports/baseline
