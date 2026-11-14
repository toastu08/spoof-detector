.PHONY: app test verify-data evaluate evaluate-rerun freeze-baseline

CHECKPOINTS = \
	--checkpoint Models/Itteration1/best_audio_spoof_model_unweighted.pth \
	--checkpoint Models/Itteration1/best_audio_spoof_model_weighted.pth

E0_3_FLAGS = --experiment-id E0.3 --title "Frozen two-checkpoint baseline"
RERUN_ROOT ?= /tmp/opencode/e0.3-rerun

app:
	@clear
	@uv run --with uvicorn uvicorn app:app --host 0.0.0.0 --port 8000

test:
	uv run python -m unittest discover -s tests -p "test_*.py" -v

verify-data:
	uv run python scripts/verify_asvspoof2019_la.py --no-report

evaluate:
	uv run python -m spoof_detection.evaluate \
		$(CHECKPOINTS) \
		--asvspoof dev eval --source image --output-root reports/baseline \
		$(E0_3_FLAGS)

evaluate-rerun:
	uv run python -m spoof_detection.evaluate \
		$(CHECKPOINTS) \
		--asvspoof dev eval --source image --output-root $(RERUN_ROOT) \
		$(E0_3_FLAGS)

freeze-baseline: evaluate evaluate-rerun
	uv run python scripts/freeze_baseline.py \
		--baseline-root reports/baseline \
		--rerun-root $(RERUN_ROOT) \
		--manifest manifests/train_seed42.csv \
		--output reports/baseline/freeze_report.json
