.PHONY: app test verify-data audit-data evaluate evaluate-rerun freeze-baseline \
	e1-1-export e1-1-train e1-1-evaluate e1-1-summarize e1-1

CHECKPOINTS = \
	--checkpoint Models/Iteration1/best_audio_spoof_model_unweighted.pth \
	--checkpoint Models/Iteration1/best_audio_spoof_model_weighted.pth

E0_3_FLAGS = --experiment-id E0.3 --title "Frozen two-checkpoint baseline"
RERUN_ROOT ?= /tmp/opencode/e0.3-rerun

E1_1_FRONTENDS = mel64 mel128 mel256 linear_stft mfcc40 randproj128
E1_1_IMAGE_ROOT ?= data/derived/e1_1_frontend
E1_1_MODEL_ROOT ?= Models/Iteration1/e_1_1
E1_1_REPORT_ROOT ?= reports/phase1/e1_1_frontend

app:
	@clear
	@uv run --with uvicorn uvicorn app:app --host 0.0.0.0 --port 8000

test:
	uv run python -m unittest discover -s tests -p "test_*.py" -v

verify-data:
	uv run python scripts/verify_asvspoof2019_la.py --no-report

audit-data:
	uv run python scripts/audit_data.py

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

e1-1-export:
	uv run python scripts/export_e1_1_frontends.py \
		--frontends $(E1_1_FRONTENDS) \
		--output-root $(E1_1_IMAGE_ROOT) \
		--manifest-root manifests/e1_1_frontend

e1-1-train:
	uv run python scripts/train_e1_1_probes.py \
		--frontends $(E1_1_FRONTENDS) \
		--image-root $(E1_1_IMAGE_ROOT) \
		--checkpoint-root $(E1_1_MODEL_ROOT)

e1-1-evaluate:
	uv run python scripts/evaluate_e1_1.py \
		--frontends $(E1_1_FRONTENDS) \
		--checkpoint-root $(E1_1_MODEL_ROOT) \
		--image-root $(E1_1_IMAGE_ROOT) \
		--output-root $(E1_1_REPORT_ROOT)/runs

e1-1-summarize:
	uv run python scripts/summarize_e1_1.py \
		--runs-root $(E1_1_REPORT_ROOT)/runs \
		--output-root $(E1_1_REPORT_ROOT)

e1-1: e1-1-export e1-1-train e1-1-evaluate e1-1-summarize
