.PHONY: app test verify-data audit-data evaluate evaluate-rerun freeze-baseline \
	e1-1-export e1-1-train e1-1-evaluate e1-1-summarize e1-1 \
	e1-2-export e1-2-train e1-2-evaluate e1-2-summarize e1-2 \
	e1-4-export e1-4-train e1-4-evaluate e1-4-summarize e1-4 e1-4-trim

CHECKPOINTS = \
	--checkpoint Models/Iteration1/best_audio_spoof_model_unweighted.pth \
	--checkpoint Models/Iteration1/best_audio_spoof_model_weighted.pth

E0_3_FLAGS = --experiment-id E0.3 --title "Frozen two-checkpoint baseline"
RERUN_ROOT ?= /tmp/opencode/e0.3-rerun

E1_1_FRONTENDS = mel64 mel128 mel256 linear_stft mfcc40 randproj128
E1_1_IMAGE_ROOT ?= data/derived/e1_1_frontend
E1_1_MODEL_ROOT ?= Models/Iteration1/e_1_1
E1_1_REPORT_ROOT ?= reports/phase1/e1_1_frontend

E1_2_CONDITIONS = magma_flip viridis_flip gray_flip rawdb_flip \
	magma_noflip viridis_noflip gray_noflip magma_flip_rgb
E1_2_SEEDS ?= 42
E1_2_IMAGE_ROOT ?= data/derived/e1_2_rendering
E1_2_MODEL_ROOT ?= Models/Iteration1/e_1_2
E1_2_REPORT_ROOT ?= reports/phase1/e1_2_rendering

E1_4_CONDITIONS = full speech_only nonspeech_only
E1_4_FEATURE_PROBES = duration_only metadata_system_id metadata_speaker_id
E1_4_SEEDS ?= 42
E1_4_IMAGE_ROOT ?= data/derived/e1_4_shortcuts
E1_4_MODEL_ROOT ?= Models/Iteration1/e_1_4
E1_4_REPORT_ROOT ?= reports/phase1/e1_4_shortcuts

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

e1-2-export:
	uv run python scripts/export_e1_2_rendering.py \
		--conditions $(E1_2_CONDITIONS) \
		--output-root $(E1_2_IMAGE_ROOT) \
		--manifest-root manifests/e1_2_rendering \
		--report-path $(E1_2_REPORT_ROOT)/export_manifest.json

e1-2-train:
	uv run python scripts/train_e1_2_probes.py \
		--conditions $(E1_2_CONDITIONS) \
		--seeds $(E1_2_SEEDS) \
		--image-root $(E1_2_IMAGE_ROOT) \
		--checkpoint-root $(E1_2_MODEL_ROOT)

e1-2-evaluate:
	uv run python scripts/evaluate_e1_2.py \
		--conditions $(E1_2_CONDITIONS) \
		--seeds $(E1_2_SEEDS) \
		--checkpoint-root $(E1_2_MODEL_ROOT) \
		--image-root $(E1_2_IMAGE_ROOT) \
		--output-root $(E1_2_REPORT_ROOT)/runs

e1-2-summarize:
	uv run python scripts/summarize_e1_2.py \
		--conditions $(E1_2_CONDITIONS) \
		--seeds $(E1_2_SEEDS) \
		--runs-root $(E1_2_REPORT_ROOT)/runs \
		--output-root $(E1_2_REPORT_ROOT)

e1-2: e1-2-export e1-2-train e1-2-evaluate e1-2-summarize

e1-4-export:
	uv run python scripts/export_e1_4_shortcuts.py \
		--conditions $(E1_4_CONDITIONS) \
		--output-root $(E1_4_IMAGE_ROOT) \
		--manifest-root manifests/e1_4_shortcuts \
		--report-path $(E1_4_REPORT_ROOT)/export_manifest.json

e1-4-train:
	uv run python scripts/train_e1_4_probes.py \
		--conditions $(E1_4_CONDITIONS) \
		--feature-probes $(E1_4_FEATURE_PROBES) \
		--seeds $(E1_4_SEEDS) \
		--image-root $(E1_4_IMAGE_ROOT) \
		--checkpoint-root $(E1_4_MODEL_ROOT)

e1-4-evaluate:
	uv run python scripts/evaluate_e1_4.py \
		--conditions $(E1_4_CONDITIONS) \
		--feature-probes $(E1_4_FEATURE_PROBES) \
		--seeds $(E1_4_SEEDS) \
		--checkpoint-root $(E1_4_MODEL_ROOT) \
		--image-root $(E1_4_IMAGE_ROOT) \
		--output-root $(E1_4_REPORT_ROOT)/runs

e1-4-summarize:
	uv run python scripts/summarize_e1_4.py \
		--conditions $(E1_4_CONDITIONS) \
		--feature-probes $(E1_4_FEATURE_PROBES) \
		--seeds $(E1_4_SEEDS) \
		--runs-root $(E1_4_REPORT_ROOT)/runs \
		--output-root $(E1_4_REPORT_ROOT)

e1-4-trim:
	uv run python scripts/e1_4_silence_trim.py \
		--output $(E1_4_REPORT_ROOT)/silence_trim_bonafide

e1-4: e1-4-export e1-4-train e1-4-evaluate e1-4-summarize
