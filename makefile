.PHONY: app verify-data

app:
	@clear
	@uv run --with uvicorn uvicorn app:app --host 0.0.0.0 --port 8000

verify-data:
	uv run python scripts/verify_asvspoof2019_la.py --no-report