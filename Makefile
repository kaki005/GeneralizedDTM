# .PHONYで定義することで、タスクランナーを簡単に定義できます

# install
.PHONY: install
install:
	uv python pin 3.13
	uv sync
	uv run pre-commit install

# main.pyを実行する
.PHONY: run_main
run_main:
	uv run src/main.py

# pre-commitを明示的に実行する
.PHONY: pre-commit
pre-commit:
	uv run pre-commit run --all-files

.PHONY: test
test:
	uv run pytest tests

.PHONY: sweep
sweep:
	uv run wandb src/config/sweep.yaml
