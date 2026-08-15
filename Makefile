.PHONY: help db-up db-down db-logs db-ps db-shell migrate revision upgrade downgrade run dev install clean

help:
	@echo "Available commands:"
	@echo "  make db-up          - Start PostgreSQL database"
	@echo "  make db-down        - Stop PostgreSQL database"
	@echo "  make db-logs        - Show database logs"
	@echo "  make db-ps          - Show database container status"
	@echo "  make db-shell       - Connect to PostgreSQL shell"
	@echo "  make migrate        - Create new migration (use MESSAGE='...')"
	@echo "  make revision       - Create empty migration (use MESSAGE='...')"
	@echo "  make upgrade        - Apply migrations"
	@echo "  make downgrade      - Rollback one migration"
	@echo "  make run            - Run FastAPI server"
	@echo "  make dev            - Run FastAPI server with reload"
	@echo "  make install        - Install dependencies"
	@echo "  make clean          - Clean Python cache files"

db-up:
	docker-compose up -d

db-down:
	docker-compose down

db-logs:
	docker-compose logs -f postgres

docker-ps:
	docker-compose ps

db-shell:
	docker-compose exec postgres psql -U $$POSTGRES_USER -d $$POSTGRES_DB

migrate:
	python -m alembic revision --autogenerate -m "$(M)"

upgrade:
	python -m alembic upgrade head

downgrade:
	python -m alembic downgrade -1

run:
	python -m uvicorn api.server:app --host 0.0.0.0 --port 8008 --workers 2

dev:
	python -m uvicorn api.server:app --host 0.0.0.0 --port 8008 --reload --workers 2

fe:
	cd frontend && npm run dev
install:
	pip install -r requirements.txt

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

