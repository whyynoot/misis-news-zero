# Social Risk Monitor

A Django application for monitoring social risks in Russian news flows. The primary analysis path is a local LLM through Ollama; RuBERT remains available as a fallback. The system stores factor-level sentiment history in PostgreSQL and lets you inspect the exact news items that caused spikes.

## Features

- **News Scraping**: Automated news collection from TASS and Interfax
- **Predefined Social Risk Factors**: A fixed factor catalog for repeatable monitoring
- **Sentiment History**: Per-factor daily history with scores from `-1` to `1`
- **News Drill-down**: Inspect the news item behind a positive or negative spike
- **REST API**: RESTful endpoints for task management
- **Async Processing**: Background task processing for news analysis
- **Docker Support**: Containerized deployment
- **Monitoring Dashboard**: Historical sentiment, spike review, recent signal-bearing news
- **Historical Backfill**: Fetch archive-capable sources by date, store real news, classify them, and rebuild dashboard aggregates
- **CI/CD Pipeline**: Automated testing and deployment with GitHub Actions

## Tech Stack

- **Backend**: Django 4.2+, Django REST Framework
- **Database**: PostgreSQL for application/runtime storage
- **ML/NLP**: Ollama LLM primary path, PyTorch/Transformers RuBERT fallback
- **Containerization**: Docker, Docker Compose
- **Testing**: Pytest, pytest-django
- **CI/CD**: GitHub Actions

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd news-zero-shot
   ```

2. **Create environment file**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Run with Docker Compose**
   ```bash
   docker compose up -d --build db web scheduler
   ```

4. **Access the application**
   - Dashboard: http://localhost:8000
   - PostgreSQL: localhost:5432 by default

### Local Development Setup

1. **Prerequisites**
   - Python 3.9+
   - PostgreSQL 14+ or `docker compose up db`

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment setup**
   ```bash
   cp .env.example .env
   # Configure your environment variables
   ```

4. **Run development server**
   ```bash
   python manage.py migrate
   python manage.py runserver
   ```

### Monitoring Workflow

- Run a real historical LLM backfill for the last year:

```bash
./scripts/backfill_history.ps1 -Days 365 -Sources interfax -LimitPerDay 8 -Engine llm -CleanSeed
```

- Run one live LLM update:

```bash
./scripts/run_live_update.ps1 -Limit 50 -Engine llm
```

- Start web + the 4-hour LLM scheduler:

```bash
./scripts/start_scheduler.ps1 -Build
```

- Direct Django commands are also available:

```bash
python manage.py backfill_news_history --days 365 --sources interfax --limit-per-day 8 --engine llm --clean-seed
python manage.py run_monitoring --engine llm --summarize --period day --limit 50
python manage.py run_monitoring_scheduler --interval-hours 4 --engine llm --summarize --period day
```

- Dashboard: http://localhost:8000/monitoring/  
- Data API: `GET /api/monitoring/` (overview + factor detail), `POST /api/monitoring/run/` (manual live refresh), `POST /api/monitoring/history/run/` (historical archive backfill)

On a fresh Docker start the web container runs migrations and syncs the factor catalog. The scheduler container then runs LLM live updates every 4 hours. Demo seed data is not loaded automatically.

## API Usage

### Create Analysis Task

```bash
POST /api/task/
Content-Type: application/json

{
    "pairs": [
        {
            "class1": "positive",
            "class2": "negative"
        },
        {
            "class1": "business",
            "class2": "politics"
        }
    ]
}
```

### Check Task Status

```bash
GET /api/task/{task_id}/
```

Response:
```json
{
    "status": "Complete",
    "result": {
        "news_results": [...],
        "summary": {...}
    },
    "error": null
}
```

## Testing

### Run Tests

```bash
export DJANGO_SETTINGS_MODULE=news_analyzer.settings
pytest analyzer/tests.py --cov=analyzer --cov=news_analyzer --cov-report=term --cov-fail-under=80
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Complete workflow testing
- **API Tests**: REST endpoint testing

## Deployment

### Production with Docker

1. **Build production image**
   ```bash
   docker build -t news-analyzer:prod .
   ```

2. **Environment configuration**
   ```bash
   # Set production environment variables
   export DEBUG=False
   export SECRET_KEY=your-production-secret-key
   export ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
   ```

3. **Run production container**
   ```bash
   docker run -d \
     --name news-analyzer \
     -p 8000:8000 \
     -e DEBUG=False \
     -e SECRET_KEY=$SECRET_KEY \
     -e ALLOWED_HOSTS=$ALLOWED_HOSTS \
     news-analyzer:prod
   ```

### CI/CD Pipeline

The project includes a GitHub Actions pipeline that:

1. **Tests**: Runs on Python 3.9, 3.10, 3.11
2. **Security**: Bandit security checks, Safety dependency checks
3. **Code Quality**: Flake8 linting, Black formatting, isort imports
4. **Coverage**: Code coverage reporting with Codecov
5. **Build**: Docker image building and pushing
6. **Deploy**: Automated deployment to production

### Required Secrets

Configure these secrets in your GitHub repository:

- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub password

## Project Structure

```
news-zero-shot/
├── analyzer/                 # Main Django app
│   ├── models.py            # ORM models
│   ├── api.py               # REST API views
│   ├── tasks.py             # Background task processing
│   ├── views.py             # Web views
│   ├── zero.py              # RuBERT classifier
│   └── tests.py             # Test suite
├── news/                    # News scraping module
├── news_analyzer/           # Django project settings
│   ├── settings.py          # Configuration
│   ├── urls.py              # URL routing
│   └── wsgi.py              # WSGI application
├── .github/workflows/       # CI/CD pipeline
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container definition
├── docker-compose.yml      # Local development setup
├── pytest.ini             # Test configuration
└── README.md               # This file
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Django secret key | Required |
| `DEBUG` | Debug mode | `True` |
| `ALLOWED_HOSTS` | Allowed host names | `localhost,127.0.0.1` |
| `DJANGO_LOG_LEVEL` | Logging level | `INFO` |
| `HF_CACHE_DIR` | Local Hugging Face cache path for Docker bind mount | `./.cache/huggingface` |
| `HF_TOKEN` | Optional Hugging Face token for higher rate limits and more stable model warmup | empty |
| `USE_CUDA` | Enable CUDA inference when available inside the container | `False` |
| `MODEL_BATCH_SIZE` | Batch size for factor inference | `8` |
| `LLM_ENABLED` | Enable local LLM analysis | `True` in Docker Compose |
| `LLM_BASE_URL` | Local Ollama URL for non-Docker runs | `http://localhost:11434` |
| `LLM_DOCKER_BASE_URL` | Ollama URL visible from Docker containers | `http://host.docker.internal:11434` |
| `LLM_MODEL` | Ollama model name | `gemma4:e2b` |
| `LLM_BATCH_NEWS_SIZE` | News items per LLM classification request | `2` |
| `DATABASE_ENGINE` | `postgresql` or explicit `sqlite` fallback for tests/local diagnostics | `postgresql` |
| `POSTGRES_DB` | PostgreSQL database name | `news_analyzer` |
| `POSTGRES_USER` | PostgreSQL user | `news_analyzer` |
| `POSTGRES_PASSWORD` | PostgreSQL password | `news_analyzer` |
| `POSTGRES_HOST` | PostgreSQL host | `localhost` locally, `db` in Compose |
| `POSTGRES_PORT` | PostgreSQL port | `5432` |
| `NEWS_SOURCES` | Comma-separated live news sources | `tass,interfax` |
| `INTERFAX_SECTIONS` | Interfax sections to scan | `russia,business,world` |
| `MONITORING_HISTORY_SOURCES` | Archive-capable historical sources | `interfax` |
| `MONITORING_HISTORY_LIMIT_PER_DAY` | Historical articles per day/source | `8` |
| `MONITORING_UPDATE_INTERVAL_HOURS` | Scheduler interval | `4` |

### Analysis Engines

LLM/Ollama is the primary analytical engine. In Docker, the app connects to host Ollama through `host.docker.internal:11434`; Ollama itself is responsible for GPU execution. `ollama ps` should show the model running on GPU when a classification is active. RuBERT (`cointegrated/rubert-base-cased-nli-threeway`) is retained as a fallback and for deterministic tests.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For questions and support, please open an issue in the GitHub repository. 
