# News Analyzer - Django Project with RuBERT NLI

A Django web application that performs Natural Language Inference (NLI) on news articles using the RuBERT model. The application scrapes news from various sources and classifies them based on user-defined class pairs.

## Features

- **News Scraping**: Automated news collection from various sources
- **NLI Classification**: Zero-shot classification using RuBERT model
- **REST API**: RESTful endpoints for task management
- **Async Processing**: Background task processing for news analysis
- **Docker Support**: Containerized deployment
- **CI/CD Pipeline**: Automated testing and deployment with GitHub Actions

## Tech Stack

- **Backend**: Django 4.2+, Django REST Framework
- **ML/NLP**: PyTorch, Transformers (RuBERT)
- **Database**: SQLite (simple file-based database)
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
   docker-compose up --build
   ```

4. **Access the application**
   - API: http://localhost:8000
   - Admin: http://localhost:8000/admin

### Local Development Setup

1. **Prerequisites**
   - Python 3.9+

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment setup**
   ```bash
   cp .env.example .env
   # Configure your environment variables
   ```

4. **Database setup**
   ```bash
   python manage.py migrate
   python manage.py createsuperuser  # Optional
   ```

5. **Run development server**
   ```bash
   python manage.py runserver
   ```

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
# Run all tests
export DJANGO_SETTINGS_MODULE=news_analyzer.settings
pytest

# Run with coverage
pytest --cov

# Run specific test file
pytest analyzer/tests.py

# Run with verbose output
pytest -v
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
│   ├── models.py            # API views and models
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

### RuBERT Model

The application uses the `cointegrated/rubert-base-cased-nli-threeway` model for Russian text NLI. The model is automatically downloaded on first use.

### Database

The application uses SQLite as a simple file-based database. No complex database setup is required.

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