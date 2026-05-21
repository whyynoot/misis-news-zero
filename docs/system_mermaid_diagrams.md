# Диаграммы системы мониторинга социальных рисков

Документ описывает исследовательскую платформу `Social Risk Monitor`: сбор новостей, классификацию сигналов по социальным факторам, хранение результатов и пользовательское взаимодействие через dashboard/API.

## Используемые технологии

| Слой | Технологии |
| --- | --- |
| Web/API | Django 4.2+, Django REST Framework, Django templates |
| Хранение | PostgreSQL 15, Django ORM, JSONField для деталей запусков и raw-ответов моделей |
| Сбор данных | Python `requests`, `BeautifulSoup`, RSS/XML parser, scraper-модуль для ТАСС и Интерфакса |
| ML/NLP | LLM через Ollama или OpenAI-compatible `/v1/chat/completions`; fallback: RuBERT zero-shot через PyTorch/Transformers |
| Оркестрация | Management commands, `ThreadPoolExecutor` для async-задач, scheduler-контейнер Docker Compose |
| Визуализация | HTML dashboard, JavaScript, Chart.js/canvas |
| Эксплуатация | Docker, Docker Compose, GitHub Actions, pytest/pytest-django |

## 1. Общая архитектура и поток данных

```mermaid
flowchart LR
    subgraph UI["Пользовательский слой"]
        User["Аналитик / исследователь"]
        MonitoringUI["Dashboard: /monitoring/"]
        AnalysisUI["Лаборатория: /analysis/"]
    end

    subgraph API["Django + DRF API"]
        DataAPI["GET /api/monitoring/"]
        SummaryAPI["GET /api/monitoring/summary/"]
        RunAPI["POST /api/monitoring/run/"]
        HistoryAPI["POST /api/monitoring/history/run/"]
        TaskAPI["GET /api/task/{task_id}/"]
        PairAPI["POST /api/task/"]
    end

    subgraph Runtime["Процессы платформы"]
        TaskPool["Async tasks: ThreadPoolExecutor"]
        Scheduler["Scheduler: run_monitoring_scheduler"]
        Service["monitoring_service"]
        Catalog["sync_factor_catalog"]
        Aggregation["rebuild_sentiment_history"]
        Summaries["generate_llm_summaries"]
    end

    subgraph Sources["Источники новостей"]
        TASS["ТАСС RSS"]
        Interfax["Интерфакс: разделы и архив"]
    end

    subgraph Models["ML/NLP анализ"]
        Analyzer["get_monitoring_analyzer(engine)"]
        LLM["LLMClient: Ollama / OpenAI-compatible API"]
        BERT["RuBERT zero-shot fallback"]
    end

    subgraph DB["PostgreSQL"]
        Factors["RiskFactor"]
        News["NewsItem"]
        Classes["NewsClassification"]
        Daily["DailySentimentSummary"]
        FactorDaily["DailyFactorSentiment"]
        Batch["MonitoringBatch"]
        LlmSummary["MonitoringSummary"]
    end

    User --> MonitoringUI
    User --> AnalysisUI
    MonitoringUI --> DataAPI
    MonitoringUI --> SummaryAPI
    MonitoringUI --> RunAPI
    MonitoringUI --> HistoryAPI
    MonitoringUI --> TaskAPI
    AnalysisUI --> PairAPI
    AnalysisUI --> TaskAPI

    RunAPI --> TaskPool
    HistoryAPI --> TaskPool
    PairAPI --> TaskPool
    TaskPool --> Service
    Scheduler --> Service
    Catalog --> Factors

    Service --> Sources
    Sources --> TASS
    Sources --> Interfax
    Service --> News
    Service --> Analyzer
    Analyzer -->|engine = llm| LLM
    Analyzer -->|engine = bert| BERT
    Analyzer --> Classes
    Service --> Batch
    Classes --> Aggregation
    Aggregation --> Daily
    Aggregation --> FactorDaily
    Summaries --> LlmSummary
    Service -->|summarize=true| Summaries

    DataAPI --> Daily
    DataAPI --> FactorDaily
    DataAPI --> Classes
    DataAPI --> Batch
    SummaryAPI --> LlmSummary
```

## 2. ERD базы данных

```mermaid
erDiagram
    RISK_FACTOR {
        bigint id PK
        string key UK
        string name
        text description
        string positive_label
        string negative_label
        float weight
        int display_order
        bool is_active
        datetime created_at
        datetime updated_at
    }

    MONITORING_BATCH {
        bigint id PK
        string mode
        string status
        string engine
        string model_name
        string prompt_version
        string factor_catalog_version
        string source
        datetime started_at
        datetime finished_at
        int news_fetched
        int news_stored
        int classifications_created
        json details
    }

    NEWS_ITEM {
        bigint id PK
        string source
        string external_id
        string url
        string title
        text summary
        text text
        string category
        datetime published_at
        bool is_seed
        datetime created_at
        datetime updated_at
    }

    NEWS_CLASSIFICATION {
        bigint id PK
        bigint news_item_id FK
        bigint factor_id FK
        bigint batch_id FK
        float positive_probability
        float negative_probability
        float sentiment_score
        float relevance
        float pressure
        string sentiment_label
        float confidence
        string evidence
        string reason
        bool is_relevant
        string engine
        string model_name
        string prompt_version
        string factor_catalog_version
        string content_hash
        json raw_response
        datetime classified_at
    }

    DAILY_SENTIMENT_SUMMARY {
        bigint id PK
        date date
        string engine
        string model_name
        string prompt_version
        string factor_catalog_version
        float average_sentiment
        float delta_from_previous
        float risk_index
        int news_count
        int factor_count
        datetime updated_at
    }

    DAILY_FACTOR_SENTIMENT {
        bigint id PK
        bigint factor_id FK
        date date
        string engine
        string model_name
        string prompt_version
        string factor_catalog_version
        float average_sentiment
        float delta_from_previous
        int total_news_count
        int relevant_news_count
        int positive_hits
        int negative_hits
        int neutral_hits
        bigint top_positive_news_id FK
        bigint top_negative_news_id FK
        datetime updated_at
    }

    MONITORING_SUMMARY {
        bigint id PK
        bigint factor_id FK
        date period_start
        date period_end
        string engine
        string model_name
        string prompt_version
        string factor_catalog_version
        string input_hash
        string trend
        string risk_level
        float confidence
        text summary
        json main_drivers
        json raw_response
        datetime created_at
        datetime updated_at
    }

    RISK_FACTOR ||--o{ NEWS_CLASSIFICATION : classified_by_factor
    NEWS_ITEM ||--o{ NEWS_CLASSIFICATION : classified_news
    MONITORING_BATCH ||--o{ NEWS_CLASSIFICATION : produced
    RISK_FACTOR ||--o{ DAILY_FACTOR_SENTIMENT : daily_factor_metrics
    NEWS_ITEM ||--o{ DAILY_FACTOR_SENTIMENT : top_positive_news
    NEWS_ITEM ||--o{ DAILY_FACTOR_SENTIMENT : top_negative_news
    RISK_FACTOR ||--o{ MONITORING_SUMMARY : llm_period_summary
```

Ключевые ограничения:

- `NewsItem`: уникальность `(source, external_id)`.
- `NewsClassification`: уникальность `(news_item, factor, engine, model_name, prompt_version, factor_catalog_version)`.
- `DailySentimentSummary`: уникальность `(date, engine, model_name, prompt_version, factor_catalog_version)`.
- `DailyFactorSentiment`: уникальность `(factor, date, engine, model_name, prompt_version, factor_catalog_version)`.
- `MonitoringSummary`: уникальность `(factor, period_start, period_end, engine, model_name, prompt_version, factor_catalog_version)`.

## 3. Пользовательское взаимодействие, API и процессы

```mermaid
sequenceDiagram
    actor User as Аналитик
    participant UI as Web UI
    participant API as DRF API
    participant Tasks as Task registry + ThreadPoolExecutor
    participant Service as Monitoring service
    participant Sources as ТАСС / Интерфакс
    participant Model as LLM или RuBERT
    participant DB as PostgreSQL

    User->>UI: Открывает /monitoring/
    UI->>API: GET /api/monitoring/?engine=llm&days=3650
    API->>DB: Читает daily summaries, factor summaries, spikes, batches
    DB-->>API: Готовая аналитическая витрина
    API-->>UI: overview, timeline, factors, factor_detail
    UI->>API: GET /api/monitoring/summary/?engine=llm&period=day
    API->>DB: Читает MonitoringSummary
    API-->>UI: trend, risk_level, main_drivers

    alt Ручное live-обновление
        User->>UI: Нажимает "Обновить новости сейчас"
        UI->>API: POST /api/monitoring/run/
        API->>Tasks: create_monitoring_task
        API-->>UI: 202 Accepted: task_id
        Tasks->>Service: run_monitoring_pipeline
        Service->>Sources: collect_news_entries
        Sources-->>Service: Нормализованные новости
        Service->>DB: update_or_create NewsItem и MonitoringBatch
        Service->>Model: Классификация новостей по каталогу факторов
        Model-->>Service: sentiment, relevance, pressure, evidence
        Service->>DB: NewsClassification
        Service->>Service: rebuild_sentiment_history
        Service->>DB: DailySentimentSummary и DailyFactorSentiment
        opt summarize = true и engine = llm
            Service->>Model: generate_llm_summaries
            Service->>DB: MonitoringSummary
        end
        UI->>API: GET /api/task/{task_id}/
        API-->>UI: status, result, errors
    end

    alt Исторический backfill
        User->>UI: Нажимает "Заполнить историю за год"
        UI->>API: POST /api/monitoring/history/run/
        API->>Tasks: create_history_backfill_task
        Tasks->>Service: backfill_news_history
        Service->>Sources: collect_historical_news_entries по датам
        Service->>Model: Классификация архивных новостей
        Service->>DB: NewsItem, NewsClassification, агрегаты
    end

    alt Лаборатория парных гипотез
        User->>UI: Открывает /analysis/ и задает пары классов
        UI->>API: POST /api/task/
        API->>Tasks: create_task
        Tasks->>Sources: collect_news_entries
        Tasks->>Model: ZeroShotClassifier.predict по каждой паре
        Tasks-->>API: summary и news_results
        UI->>API: GET /api/task/{task_id}/
        API-->>UI: результат анализа
    end
```

### Карта API

| Endpoint | Метод | Назначение |
| --- | --- | --- |
| `/monitoring/` | `GET` | Основная HTML-панель мониторинга. |
| `/analysis/` | `GET` | Лаборатория ручных пар классификации. |
| `/api/monitoring/` | `GET` | Данные dashboard: overview, timeline, факторы, всплески, детализация фактора. |
| `/api/monitoring/summary/` | `GET` | LLM-сводки по фактору/периоду: trend, risk level, drivers. |
| `/api/monitoring/run/` | `POST` | Запуск live-обновления новостей и классификации. |
| `/api/monitoring/history/run/` | `POST` | Исторический backfill по датам, источникам и лимитам. |
| `/api/task/` | `POST` | Запуск ad-hoc zero-shot анализа по заданным парам классов. |
| `/api/task/{task_id}/` | `GET` | Проверка статуса async-задачи и получение результата. |
