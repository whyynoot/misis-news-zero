import io
import json
import uuid
import datetime
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import numpy as np
import requests
from django.core.management import call_command
from django.test import Client, TestCase
from django.urls import reverse
from django.utils import timezone
from rest_framework.test import APITestCase

from analyzer.constants import ENGINE_LLM
from analyzer.llm_client import LLMClient, LLMJSONError, LLMRequestError, LLMSettings, parse_json_object
from analyzer.llm_services import (
    classify_news_items_with_llm,
    generate_llm_summaries,
    llm_classification_identity,
)
from analyzer.monitoring_service import (
    FACTOR_CONFIG,
    backfill_news_history,
    backfill_existing_news,
    get_monitoring_feed,
    load_seed_dataset,
    run_monitoring_pipeline,
)
from analyzer.api import TaskCreateView, TaskStatusView
from analyzer.models import (
    DailyFactorSentiment,
    DailySentimentSummary,
    MonitoringBatch,
    MonitoringSummary,
    NewsClassification,
    NewsItem,
    RiskFactor,
)
from analyzer.tasks import create_history_backfill_task, create_task, finalize_task, get_task_status, process_task, tasks
from analyzer.zero import ZeroShotClassifier
from news.sources import collect_historical_news_entries, get_scrapers
from news.scraper.interfax import InterfaxScraper
from news.scraper.tass import TassScraper


def llm_test_settings():
    return LLMSettings(
        enabled=True,
        base_url="http://ollama.test",
        api_key="",
        model="gemma4:e2b",
        timeout_seconds=1,
        temperature=0,
        max_tokens=256,
        batch_news_size=4,
        concurrency=1,
        think=False,
        json_mode=True,
        classification_prompt_version="social-risk-classification-v1",
        summary_prompt_version="social-risk-summary-v1",
    )


class ZeroShotClassifierTest(TestCase):
    @patch("analyzer.zero.AutoTokenizer.from_pretrained")
    @patch("analyzer.zero.AutoModelForSequenceClassification.from_pretrained")
    def test_classifier_initialization(self, mock_model, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()

        classifier = ZeroShotClassifier()

        self.assertIsNotNone(classifier.tokenizer)
        self.assertIsNotNone(classifier.model)
        self.assertEqual(classifier.target_label, "entailment")

    @patch("analyzer.zero.torch.softmax")
    @patch("analyzer.zero.torch.inference_mode")
    @patch("analyzer.zero.AutoTokenizer.from_pretrained")
    @patch("analyzer.zero.AutoModelForSequenceClassification.from_pretrained")
    def test_predict_method(self, mock_model_cls, mock_tokenizer_cls, mock_inference, mock_softmax):
        mock_inference.return_value.__enter__ = MagicMock()
        mock_inference.return_value.__exit__ = MagicMock()

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer_cls.return_value = mock_tokenizer
        mock_model_cls.return_value = mock_model

        classifier = ZeroShotClassifier()

        mock_model.device = "cpu"
        mock_model.config.label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}

        mock_tokens = MagicMock()
        mock_tokens.to.return_value = mock_tokens
        mock_tokenizer.return_value = mock_tokens

        mock_logits = MagicMock()
        mock_model_output = MagicMock()
        mock_model_output.logits = mock_logits
        mock_model.return_value = mock_model_output

        mock_proba_tensor = MagicMock()
        mock_proba_tensor.reshape.return_value.cpu.return_value.numpy.return_value = np.array([[0.7, 0.3]])
        mock_softmax.return_value = MagicMock()
        mock_softmax.return_value.__getitem__.return_value = mock_proba_tensor

        result = classifier.predict("This is a test news article", ["positive", "negative"])

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 2)


class LLMClientTest(TestCase):
    def test_parse_json_object_valid(self):
        self.assertEqual(parse_json_object('{"items": []}'), {"items": []})

    def test_parse_json_object_invalid(self):
        with self.assertRaises(LLMJSONError):
            parse_json_object("not json")

    @patch("analyzer.llm_client.requests.post")
    def test_complete_json_retries_invalid_json_once(self, mock_post):
        first_response = MagicMock()
        first_response.raise_for_status.return_value = None
        first_response.json.return_value = {"message": {"content": "not json"}}
        second_response = MagicMock()
        second_response.raise_for_status.return_value = None
        second_response.json.return_value = {"message": {"content": '{"ok": true}'}}
        mock_post.side_effect = [first_response, second_response]

        client = LLMClient(llm_test_settings())
        payload, raw = client.complete_json("system", "user")

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(raw, '{"ok": true}')
        self.assertEqual(mock_post.call_count, 2)

    @patch("analyzer.llm_client.requests.post", side_effect=requests.Timeout())
    def test_complete_json_timeout(self, mock_post):
        client = LLMClient(llm_test_settings())

        with self.assertRaises(LLMRequestError):
            client.complete_json("system", "user")


class LLMClassificationServiceTest(TestCase):
    def setUp(self):
        NewsClassification.objects.all().delete()
        DailyFactorSentiment.objects.all().delete()
        DailySentimentSummary.objects.all().delete()
        NewsItem.objects.all().delete()
        MonitoringBatch.objects.all().delete()
        RiskFactor.objects.all().delete()
        self.factors = load_seed_dataset(clear_existing=True) and list(RiskFactor.objects.order_by("display_order"))
        NewsClassification.objects.all().delete()
        DailyFactorSentiment.objects.all().delete()
        DailySentimentSummary.objects.all().delete()
        self.news_item = NewsItem.objects.create(
            source="tass",
            external_id="llm-news-1",
            title="В регионе открылся новый медицинский центр",
            summary="Пациентам обещают более доступную диагностику.",
            text="В регионе открылся новый медицинский центр. Пациентам обещают более доступную диагностику.",
            published_at=timezone.make_aware(datetime.datetime(2026, 4, 8, 10, 0, 0)),
        )

    def test_llm_classification_creates_all_factor_results_and_uses_cache(self):
        factors_payload = []
        for factor in self.factors:
            if factor.key == "outpatient_clinics":
                factors_payload.append(
                    {
                        "factor_id": factor.key,
                        "relevance": 1.4,
                        "sentiment": 0.8,
                        "pressure": -0.2,
                        "confidence": 1.2,
                        "label": "positive",
                        "evidence": "открылся новый медицинский центр",
                        "reason": "улучшает доступность помощи",
                    }
                )
            else:
                factors_payload.append(
                    {
                        "factor_id": factor.key,
                        "relevance": 0,
                        "sentiment": 0,
                        "pressure": 0,
                        "confidence": 0.7,
                        "label": "neutral",
                        "evidence": "",
                        "reason": "нет связи",
                    }
                )

        fake_client = MagicMock()
        fake_client.complete_json.return_value = (
            {"items": [{"news_id": str(self.news_item.id), "factors": factors_payload}]},
            "{}",
        )

        result = classify_news_items_with_llm([self.news_item], self.factors, client=fake_client)

        self.assertEqual(result["classified_count"], len(self.factors))
        self.assertEqual(NewsClassification.objects.filter(engine=ENGINE_LLM).count(), len(self.factors))
        healthcare = NewsClassification.objects.get(engine=ENGINE_LLM, factor__key="outpatient_clinics")
        self.assertEqual(healthcare.relevance, 1.0)
        self.assertEqual(healthcare.pressure, 0.0)
        self.assertEqual(healthcare.confidence, 1.0)
        child_benefits = NewsClassification.objects.get(engine=ENGINE_LLM, factor__key="children_benefits")
        self.assertEqual(child_benefits.sentiment_label, "neutral")
        self.assertFalse(child_benefits.is_relevant)

        cached_result = classify_news_items_with_llm([self.news_item], self.factors, client=fake_client)
        self.assertEqual(cached_result["classified_count"], 0)
        self.assertEqual(fake_client.complete_json.call_count, 1)


class LLMSummaryServiceTest(TestCase):
    def setUp(self):
        NewsClassification.objects.all().delete()
        DailyFactorSentiment.objects.all().delete()
        DailySentimentSummary.objects.all().delete()
        MonitoringSummary.objects.all().delete()
        NewsItem.objects.all().delete()
        MonitoringBatch.objects.all().delete()
        RiskFactor.objects.all().delete()
        load_seed_dataset(clear_existing=True)
        NewsClassification.objects.all().delete()
        DailyFactorSentiment.objects.all().delete()
        DailySentimentSummary.objects.all().delete()
        self.factor = RiskFactor.objects.get(key="outpatient_clinics")
        identity = llm_classification_identity(llm_test_settings())
        self.news_items = []
        for index, pressure in enumerate([0.2, 0.9, 0.5]):
            news_item = NewsItem.objects.create(
                source="tass",
                external_id=f"summary-news-{index}",
                title=f"Новость медицины {index}",
                summary="Короткое описание",
                text="Короткий текст про медицину",
                published_at=timezone.make_aware(datetime.datetime(2026, 4, 8, 10, 0, 0)),
            )
            self.news_items.append(news_item)
            NewsClassification.objects.create(
                news_item=news_item,
                factor=self.factor,
                positive_probability=0.2,
                negative_probability=0.8,
                sentiment_score=-0.6,
                relevance=0.8,
                pressure=pressure,
                sentiment_label="negative",
                confidence=0.8,
                is_relevant=True,
                content_hash="test",
                **identity,
            )

    def test_summary_generation_uses_top_news_and_cache(self):
        fake_client = MagicMock()
        fake_client.complete_json.return_value = (
            {
                "period_start": "2026-04-08",
                "period_end": "2026-04-08",
                "engine": "llm",
                "factors": [
                    {
                        "factor_id": "outpatient_clinics",
                        "trend": "worsening",
                        "risk_level": "medium",
                        "confidence": 0.8,
                        "summary": "Негативный сигнал связан с доступностью медицины.",
                        "main_drivers": [
                            {
                                "title": "Новость медицины 1",
                                "date": "2026-04-08",
                                "impact": -0.6,
                                "why": "высокое давление",
                            }
                        ],
                    }
                ],
            },
            "{}",
        )

        result = generate_llm_summaries(
            from_date="2026-04-08",
            to_date="2026-04-08",
            factor_key="medicine",
            top_n=2,
            client=fake_client,
        )

        self.assertEqual(result["created_count"], 1)
        summary = MonitoringSummary.objects.get(factor=self.factor)
        self.assertEqual(summary.trend, "worsening")
        self.assertEqual(summary.risk_level, "medium")
        self.assertEqual(summary.main_drivers[0]["title"], "Новость медицины 1")
        self.assertEqual(summary.raw_response["input"]["top_news"][0]["title"], "Новость медицины 1")

        cached = generate_llm_summaries(
            from_date="2026-04-08",
            to_date="2026-04-08",
            factor_key="medicine",
            top_n=2,
            client=fake_client,
        )
        self.assertEqual(cached["created_count"], 0)
        self.assertEqual(cached["cached_count"], 1)
        self.assertEqual(fake_client.complete_json.call_count, 1)


class TasksTest(TestCase):
    def setUp(self):
        tasks.clear()

    @patch("analyzer.tasks.process_task_async")
    def test_create_task(self, mock_async):
        test_data = {"pairs": [{"class1": "positive", "class2": "negative"}]}

        task_id = create_task(test_data)

        self.assertIsInstance(task_id, str)
        self.assertIn(task_id, tasks)
        self.assertEqual(tasks[task_id]["status"], "Pending")
        self.assertEqual(tasks[task_id]["input_data"], test_data)
        mock_async.assert_called_once_with(task_id)

    @patch("analyzer.tasks.process_history_backfill_task_async")
    def test_create_history_backfill_task(self, mock_async):
        task_id = create_history_backfill_task({"days": 365, "sources": ["interfax"]})

        self.assertIn(task_id, tasks)
        self.assertEqual(tasks[task_id]["kind"], "history_backfill")
        mock_async.assert_called_once_with(task_id)

    @patch("analyzer.tasks.process_task_async")
    def test_get_task_status_existing(self, mock_async):
        task_id = create_task({"pairs": []})
        status = get_task_status(task_id)

        self.assertEqual(status["status"], "Pending")
        self.assertIsNone(status["result"])
        self.assertIsNone(status["error"])

    def test_get_task_status_nonexistent(self):
        status = get_task_status(str(uuid.uuid4()))
        self.assertEqual(status["status"], "Task not found")

    def test_finalize_task_marks_complete_when_no_error(self):
        task_id = str(uuid.uuid4())
        tasks[task_id] = {"status": "Running", "input_data": {"pairs": []}, "result": {"ok": True}, "error": None}
        future = Future()
        future.set_result(None)

        finalize_task(task_id, future)

        self.assertEqual(tasks[task_id]["status"], "Complete")

    def test_finalize_task_marks_failed_on_exception(self):
        task_id = str(uuid.uuid4())
        tasks[task_id] = {"status": "Running", "input_data": {"pairs": []}, "result": None, "error": None}
        future = Future()
        future.set_exception(RuntimeError("boom"))

        finalize_task(task_id, future)

        self.assertEqual(tasks[task_id]["status"], "Failed")
        self.assertEqual(tasks[task_id]["error"], "boom")

    @patch("analyzer.tasks.collect_news_entries")
    @patch("analyzer.tasks.get_classifier")
    def test_process_task_success(self, mock_get_classifier, mock_collect_news_entries):
        mock_collect_news_entries.return_value = [{"text": "Test news article 1"}, {"text": "Test news article 2"}]
        fake_classifier = MagicMock()
        fake_classifier.predict.return_value = [0.6, 0.4]
        mock_get_classifier.return_value = fake_classifier

        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            "status": "Pending",
            "input_data": {"pairs": [{"class1": "positive", "class2": "negative"}]},
            "result": None,
            "error": None,
        }

        process_task(task_id, tasks[task_id]["input_data"])

        self.assertIsNotNone(tasks[task_id]["result"])
        self.assertIn("news_results", tasks[task_id]["result"])
        self.assertIn("summary", tasks[task_id]["result"])

    @patch("analyzer.tasks.collect_news_entries")
    def test_process_task_empty_news(self, mock_collect_news_entries):
        mock_collect_news_entries.return_value = []

        task_id = str(uuid.uuid4())
        tasks[task_id] = {"status": "Pending", "input_data": {"pairs": []}, "result": None, "error": None}

        process_task(task_id, tasks[task_id]["input_data"])

        self.assertEqual(tasks[task_id]["error"], "Parsing error")


class APIViewsTest(APITestCase):
    def setUp(self):
        self.client = Client()
        tasks.clear()

    @patch("analyzer.api.create_task")
    def test_task_create_view_valid_data(self, mock_create_task):
        mock_create_task.return_value = "test-task-id"
        url = reverse("create_task")
        data = {"pairs": [{"class1": "positive", "class2": "negative"}]}

        with patch("analyzer.api.BaseClassificationSerializer") as mock_serializer:
            serializer_instance = MagicMock()
            serializer_instance.is_valid.return_value = True
            serializer_instance.validated_data = data
            mock_serializer.return_value = serializer_instance

            response = self.client.post(url, json.dumps(data), content_type="application/json")

        self.assertEqual(response.status_code, 201)

    @patch("analyzer.tasks.process_task_async")
    def test_task_status_view(self, mock_async):
        task_id = create_task({"pairs": []})
        response = self.client.get(reverse("task_status", kwargs={"task_id": task_id}))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.content)["status"], "Pending")


class MonitoringServiceTest(TestCase):
    def setUp(self):
        NewsClassification.objects.all().delete()
        DailyFactorSentiment.objects.all().delete()
        DailySentimentSummary.objects.all().delete()
        NewsItem.objects.all().delete()
        MonitoringBatch.objects.all().delete()
        RiskFactor.objects.all().delete()

    def test_load_seed_dataset_creates_history(self):
        result = load_seed_dataset(clear_existing=True)

        self.assertFalse(result["skipped"])
        self.assertEqual(RiskFactor.objects.count(), len(FACTOR_CONFIG))
        self.assertEqual(len(FACTOR_CONFIG), 36)
        self.assertEqual(FACTOR_CONFIG[0]["key"], "population_size")
        self.assertEqual(FACTOR_CONFIG[-1]["key"], "birth_rate_per_1000")
        self.assertEqual(FACTOR_CONFIG[-1]["weight"], 1.40)
        self.assertGreater(NewsItem.objects.count(), 0)
        self.assertGreater(NewsClassification.objects.count(), 0)
        self.assertGreater(DailySentimentSummary.objects.count(), 0)
        self.assertGreater(DailyFactorSentiment.objects.count(), 0)

    def test_get_monitoring_feed_returns_factor_detail_and_spikes(self):
        load_seed_dataset(clear_existing=True)

        feed = get_monitoring_feed(days=30)

        self.assertIn("overview", feed)
        self.assertIn("factors", feed)
        self.assertIn("spikes", feed)
        self.assertIn("factor_detail", feed)
        self.assertTrue(feed["factor_detail"]["recent_news"])

    @patch("analyzer.monitoring_service.get_classifier")
    def test_run_monitoring_pipeline_ingests_live_news(self, mock_get_classifier):
        fake_classifier = MagicMock()

        def predict_batch(texts, labels):
            outputs = []
            for text in texts:
                text = text.lower()
                if "сокращ" in text or "дефицит" in text:
                    outputs.append([0.18, 0.82])
                else:
                    outputs.append([0.76, 0.24])
            return np.array(outputs)

        fake_classifier.predict_batch.side_effect = predict_batch
        mock_get_classifier.return_value = fake_classifier

        live_news = [
            {
                "source": "tass",
                "external_id": "live-1",
                "title": "Регион увеличил выплаты молодым семьям",
                "summary": "Новые меры поддержки должны стимулировать рождаемость.",
                "text": "Регион увеличил выплаты молодым семьям. Новые меры поддержки должны стимулировать рождаемость.",
                "published_at": "2026-04-08T10:00:00+03:00",
                "url": "https://example.com/live/1",
                "category": "Общество",
            },
            {
                "source": "tass",
                "external_id": "live-2",
                "title": "В районе возник дефицит врачей и отменены маршруты скорой помощи",
                "summary": "Пациенты жалуются на рост времени ожидания помощи.",
                "text": "В районе возник дефицит врачей и отменены маршруты скорой помощи.",
                "published_at": "2026-04-08T12:30:00+03:00",
                "url": "https://example.com/live/2",
                "category": "Общество",
            },
        ]

        result = run_monitoring_pipeline(news_entries=live_news, bootstrap_if_empty=False)

        self.assertFalse(result["skipped"])
        self.assertEqual(result["news_stored"], 2)
        self.assertEqual(NewsItem.objects.count(), 2)
        self.assertEqual(RiskFactor.objects.count(), len(FACTOR_CONFIG))
        self.assertGreater(DailySentimentSummary.objects.count(), 0)

    @patch("analyzer.monitoring_service.get_historical_news_entries")
    @patch("analyzer.monitoring_service.get_classifier")
    def test_backfill_news_history_fetches_stores_and_classifies_archive(self, mock_get_classifier, mock_history_entries):
        fake_classifier = MagicMock()
        fake_classifier.predict_batch.return_value = np.array([[0.78, 0.22]])
        mock_get_classifier.return_value = fake_classifier
        mock_history_entries.return_value = [
            {
                "source": "interfax",
                "external_id": "https://www.interfax.ru/russia/1",
                "title": "В регионе открыли новую поликлинику",
                "summary": "Медицинская помощь стала доступнее.",
                "text": "В регионе открыли новую поликлинику. Медицинская помощь стала доступнее.",
                "published_at": timezone.make_aware(datetime.datetime(2026, 4, 8, 9, 0, 0)),
                "url": "https://www.interfax.ru/russia/1",
                "category": "Россия",
            }
        ]

        result = backfill_news_history(
            start_date="2026-04-08",
            end_date="2026-04-08",
            source_names=["interfax"],
            limit_per_day=5,
            delay_seconds=0,
        )

        self.assertFalse(result["skipped"])
        self.assertEqual(result["news_fetched"], 1)
        self.assertEqual(result["news_stored"], 1)
        self.assertEqual(NewsItem.objects.filter(source="interfax").count(), 1)
        self.assertEqual(NewsClassification.objects.count(), len(FACTOR_CONFIG))
        self.assertEqual(MonitoringBatch.objects.latest("started_at").source, "history:interfax")

    def test_backfill_command_loads_seed(self):
        stdout = io.StringIO()

        call_command("backfill_monitoring", "--seed", "--clear", stdout=stdout)

        self.assertIn("News:", stdout.getvalue())
        self.assertGreater(NewsItem.objects.count(), 0)

    @patch("analyzer.monitoring_service.get_classifier")
    def test_backfill_existing_news_excludes_seed_by_default(self, mock_get_classifier):
        fake_classifier = MagicMock()
        fake_classifier.predict_batch.return_value = np.array([[0.72, 0.28]])
        mock_get_classifier.return_value = fake_classifier
        load_seed_dataset(clear_existing=True)
        real_news = NewsItem.objects.create(
            source="tass",
            external_id="real-backfill-1",
            title="Real TASS news",
            summary="Real summary",
            text="Real monitoring text",
            published_at=timezone.make_aware(datetime.datetime(2026, 4, 9, 10, 0, 0)),
            is_seed=False,
        )
        NewsClassification.objects.all().delete()
        DailyFactorSentiment.objects.all().delete()
        DailySentimentSummary.objects.all().delete()

        result = backfill_existing_news(engine="bert", limit=5000, source=None)

        self.assertEqual(result["news_count"], 1)
        self.assertIsNone(result["source"])
        self.assertEqual(NewsClassification.objects.filter(news_item=real_news).count(), len(FACTOR_CONFIG))
        self.assertEqual(NewsClassification.objects.filter(news_item__is_seed=True).count(), 0)

    @patch("analyzer.monitoring_service.get_classifier")
    def test_run_monitoring_command_updates_feed(self, mock_get_classifier):
        fake_classifier = MagicMock()
        fake_classifier.predict_batch.return_value = np.array([[0.72, 0.28]])
        mock_get_classifier.return_value = fake_classifier

        with patch("analyzer.monitoring_service.get_news_entries") as mock_entries:
            mock_entries.return_value = [
                {
                    "source": "tass",
                    "external_id": "command-live-1",
                    "title": "В городе открыли новый диагностический центр",
                    "summary": "Это должно повысить доступность медицины.",
                    "text": "В городе открыли новый диагностический центр.",
                    "published_at": "2026-04-08T09:00:00+03:00",
                    "url": "https://example.com/command/live-1",
                    "category": "Общество",
                }
            ]
            stdout = io.StringIO()
            call_command("run_monitoring", "--no-bootstrap", "--engine", "bert", stdout=stdout)

        self.assertIn("Live update completed", stdout.getvalue())
        self.assertEqual(NewsItem.objects.count(), 1)


class MonitoringApiTest(APITestCase):
    def setUp(self):
        NewsClassification.objects.all().delete()
        DailyFactorSentiment.objects.all().delete()
        DailySentimentSummary.objects.all().delete()
        NewsItem.objects.all().delete()
        MonitoringBatch.objects.all().delete()
        RiskFactor.objects.all().delete()
        load_seed_dataset(clear_existing=True)

    def test_monitoring_data_view(self):
        response = self.client.get(reverse("monitoring_data"))
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertIn("overview", payload)
        self.assertIn("factor_detail", payload)
        self.assertTrue(payload["factors"])

    def test_monitoring_data_view_llm_engine(self):
        response = self.client.get(reverse("monitoring_data") + "?engine=llm")
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["engine"], "llm")
        self.assertIn("factors", payload)

    @patch("analyzer.tasks.create_monitoring_task")
    def test_monitoring_run_llm_creates_task(self, mock_create_task):
        mock_create_task.return_value = "monitoring-task-id"

        response = self.client.post(
            reverse("monitoring_run"),
            json.dumps({"engine": "llm", "force": False, "summarize": True}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["task_id"], "monitoring-task-id")
        mock_create_task.assert_called_once()

    @patch("analyzer.tasks.create_history_backfill_task")
    def test_monitoring_history_run_creates_task(self, mock_create_task):
        mock_create_task.return_value = "history-task-id"

        response = self.client.post(
            reverse("monitoring_history_run"),
            json.dumps({"engine": "bert", "days": 365, "sources": ["interfax"]}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["task_id"], "history-task-id")
        mock_create_task.assert_called_once()

    def test_monitoring_summary_view(self):
        response = self.client.get(reverse("monitoring_summary") + "?engine=llm&period=day")
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertIn("factors", payload)
        self.assertEqual(payload["summary_engine"], "llm")


class MonitoringCommandTest(TestCase):
    @patch("analyzer.management.commands.backfill_news_history.backfill_news_history")
    def test_backfill_news_history_command_invokes_pipeline(self, mock_backfill):
        mock_backfill.return_value = {
            "skipped": False,
            "start_date": "2026-04-08",
            "end_date": "2026-04-08",
            "sources": ["interfax"],
            "engine": "bert",
            "news_fetched": 1,
            "news_stored": 1,
            "cached_count": 0,
            "classifications_created": 36,
            "dates": ["2026-04-08"],
            "errors": [],
            "warnings": [],
        }
        stdout = io.StringIO()

        call_command(
            "backfill_news_history",
            "--from-date",
            "2026-04-08",
            "--to-date",
            "2026-04-08",
            "--sources",
            "interfax",
            "--limit-per-day",
            "1",
            stdout=stdout,
        )

        self.assertIn("Historical backfill completed", stdout.getvalue())
        mock_backfill.assert_called_once()

    def test_sync_factor_catalog_command(self):
        stdout = io.StringIO()

        call_command("sync_factor_catalog", stdout=stdout)

        self.assertIn("Active factors: 36", stdout.getvalue())

    @patch("analyzer.management.commands.run_monitoring_scheduler.run_monitoring_pipeline")
    @patch("analyzer.management.commands.run_monitoring_scheduler.ensure_factor_catalog")
    def test_run_monitoring_scheduler_once(self, mock_ensure, mock_run):
        mock_run.return_value = {
            "status": "success",
            "news_fetched": 1,
            "news_stored": 1,
            "classifications_created": 36,
        }
        stdout = io.StringIO()

        call_command("run_monitoring_scheduler", "--once", "--interval-hours", "0.01", stdout=stdout)

        self.assertIn("Scheduled live update", stdout.getvalue())
        mock_ensure.assert_called_once()
        mock_run.assert_called_once()


class ModelTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_task_create_view_post_invalid_data(self):
        view = TaskCreateView()
        self.assertTrue(hasattr(view, "post"))
        self.assertEqual(view.__class__.__name__, "TaskCreateView")

    def test_task_status_view_get(self):
        view = TaskStatusView()
        self.assertTrue(hasattr(view, "get"))
        self.assertEqual(view.__class__.__name__, "TaskStatusView")


class TassScraperTest(TestCase):
    SAMPLE_RSS = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Рост доходов семей</title>
      <description>Доходы семей увеличиваются быстрее инфляции.</description>
      <category>Экономика и бизнес</category>
      <link>https://example.com/news/1</link>
      <pubDate>Tue, 08 Apr 2026 09:00:00 +0300</pubDate>
    </item>
    <item>
      <title>Спортивный матч</title>
      <description>Нейтральная новость без релевантной тематики.</description>
      <category>Спорт</category>
      <link>https://example.com/news/2</link>
      <pubDate>Tue, 08 Apr 2026 10:00:00 +0300</pubDate>
    </item>
  </channel>
</rss>
"""

    @patch("news.scraper.tass.requests.get")
    def test_get_news_entries_uses_public_rss_feed(self, mock_get):
        mock_response = MagicMock()
        mock_response.content = self.SAMPLE_RSS.encode("utf-8")
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        scraper = TassScraper()
        scraper.parse_pages = 1

        entries = scraper.get_news_entries()

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["title"], "Рост доходов семей")
        self.assertEqual(entries[0]["url"], "https://example.com/news/1")
        self.assertIsNotNone(entries[0]["published_at"])
        mock_get.assert_called_once_with(scraper.feed_url, headers=scraper.headers, timeout=10)


class InterfaxScraperTest(TestCase):
    SECTION_HTML = """
<html>
  <body>
    <div class="timeline">
      <div class="timeline__text">
        <time datetime="2026-04-08T09:00:00+03:00"></time>
        <a href="/russia/123" title="В регионе открыли новую поликлинику">В регионе открыли новую поликлинику</a>
      </div>
    </div>
  </body>
</html>
"""
    ARTICLE_HTML = """
<html>
  <head>
    <link rel="canonical" href="https://www.interfax.ru/russia/123">
    <meta property="article:section" content="Россия">
    <meta property="article:published_time" content="2026-04-08T09:00:00+03:00">
    <meta name="description" content="Новая поликлиника повысит доступность помощи.">
  </head>
  <body>
    <article itemprop="articleBody">
      <h1 itemprop="headline">В регионе открыли новую поликлинику</h1>
      <p>В регионе открыли новую поликлинику для детей и взрослых.</p>
      <p>Власти сообщили, что прием врачей станет доступнее.</p>
    </article>
  </body>
</html>
"""
    ARCHIVE_HTML = """
<html>
  <body>
    <div class="an">
      <div data-id="123">
        <span>09:00</span>
        <a href="/russia/123"><h3>В регионе открыли новую поликлинику</h3></a>
      </div>
      <div data-id="124">
        <span>09:15</span>
        <a href="/sport/124"><h3>Команда выиграла матч</h3></a>
      </div>
    </div>
  </body>
</html>
"""

    def test_get_news_entries_hydrates_interfax_articles(self):
        scraper = InterfaxScraper(sections=("russia",), hydrate_articles=True)

        def fake_fetch(url, referer=None):
            if url.endswith("/russia/123"):
                return self.ARTICLE_HTML
            return self.SECTION_HTML

        with patch.object(scraper, "_fetch", side_effect=fake_fetch):
            entries = scraper.get_news_entries(limit=1)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["source"], "interfax")
        self.assertEqual(entries[0]["url"], "https://www.interfax.ru/russia/123")
        self.assertIn("поликлинику", entries[0]["text"])
        self.assertIsNotNone(entries[0]["published_at"])

    def test_article_href_accepts_internal_absolute_urls(self):
        scraper = InterfaxScraper()

        self.assertTrue(scraper._is_article_href("https://www.interfax.ru/russia/123"))
        self.assertFalse(scraper._is_article_href("https://example.com/russia/123"))

    def test_get_news_entries_for_date_reads_interfax_archive(self):
        scraper = InterfaxScraper(hydrate_articles=True)

        def fake_fetch(url, referer=None):
            if url.endswith("/russia/123"):
                return self.ARTICLE_HTML
            return self.ARCHIVE_HTML

        with patch.object(scraper, "_fetch", side_effect=fake_fetch):
            entries = scraper.get_news_entries_for_date(datetime.date(2026, 4, 8), limit=5)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["url"], "https://www.interfax.ru/russia/123")
        self.assertEqual(entries[0]["published_at"].date(), datetime.date(2026, 4, 8))
        self.assertIn("поликлинику", entries[0]["text"])


class SourceRegistryTest(TestCase):
    def test_get_scrapers_filters_unknown_sources(self):
        scrapers = get_scrapers(["tass", "unknown", "interfax"])

        self.assertEqual(list(scrapers), ["tass", "interfax"])

    def test_collect_historical_news_entries_uses_date_capable_scraper(self):
        fake_scraper = MagicMock()
        fake_scraper.get_news_entries_for_date.return_value = [
            {
                "title": "Историческая новость",
                "text": "Историческая новость про доходы населения.",
                "published_at": "2026-04-08T10:00:00+03:00",
            }
        ]

        with patch("news.sources.get_scrapers", return_value={"fake": fake_scraper}):
            entries = collect_historical_news_entries(
                datetime.date(2026, 4, 8),
                datetime.date(2026, 4, 8),
                limit_per_day=3,
                source_names=["fake"],
                delay_seconds=0,
            )

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["source"], "fake")
        fake_scraper.get_news_entries_for_date.assert_called_once()
