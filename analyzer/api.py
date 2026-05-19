from rest_framework.response import Response
from rest_framework.views import APIView

from analyzer.constants import ENGINE_LLM, normalize_engine, normalize_factor_key, parse_bool
from analyzer.monitoring_service import DEFAULT_WINDOW_DAYS, backfill_news_history, get_monitoring_feed, run_monitoring_pipeline
from analyzer.tasks import create_task, get_task_status
from news.base_news.baseNews import BaseClassificationSerializer


class TaskCreateView(APIView):
    def post(self, request):
        serializer = BaseClassificationSerializer(data=request.data)
        if serializer.is_valid():
            task_id = create_task(serializer.validated_data)
            return Response({"task_id": task_id}, status=201)
        return Response(serializer.errors, status=400)


class TaskStatusView(APIView):
    def get(self, request, task_id):
        return Response(get_task_status(task_id))


class MonitoringDataView(APIView):
    def get(self, request):
        days = int(request.query_params.get("days", DEFAULT_WINDOW_DAYS))
        factor = normalize_factor_key(request.query_params.get("factor"))
        engine = normalize_engine(request.query_params.get("engine") or "llm")
        payload = get_monitoring_feed(days=days, factor_key=factor, engine=engine)
        return Response(payload, status=200)


class MonitoringRunView(APIView):
    def post(self, request):
        force = parse_bool(request.data.get("force"), False)
        engine = normalize_engine(request.data.get("engine") or "llm")
        summarize = parse_bool(request.data.get("summarize"), False)
        period = request.data.get("period") or "day"
        limit = request.data.get("limit")
        async_run = parse_bool(request.data.get("async"), engine == ENGINE_LLM)

        if async_run:
            from analyzer.tasks import create_monitoring_task

            task_id = create_monitoring_task(
                {
                    "engine": engine,
                    "force": force,
                    "summarize": summarize,
                    "period": period,
                    "limit": limit,
                }
            )
            return Response({"task_id": task_id, "status": "Pending", "engine": engine}, status=202)

        result = run_monitoring_pipeline(
            force=force,
            engine=engine,
            summarize=summarize,
            period=period,
            limit=limit,
        )
        status_code = 200 if result.get("skipped") else 201
        return Response(result, status=status_code)


class MonitoringHistoryRunView(APIView):
    def post(self, request):
        force = parse_bool(request.data.get("force"), False)
        engine = normalize_engine(request.data.get("engine") or "llm")
        summarize = parse_bool(request.data.get("summarize"), False)
        period = request.data.get("period") or "day"
        async_run = parse_bool(request.data.get("async"), True)
        payload = {
            "engine": engine,
            "force": force,
            "summarize": summarize,
            "period": period,
            "days": request.data.get("days") or 365,
            "start_date": request.data.get("start_date") or request.data.get("from_date"),
            "end_date": request.data.get("end_date") or request.data.get("to_date"),
            "sources": request.data.get("sources"),
            "limit_per_day": request.data.get("limit_per_day") or 8,
            "limit": request.data.get("limit"),
            "delay_seconds": request.data.get("delay_seconds") or 0.0,
        }

        if async_run:
            from analyzer.tasks import create_history_backfill_task

            task_id = create_history_backfill_task(payload)
            return Response({"task_id": task_id, "status": "Pending", "engine": engine}, status=202)

        result = backfill_news_history(
            start_date=payload.get("start_date"),
            end_date=payload.get("end_date"),
            days=payload.get("days"),
            source_names=payload.get("sources"),
            limit_per_day=payload.get("limit_per_day"),
            total_limit=payload.get("limit"),
            delay_seconds=payload.get("delay_seconds"),
            engine=payload.get("engine"),
            force=payload.get("force"),
            summarize=payload.get("summarize"),
            period=payload.get("period"),
        )
        status_code = 200 if result.get("skipped") else 201
        return Response(result, status=status_code)


class MonitoringSummaryView(APIView):
    def get(self, request):
        from analyzer.llm_services import get_summary_feed

        engine = normalize_engine(request.query_params.get("engine") or "llm")
        factor = normalize_factor_key(request.query_params.get("factor"))
        payload = get_summary_feed(
            engine=engine,
            period=request.query_params.get("period") or "day",
            from_date=request.query_params.get("from"),
            to_date=request.query_params.get("to"),
            factor_key=factor,
        )
        return Response(payload, status=200)
