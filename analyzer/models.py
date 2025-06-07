from django.db import models
from rest_framework.views import APIView
from rest_framework.response import Response
import news.news as news
from analyzer.tasks import get_task_status, create_task
from news.base_news.baseNews import BaseClassificationSerializer

scrapers = news.get_scrappers()

class TaskCreateView(APIView):
    def post(self, request):
        serializer = BaseClassificationSerializer(data=request.data)
        if serializer.is_valid():
            task_id = create_task(serializer.validated_data)
            return Response({'task_id': task_id}, status=201)
        else:
            return Response(serializer.errors, status=400)

class TaskStatusView(APIView):
    def get(self, request, task_id):
        status = get_task_status(task_id)
        return Response(status)