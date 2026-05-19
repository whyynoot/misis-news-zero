"""news_analyzer URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from analyzer.api import (
    MonitoringDataView,
    MonitoringHistoryRunView,
    MonitoringRunView,
    MonitoringSummaryView,
    TaskCreateView,
    TaskStatusView,
)
from analyzer.views import analysis_page, monitoring_page


urlpatterns = [
    path("", monitoring_page),
    path("analysis/", analysis_page, name="analysis"),
    path("monitoring/", monitoring_page, name="monitoring"),

    path("api/task/", TaskCreateView.as_view(), name="create_task"),
    path("api/task/<str:task_id>/", TaskStatusView.as_view(), name="task_status"),

    path("api/monitoring/", MonitoringDataView.as_view(), name="monitoring_data"),
    path("api/monitoring/run/", MonitoringRunView.as_view(), name="monitoring_run"),
    path("api/monitoring/history/run/", MonitoringHistoryRunView.as_view(), name="monitoring_history_run"),
    path("api/monitoring/summary/", MonitoringSummaryView.as_view(), name="monitoring_summary"),
]
