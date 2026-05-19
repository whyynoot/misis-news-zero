from django.shortcuts import render

from analyzer.factors import FACTOR_CONFIG


def analysis_page(request):
    default_pairs = [{"class1": factor["positive_label"], "class2": factor["negative_label"]} for factor in FACTOR_CONFIG]
    return render(request, "analysis.html", {"default_pairs": default_pairs})


def monitoring_page(request):
    return render(request, "monitoring.html", {})
