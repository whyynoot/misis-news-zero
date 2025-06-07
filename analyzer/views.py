from django.shortcuts import render

# Create your views here.
def about_page(request):
    return render(request, 'about.html', {})

def analysis_page(request):
    return render (request, 'analysis.html', {})

def monitoring_page(request):
    return render (request, 'monitoring.html', {})