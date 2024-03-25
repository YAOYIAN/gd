from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('download/', views.download, name='download'),
    path('your-url/', views.your_view, name='your-view'),
    path('your-ajax-url/', views.your_ajax_view, name='your-ajax-view'),
    path('your_api_url/', views.your_api_url, name='your_api_url'),
]


