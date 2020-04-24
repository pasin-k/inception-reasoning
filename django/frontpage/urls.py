from django.urls import path
from django.contrib import admin
from .views import HomePageView, upload_file, get_explanation

urlpatterns = [
    path('', HomePageView.as_view(), name='home'),
    path('upload-form/', upload_file, name='uploading'),
    path('explain-form/', get_explanation, name='uploading'),
]