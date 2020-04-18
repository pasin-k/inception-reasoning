from django.urls import path
from django.contrib import admin
from .views import HomePageView, uploads

urlpatterns = [
    path('', HomePageView.as_view(), name='home'),
    path('upload-form/', uploads, name='uploading'),
]