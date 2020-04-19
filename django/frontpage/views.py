from django.shortcuts import render
from django.views.generic import ListView, CreateView # new
from django.urls import reverse_lazy # new
from django.conf import settings

from pathlib import Path
import os
from PIL import Image as PILImg
from PIL import ImageEnhance as PILImageEnhance
from .forms import UploadForm # new
from .models import Upload

class HomePageView(ListView):
    model = Upload
    template_name = 'home.html'


def upload_file(request):
    template = "uploads.html"
    folder = "images"
    data = {}
    if request.POST:
        userform = UploadForm(request.POST, request.FILES)
        print(userform)
        if userform.is_valid():
            origin_form = userform.cleaned_data["user_file"]
            origin_name = Path(folder) / origin_form.name
            original_file = Path(settings.MEDIA_ROOT).joinpath(origin_name)
            thumb_name = Path(folder) / (original_file.stem + "_thumb.jpg")
            thumb_file = Path(settings.MEDIA_ROOT).joinpath(thumb_name)
            if original_file.is_file():
                original_file.unlink()
            if thumb_file.is_file():
                thumb_file.unlink()
            with open(original_file, 'wb+') as f:
                f.write(origin_form.read())
            origin_form.seek(0)
            # resize image
            image = PILImg.open(origin_form)
            image = image.resize((150, 150), PILImg.ANTIALIAS)
            # sharpness image
            image = PILImageEnhance.Sharpness(image)
            image = image.enhance(1.3)
            print("Debug", thumb_file)
            image.save(thumb_file, 'JPEG')
            data.update(origin_name=origin_name)
            data.update(thumb_name=thumb_name)
            userform = UploadForm()
    else:
        userform = UploadForm()

    data.update(userform=userform)
    return render(request, template, data)
# class CreatePostView(CreateView):  # new
#     model = Upload
#     form_class = UploadForm
#     template_name = 'uploads.html'
#     success_url = reverse_lazy('home')


# # Create your views here.
# def index(request):
#     # dests = Destination.objects.all
#     return render(request, 'home.html')
#     # return render(request, 'index.html', {'dests':dests})
#
# def upload(request):
#     if request.method == 'POST':
#         form = UploadForm(request.POST, request.FILES)
