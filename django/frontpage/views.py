from django.shortcuts import render
from django.views.generic import ListView # new
from django.urls import reverse_lazy # new
from django.conf import settings

from pathlib import Path
import os, sys
from PIL import Image as PILImg
from PIL import ImageEnhance as PILImageEnhance
from .forms import UploadForm
from .models import Upload

sys.path.append('..')
from algorithm.main import predict

# Create your views here.
class HomePageView(ListView):
    model = Upload
    template_name = 'home.html'


def upload_file(request):
    template = "uploads.html"
    folder = "images/"
    data = {}
    if request.POST:
        userform = UploadForm(request.POST, request.FILES)
        if userform.is_valid():
            #TODO: Connect this part with our model
            origin_form = userform.cleaned_data["user_file"]
            origin_name = origin_form.name
            original_file = Path(settings.MEDIA_ROOT).joinpath(folder, origin_name)
            predicted_name = original_file.stem + "_prediction.png"
            prediction_file = Path(settings.MEDIA_ROOT).joinpath(folder, predicted_name)
            if original_file.is_file():
                original_file.unlink()
            if prediction_file.is_file():
                prediction_file.unlink()

            # resize image to 299,299
            image = PILImg.open(origin_form)
            image = image.resize((299, 299), PILImg.ANTIALIAS)
            image.save(original_file, 'PNG')

            prediction_array = predict(original_file)
            predict_image = PILImg.fromarray(prediction_array)
            predict_image.save(prediction_file, 'PNG')

            data.update(title=userform['title'].value)
            data.update(origin_name=origin_name)
            data.update(folder_dir=folder)
            data.update(thumb_name=predicted_name)
            userform = UploadForm()
    else:
        userform = UploadForm()

    data.update(userform=userform)
    return render(request, template, data)




