from django.shortcuts import render
from django.views.generic import ListView  # new
from django.urls import reverse_lazy  # new
from django.conf import settings

from pathlib import Path
import os, sys, json
from PIL import Image as PILImg
from PIL import ImageEnhance as PILImageEnhance
from .forms import UploadForm, ExplainForm
from .models import Upload

sys.path.append('..')
import numpy as np
from algorithm.main import predict, explain, get_model


# Create your views here.
class HomePageView(ListView):
    model = Upload
    template_name = 'home.html'


def upload_file(request):
    template = "uploads.html"
    folder = "images/"
    data = {}
    if request.POST:
        if 'upload_image' in request.POST:
            userform = UploadForm(request.POST, request.FILES)
            if userform.is_valid():

                origin_form = userform.cleaned_data["user_file"]
                origin_name = origin_form.name
                original_file = Path(settings.MEDIA_ROOT).joinpath(folder, origin_name)
                if original_file.is_file():
                    original_file.unlink()

                # resize image to 299,299
                image = PILImg.open(origin_form)
                image = image.resize((299, 299), PILImg.ANTIALIAS)
                image.save(original_file, 'PNG')

                # Get the top5 prediction & Save it as temp file
                _, prediction, _ = predict(original_file)
                decoded_prediction = {'class_name': [i[1] for i in prediction[0]],
                                      # Prediction has [id, class_name, confident]
                                      'confident': [str(i[2]) for i in prediction[0]],
                                      'file_name': str(original_file),
                                      }

                prediction_path = Path(settings.MEDIA_ROOT).joinpath('json', 'prediction_temp.json')
                with open(prediction_path, 'w') as f:
                    json.dump(decoded_prediction, f)

                # Update dictionary to display in uploads.html
                data.update(origin_name=origin_name)
                data.update(folder_dir=folder)
                data.update(prediction=decoded_prediction['class_name'])
                # data.update(thumb_name=predicted_name)
                explainform = ExplainForm()
                data.update(explainform=explainform)
        elif 'explain_image' in request.POST:
            explainform = ExplainForm(request.POST, request.FILES)
            if explainform.is_valid():
                choice = int(explainform.cleaned_data["Top_5_prediction"])
                prediction_path = Path(settings.MEDIA_ROOT).joinpath('json', 'prediction_temp.json')
                with open(prediction_path) as f:
                    data = json.load(f)
                    class_name = data['class_name']
                    confident = data['confident']
                    original_file = Path(data['file_name'])

                images, my_model = get_model(original_file)
                prediction_array = explain(images, my_model, prediction_rank=choice, show_img=False)
                predicted_name = original_file.stem + "_prediction.png"
                prediction_file = Path(settings.MEDIA_ROOT).joinpath(folder, predicted_name)
                original_file = original_file.stem + original_file.suffix
                if prediction_file.is_file():
                    prediction_file.unlink()
                if prediction_array.dtype == np.float32 or prediction_array.dtype == np.float64:
                    prediction_array = np.uint8(prediction_array * 255)
                predict_image = PILImg.fromarray(prediction_array)
                predict_image.save(prediction_file, 'PNG')

                userform = UploadForm()
                data.update(class_name=class_name[choice])
                data.update(confident=round(float(confident[choice]) * 100,2))
                data.update(origin_name=original_file)
                data.update(predict_name=predicted_name)
                data.update(folder_dir=folder)
                data.update(userform=userform)  # Refresh form again
    else:
        userform = UploadForm()
        data.update(userform=userform)

    return render(request, template, data)
