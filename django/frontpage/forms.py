from django import forms
from django.conf import settings
from .models import Upload
from pathlib import Path
import json

class UploadForm(forms.Form):
    # title = forms.CharField(max_length=50)
    user_file = forms.FileField()

    def clean_user_file(self, *args, **kwargs):
        cleaned_data = super(UploadForm,self).clean()
        user_file = cleaned_data.get("user_file")

        if user_file:
            if user_file.size > 5*1024*1024:
                raise forms.ValidationError("File is too big")
            if not Path(user_file.name).suffix.strip().lower() in ['.jpg','.png','.gif','.jpeg']:
                raise forms.ValidationError("File does not look like as picture.")
        return user_file


def get_my_choices():
    prediction_path = Path(settings.MEDIA_ROOT).joinpath('json', 'prediction_temp.json')
    with open(prediction_path) as f:
        data = json.load(f)
        class_name = data['class_name']
        confident = data['confident']
    class_name = [(i, "{}, Confident: {}%".format(n,float(c)*100)) for i,(n,c) in enumerate(zip(class_name, confident))]
    return class_name


class ExplainForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super(ExplainForm, self).__init__(*args, **kwargs)
        self.fields['my_choice_field'] = forms.ChoiceField(choices=get_my_choices())
