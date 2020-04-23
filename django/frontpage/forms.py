from django import forms
from .models import Upload
from pathlib import Path

class UploadForm(forms.Form):
    title = forms.CharField(max_length=50)
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
