from django import forms
from .models import Upload
from pathlib import Path

class UploadForm(forms.Form):
    user_file = forms.FileField()

    def clean_user_file(self):
        cleaned_data = super(UploadForm,self).clean()
        user_file = cleaned_data.get("user_file")

        if user_file:
            print("Debug", user_file)
            if user_file.size > 5*1024*1024:
                raise forms.ValidationError("File is too big")
            if not Path(user_file).suffix.strip().lower() in ['.jpg','.png','.gif','.jpeg']:
                raise forms.ValidationError("File does not look like as picture.")
        return user_file

    # class Meta:
    #     model = Upload
    #     fields = ['title', 'img']