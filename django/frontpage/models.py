from django.db import models

# Create your models here.
class Upload(models.Model):
    img = models.ImageField(upload_to='images')
