from django.db import models

# Create your models here.
class Upload(models.Model):
    title = models.CharField(max_length=50)
    img = models.ImageField(upload_to='pics')

    def __str__(self):
        return self.title