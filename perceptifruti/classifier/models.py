from django.contrib import admin
from django.utils.timezone import localtime, now
from django.db import models
from .enums import Ripeness


class FruitReading(models.Model):
    fruit = models.ForeignKey('main.Fruit', on_delete=models.CASCADE)
    reading = models.CharField(max_length=8, choices=Ripeness.choices())
    read = models.DateTimeField(default=now)

    @admin.display(description='Reading')
    def reading_display(self):
        return Ripeness(self.reading).label

    @admin.display(description='Read')
    def read_display(self):
        return localtime(self.read).strftime('%d/%m/%Y %H:%M:%S')
