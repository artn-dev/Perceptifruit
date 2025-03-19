from django.contrib import admin
from .models import *


@admin.register(Fruit)
class FruitAdmin(admin.ModelAdmin):
    list_display = ['name']
