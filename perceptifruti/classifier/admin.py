from django.contrib import admin
from .models import *


@admin.register(FruitReading)
class FruitReading(admin.ModelAdmin):
    list_display = ['fruit', 'reading_display', 'read_display']
    readonly_fields = ['read_display']
