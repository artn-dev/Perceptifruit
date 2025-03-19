import os
import sys
from .nn import RipenessClassifier
from django.apps import AppConfig
from django.conf import settings as s


class ClassifierConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'classifier'
    model = None

    def _setup_classifier(self):
        files = os.listdir(s.CLASSIFIER_PATH)
        if not files:
            return

        filename = os.path.join(s.CLASSIFIER_PATH, files[0])
        ClassifierConfig.model = RipenessClassifier(filename)

    def ready(self):
        if len(sys.argv) > 1 and sys.argv[1] == 'runserver':
            self._setup_classifier()
