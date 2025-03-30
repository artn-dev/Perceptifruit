from enum import Enum
from django.utils.translation import gettext_lazy as _


class Ripeness(Enum):
    GREEN = 'A', _('Verde')
    RIPENING = 'B', _('Amadurecendo')
    RIPE = 'C', _('Madura')
    OVERRIPE = 'D', _('Passada')

    def __new__(cls, value, label):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label = label
        return obj

    @classmethod
    def choices(cls):
        return [(key.name, key.label) for key in cls]
