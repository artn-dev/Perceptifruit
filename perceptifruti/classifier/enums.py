from enum import Enum
from django.utils.translation import gettext_lazy as _


class Ripeness(Enum):
    GREEN = 'GREEN', _('Green')
    RIPENING = 'RIPENING', _('Ripening')
    RIPE = 'RIPE', _('Ripe')
    OVERRIPE = 'OVERRIPE', _('Overripe')
    # GREEN = 'GREEN', _('Verde')
    # RIPENING = 'RIPENING', _('Amadurecendo')
    # RIPE = 'RIPE', _('Madura')
    # OVERRIPE = 'OVERRIPE', _('Passada')

    def __new__(cls, value, label):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label = label
        return obj

    @classmethod
    def choices(cls):
        return [(key.name, key.label) for key in cls]
