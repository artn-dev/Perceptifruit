from django.core.management.base import BaseCommand
from classifier.models import FruitReading
from classifier.enums import Ripeness
from datetime import datetime, timedelta
from django.utils import timezone
import random


class Command(BaseCommand):
    help = 'Popula a tabela FruitReading com dados de exemplo'

    def handle(self, *args, **options):
        # Configurações básicas
        fruit_id = 1
        start_date = timezone.make_aware(datetime(2025, 3, 27))
        end_date = timezone.make_aware(datetime(2025, 4, 4))
        
        # Padrão de maturação progressiva
        stage_data = {
            Ripeness.GREEN: [120, 110, 100, 90, 80, 70, 60, 50, 40],
            Ripeness.RIPENING: [20, 30, 40, 50, 60, 70, 80, 90, 100],
            Ripeness.RIPE: [5, 10, 15, 20, 30, 40, 50, 60, 70],
            Ripeness.OVERRIPE: [0, 0, 0, 5, 10, 15, 20, 25, 30]
        }

        # Limpa dados existentes (opcional)
        FruitReading.objects.filter(fruit_id=fruit_id).delete()

        total_created = 0
        current_date = start_date
        
        # Para cada dia no período
        day_index = 0
        while current_date <= end_date:
            for stage in Ripeness:
                # Quantidade base com pequena variação aleatória
                count = max(0, stage_data[stage][day_index] + random.randint(-5, 5))
                
                # Cria os registros com timezone
                for _ in range(count):
                    # Cria um datetime com timezone para o dia atual
                    random_time = timezone.make_aware(
                        datetime.combine(
                            current_date.date(),
                            datetime.min.time()
                        ).replace(
                            hour=random.randint(8, 18),
                            minute=random.randint(0, 59),
                            second=random.randint(0, 59)
                        )
                    )
                    
                    FruitReading.objects.create(
                        fruit_id=fruit_id,
                        reading=stage.value,
                        read=random_time
                    )
                    total_created += 1
            
            current_date += timedelta(days=1)
            day_index += 1

        self.stdout.write(
            self.style.SUCCESS(f'Criados {total_created} registros para fruta {fruit_id} '
                             f'({start_date.date()} a {end_date.date()})')
        )