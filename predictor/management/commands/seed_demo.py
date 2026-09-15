from django.core.management.base import BaseCommand
from django.core.management import call_command
from django.db import connection
from django.contrib.auth.models import User
from predictor.models import Company, Membership, UserPrediction
from predictor.views import _provision_company
from predictor.model_service import rescore_audience


def ensure_predictor_schema():
    """Repair a stale demo SQLite migration state before seeding."""
    required = {
        'predictor_company', 'predictor_membership', 'predictor_userprediction',
        'predictor_trainingrun', 'predictor_predictionfeedback', 'predictor_dataset'
    }
    with connection.cursor() as cursor:
        existing = set(connection.introspection.table_names(cursor))
    missing = required - existing
    if not missing:
        return
    # If Django's migration history says predictor is applied while tables are missing,
    # roll back only the predictor app and recreate it. This does not touch model/data files.
    call_command('migrate', 'predictor', 'zero', verbosity=0)
    call_command('migrate', 'predictor', verbosity=0)


class Command(BaseCommand):
    help = 'Create demo company/user and seed its audience predictions.'

    def handle(self, *args, **opts):
        ensure_predictor_schema()
        u, _ = User.objects.get_or_create(
            username='demo_admin',
            defaults={'email': 'demo@vibeiq.local', 'is_staff': True}
        )
        u.set_password('demo1234')
        u.is_staff = True
        u.save()
        c, _ = Company.objects.get_or_create(
            slug='demo-company', defaults={'name': 'Demo Streaming Company'}
        )
        Membership.objects.update_or_create(
            user=u, defaults={'company': c, 'role': 'OWNER'}
        )
        _provision_company(c)
        out = rescore_audience(c)
        UserPrediction.objects.filter(company=c).delete()
        objs = [
            UserPrediction(
                company=c, user_id=str(r.user_id), segment=str(r.segment),
                churn_probability=float(r.churn_probability),
                engagement_health=float(r.engagement_health),
                predicted_ltv=float(r.predicted_ltv),
                next_action=str(r.predicted_next_action), risk=str(r.risk),
                recommendation=str(r.recommendation)
            )
            for r in out.itertuples()
        ]
        UserPrediction.objects.bulk_create(objs, batch_size=1000)
        self.stdout.write(self.style.SUCCESS(
            f'Seeded {len(objs):,} users for {c.name}. Login: demo_admin / demo1234'
        ))
