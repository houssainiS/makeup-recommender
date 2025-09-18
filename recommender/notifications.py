from django.core.mail import send_mail
from django.utils import timezone
from .models import Purchase

def send_usage_expiry_notifications():
    """
    Check all purchases whose usage_duration has ended and send email.
    """
    now = timezone.now()
    purchases_to_notify = Purchase.objects.filter(notified=False, purchase_date__lte=now)

    for purchase in purchases_to_notify:
        expiry = purchase.purchase_date + timezone.timedelta(days=purchase.usage_duration_days)
        if now >= expiry:
            subject = f"Your product '{purchase.product_name}' usage has ended"
            message = (
                f"Hello,\n\n"
                f"Your usage duration for the product '{purchase.product_name}' has ended today.\n"
                f"Thank you for using our service!"
            )
            send_mail(
                subject,
                message,
                None,  # uses DEFAULT_FROM_EMAIL
                [purchase.email],
                fail_silently=False,
            )
            purchase.notified = True
            purchase.save()
