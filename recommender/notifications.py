from django.core.mail import send_mail
from django.utils import timezone
from .models import Purchase

def send_usage_expiry_notifications():
    """
    Check all purchases whose usage_duration has ended and send email.
    Added debugging prints and improved timezone handling.
    """
    now = timezone.now()
    print(f"[DEBUG] Current time (UTC): {now}")

    # Get purchases that are not notified
    purchases_to_notify = Purchase.objects.filter(notified=False)
    print(f"[DEBUG] Found {purchases_to_notify.count()} purchases to check")

    for purchase in purchases_to_notify:
        # Ensure purchase_date is timezone-aware
        purchase_date = purchase.purchase_date
        if timezone.is_naive(purchase_date):
            purchase_date = timezone.make_aware(purchase_date, timezone.get_current_timezone())

        expiry = purchase_date + timezone.timedelta(days=purchase.usage_duration_days)
        print(f"[DEBUG] Checking purchase {purchase.id} for {purchase.email}")
        print(f"[DEBUG] Purchase date: {purchase_date}, Usage days: {purchase.usage_duration_days}, Expiry: {expiry}")

        if now >= expiry:
            subject = f"Your product '{purchase.product_name}' usage has ended"
            message = (
                f"Hello,\n\n"
                f"Your usage duration for the product '{purchase.product_name}' has ended today.\n"
                f"Thank you for using our service!"
            )
            try:
                send_mail(
                    subject,
                    message,
                    None,  # uses DEFAULT_FROM_EMAIL
                    [purchase.email],
                    fail_silently=False,
                )
                print(f"[DEBUG] Email sent to {purchase.email}")
                purchase.notified = True
                purchase.save()
            except Exception as e:
                print(f"[ERROR] Failed to send email to {purchase.email}: {e}")
        else:
            print(f"[DEBUG] Purchase {purchase.id} not yet expired.")
