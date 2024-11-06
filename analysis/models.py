# analysis/models.py

from django.db import models

class Customer(models.Model):
    customer_id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=100, blank=True, null=True)
    age = models.IntegerField(blank=True, null=True)
    gender = models.CharField(max_length=1, blank=True, null=True)
    purchase_date = models.DateField(blank=True, null=True)
    product_category = models.CharField(max_length=50, blank=True, null=True)
    amount_spent = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)

    def __str__(self):
        return f"{self.name} ({self.customer_id})"
