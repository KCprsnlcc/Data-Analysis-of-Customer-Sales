# analysis/admin.py

from django.contrib import admin
from .models import Customer  # Import your models

@admin.register(Customer)
class CustomerAdmin(admin.ModelAdmin):
    list_display = ('customer_id', 'name', 'age', 'gender', 'purchase_date', 'product_category', 'amount_spent')
    search_fields = ('name', 'product_category')
    list_filter = ('gender', 'product_category')
