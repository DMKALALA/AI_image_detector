from django import template

register = template.Library()

@register.filter
def zip_lists(list1, list2):
    """Zip two lists together"""
    return zip(list1, list2)

@register.filter
def percentage(value):
    """Convert decimal to percentage"""
    try:
        return float(value) * 100
    except (ValueError, TypeError):
        return 0
