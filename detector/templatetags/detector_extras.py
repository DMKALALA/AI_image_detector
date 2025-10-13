from django import template

register = template.Library()

@register.filter
def zip_lists(list1, list2):
    """Zip two lists together"""
    return zip(list1, list2)
