from django.contrib import admin
from .models import (
    UserProfile,
    Appel,
    Sentiment,
    InviteCode,
)

class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'name', 'address', 'created_date']

admin.site.register(UserProfile, UserProfileAdmin)
admin.site.register(Appel)
admin.site.register(Sentiment)
admin.site.register(InviteCode)
