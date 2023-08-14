from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from store.models import UserProfile, Appel

@login_required(login_url='login')
def dashboard(request):
    total_users = UserProfile.objects.count()
    total_appels = Appel.objects.count()

    context = {
        'total_users': total_users,
        'total_appels': total_appels,
    }
    return render(request, 'dashboard.html', context)
