from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from .forms import LoginForm, RegisterForm
from store.models import UserProfile  
from django.contrib import messages
from store.models import InviteCode

def login_page(request):
    forms = LoginForm()
    if request.method == 'POST':
        forms = LoginForm(request.POST)
        if forms.is_valid():
            username = forms.cleaned_data['username']
            password = forms.cleaned_data['password']
            user = authenticate(username=username, password=password)
            if user:
                login(request, user)
                return redirect('dashboard')
    context = {'form': forms}
    return render(request, 'users/login.html', context)

def register(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)
        invitation_code = request.POST.get('invitation_code')

        # Vérifier si le code d'invitation est valide
        try:
            invite_code = InviteCode.objects.get(code=invitation_code)
            if invite_code.used:
                messages.error(request, 'This invitation code has already been used.')
                return render(request, "users/register.html", {'form': form})

        except InviteCode.DoesNotExist:
            messages.error(request, 'Invalid invitation code')
            return render(request, "users/register.html", {'form': form})

        if form.is_valid():
            user = form.save()
            name = form.cleaned_data.get('name')
            address = form.cleaned_data.get('address')
            profile = UserProfile(user=user, name=name, address=address)
            profile.save()

            # Marquez le code comme utilisé
            invite_code.used = True
            invite_code.save()

            login(request, user)
            return redirect('dashboard')

        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, error)
    else:
        form = RegisterForm()
    return render(request, "users/register.html", {'form': form})



def logout_page(request):
    logout(request)
    return redirect('login')

def user_list(request):
    users = UserProfile.objects.all()
    return render(request, "users/user-list.html", {'users': users})
