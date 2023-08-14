from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import User 
from django.core.exceptions import ValidationError

class LoginForm(forms.Form):
    username = forms.CharField(widget=forms.TextInput(attrs={
        'class': 'form-control',
        'placeholder': 'Username',
    }))
    password = forms.CharField(widget=forms.TextInput(attrs={
        'class': 'form-control',
        'placeholder': 'Password',
        'type': 'password'
    }))

class RegisterForm(UserCreationForm):
    name = forms.CharField(max_length=120, required=True)
    address = forms.CharField(max_length=220, required=True)
    email = forms.EmailField(required=True)
    invitation_code = forms.CharField(max_length=12)  


    class Meta:
        model = User
        fields = ["username", "name", "address", "email", "password1", "password2"]

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            msg = "This email is already registered."
            self.add_error('email', msg)
            return
        return email
