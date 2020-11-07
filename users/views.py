from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from .forms import UserRegisterForm, UserUpdateForm, ProfileUpdateForm
from django.contrib.auth.decorators import login_required
# Create your views here.

def reg(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, f'Account Created')
            return redirect('home-home')
    else:
        form = UserRegisterForm()
    return render(request, 'users/register.html', {'form': form})

@login_required
def profile(request):
    return render(request, 'users/profile.html')

@login_required
def update(request):
    if request.method == 'POST':
        form1 = UserUpdateForm(request.POST, instance=request.user)
        form2 = ProfileUpdateForm(request.POST,request.FILES)
        if form1.is_valid() and form2.is_valid():
            form1.save()
            messages.success(request, f'Profile Updated')
            return redirect('users-profile')
    else:
        form1 = UserUpdateForm(instance=request.user)
        form2 = ProfileUpdateForm()
    return render(request, 'users/update.html', {'form1':form1, 'form2':form2})