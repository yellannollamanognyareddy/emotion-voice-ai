from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('receive_image/', views.receive_image, name='receive_image'),
    path('voice_input/', views.voice_input, name='voice_input'),
    path('english_voice_input/', views.english_voice_input, name='english_voice_input'),
]