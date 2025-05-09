from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('<str:tool>/', views.tool_panel, name='tool_panel'),
]
