from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

from .views import (
    create_user,
    UserProfileListView,
    AppelListView,
    create_appel,
    analyse_list_view,
    create_sentiment,
    send_email,
    ajax_generate_invite_code,
    delete_user,

)

urlpatterns = [
    path('create-user/', create_user, name='create-user'),
    path('user-list/', UserProfileListView.as_view(), name='user-list'),
    path('create-appel/', create_appel, name='create-appel'),
    path('appel-list/', AppelListView.as_view(), name='appel-list'),
    path('transcription/<int:pk>/', views.transcription_view, name='transcription-view'),
    path('analyse-list/', views.analyse_list_view, name='analyse-list'),
    path('create-sentiment/', create_sentiment, name='create-sentiment'),
    path('send_email/', views.send_email, name='send_email'),
    path('ajax_generate_invite_code/', ajax_generate_invite_code, name='ajax_generate_invite_code'),
    path('user/delete/<int:user_id>/', views.delete_user, name='delete_user'),




 

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
