from django.contrib import admin
from django.urls import path, include

from home import views
from .views import starting, about, UploadView, SelectPredFileView, FilesList, Predict, deletefile, predictfile

urlpatterns = [
    path('', starting, name='home-home'),
    path('about/', about, name='home-about'),
    path('upload/', UploadView.as_view(), name='upload_file'),

    # Url to select a file for the predictions
    path('fileselect/', SelectPredFileView.as_view(), name='file_select'),

    # Url to list all the files in the server
    path('files_list/', FilesList.as_view(), name='files_list'),
    path('predict/<str:myfile>', predictfile, name='predict'),
    path('deletefile/<str:myfile>', deletefile, name='deletefile')
]