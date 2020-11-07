from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.views.generic import ListView, CreateView, UpdateView, DetailView, DeleteView
from django.contrib.auth.models import User
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.template.loader import render_to_string
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from home.forms import FileForm
import os
from os import listdir
from os.path import join
from os.path import isfile
import requests

import librosa
import numpy as np
from django.conf import settings
from django.views.generic import ListView
from django.views.generic import TemplateView
from django.views.generic.edit import CreateView
from rest_framework import views
from rest_framework import status
from rest_framework.generics import get_object_or_404
from rest_framework.parsers import FormParser
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework.renderers import TemplateHTMLRenderer

from home.models import FileModel, SkinCancer
from home.serialize import FileSerializer

import tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras_preprocessing import image
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
import efficientnet.tfkeras as efn

# import cv2
from PIL import Image


def build_model(backbone, lr=1e-4):
    model = keras.Sequential()
    model.add(backbone)
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(2, activation='softmax'))

    return model


# Create your views here.
def starting(request):
    return render(request, 'home/starting.html', {})


def about(request):
    return render(request, 'home/about.html', {})


class FilesList(ListView):
    """
    ListView that display companies query list.
    :param model: Specifies the objects of which model we are listing
    :param template_name; Specifies the static display template file.
    :param context_object_name: Custom defined context object value,
                     this can override default context object value.
    """
    model = FileModel
    template_name = 'home/files_list.html'
    context_object_name = 'files_list'


class UploadView(CreateView):
    """
    This is the view that is used by the user of the web UI to upload a file.
    :param model: Specifies the objects of which model we are listing
    :param template_name; Specifies the static display template file.
    :param fields: Specifies the model field to be used
    :param success_url: Specifies the redirect url in case of successful upload.
    """
    model = FileModel
    fields = ['file']
    template_name = 'home/post_file.html'
    success_url = '/'

    def form_valid(self, form):
        messages.success(self.request,
                         f'The Image File Is successfully uploaded')
        return super().form_valid(form)


class SelectPredFileView(TemplateView):
    """
    This view is used to select a file from the list of files in the server.
    After the selection, it will send the file to the server.
    The server will return the predictions.
    """

    template_name = 'home/select_file_predictions.html'
    parser_classes = FormParser
    queryset = FileModel.objects.all()

    def get_context_data(self, **kwargs):
        """
        This function is used to render the list of files in the MEDIA_ROOT in the html template.
        """
        context = super().get_context_data(**kwargs)
        media_path = settings.MEDIA_ROOT
        myfiles = [
            f for f in listdir(media_path) if isfile(join(media_path, f))
        ]
        context['filename'] = myfiles
        # print(myfiles)
        return context


def deletefile(request, myfile):
    print(myfile)
    FileModel.objects.get(file=myfile).delete()
    return redirect('file_select')


def predictfile(request, myfile):
    print(myfile)
    f = FileModel.objects.get(file=myfile).file
    print(f)
    images = image.load_img(f, target_size=(224, 224))
    x = image.img_to_array(images)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    efficientnetb3 = efn.EfficientNetB0(weights='imagenet',
                                        input_shape=(224, 224, 3),
                                        include_top=False)

    model = build_model(efficientnetb3)
    model_name = 'efficientnet.h5'
    model.load_weights(os.path.join(settings.MODEL_ROOT, model_name))
    mal = model.predict(x)
    res1 = np.argmax(mal, axis=1)[0]
    print(res1)
    if res1 == 1:
        mobile = tensorflow.keras.applications.mobilenet.MobileNet()
        x2 = mobile.layers[-6].output

        x2 = Dropout(0.25)(x2)
        predictions = Dense(7, activation='softmax')(x2)

        mmodel = Model(inputs=mobile.input, outputs=predictions)
        model_name = 'model.h5'
        mmodel.load_weights(os.path.join(settings.MODEL_ROOT, model_name))
        arr = mmodel.predict(x)
        type = np.argmax(arr, axis=1)[0]
        type2 = SkinCancer.objects.get(idskin=type)
    return render(request, 'home/prediction.html', {
        'result1': res1,
        'type': type2,
        'img': myfile
    })


class Predict(views.APIView):
    """
    This class is used to making predictions.

    Example of input:
    {'filename': '01-01-01-01-01-01-01.wav'}

    Example of output:
    [['neutral']]
    """

    template_name = 'home/starting.html'
    # Removing the line below shows the APIview instead of the template.
    renderer_classes = [TemplateHTMLRenderer]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_name = 'Emotion_Voice_Detection_Model.h5'
        self.graph = tf.get_default_graph()
        self.loaded_model = keras.models.load_model(
            os.path.join(settings.MODEL_ROOT, model_name))
        self.predictions = []

    def file_elaboration(self, filepath):
        """
        This function is used to elaborate the file used for the predictions with librosa.
        :param filepath:
        :return: predictions
        """
        data, sampling_rate = librosa.load(filepath)
        try:
            mfccs = np.mean(librosa.feature.mfcc(y=data,
                                                 sr=sampling_rate,
                                                 n_mfcc=40).T,
                            axis=0)
            training_data = np.expand_dims(mfccs, axis=2)
            training_data_expanded = np.expand_dims(training_data, axis=0)
            numpred = self.loaded_model.predict_classes(training_data_expanded)
            self.predictions.append([self.classtoemotion(numpred)])
            return self.predictions
        except ValueError as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

    def post(self, request):
        """
        This method is used to making predictions on audio files
        loaded with FileView.post
        """
        with self.graph.as_default():
            filename = request.POST.getlist('file_name').pop()
            filepath = str(os.path.join(settings.MEDIA_ROOT, filename))
            predictions = self.file_elaboration(filepath)
            try:
                print(predictions)
                messages.success(
                    request,
                    f'The Emotion In This Audio File is {predictions[0][0]}')
                return Response({'predictions': predictions.pop()},
                                status=status.HTTP_200_OK)
            except ValueError as err:
                return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

    @staticmethod
    def classtoemotion(pred):
        """
        This method is used to convert the predictions (int) into human readable strings.
        ::pred:: An int from 0 to 7.
        ::output:: A string label

        Example:
        classtoemotion(0) == neutral
        """

        label_conversion = {
            '0': 'neutral',
            '1': 'calm',
            '2': 'happy',
            '3': 'sad',
            '4': 'angry',
            '5': 'fearful',
            '6': 'disgust',
            '7': 'surprised'
        }

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value

        print(label)
        return label