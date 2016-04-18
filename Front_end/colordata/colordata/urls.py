"""colordata URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.9/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from deeplearning import views as deeplearning_views
urlpatterns = [
    url(r'^$', deeplearning_views.home, name='home'),
    url(r'^admin/', admin.site.urls),
    url(r'^age_prediction/$', deeplearning_views.age_prediction, name='age_prediction'),
    url(r'^age_prediction_1/$', deeplearning_views.age_prediction_1, name='age_prediction_1'),
    url(r'^age_prediction_2/$', deeplearning_views.age_prediction_2, name='age_prediction_2'),
    url(r'^age_prediction_3/$', deeplearning_views.age_prediction_3, name='age_prediction_3'),
    url(r'^age_prediction_4/$', deeplearning_views.age_prediction_4, name='age_prediction_4'),
    url(r'^age_prediction_5/$', deeplearning_views.age_prediction_5, name='age_prediction_5'),
    url(r'^age_prediction_6/$', deeplearning_views.age_prediction_6, name='age_prediction_6'),
    url(r'^age_prediction_7/$', deeplearning_views.age_prediction_7, name='age_prediction_7'),
    url(r'^age_prediction_8/$', deeplearning_views.age_prediction_8, name='age_prediction_8'),
    url(r'^age_prediction_9/$', deeplearning_views.age_prediction_9, name='age_prediction_9'),
    url(r'^age_prediction_10/$', deeplearning_views.age_prediction_10, name='age_prediction_10'),
    url(r'^age_progression/$', deeplearning_views.age_progression, name='age_progression')
]
