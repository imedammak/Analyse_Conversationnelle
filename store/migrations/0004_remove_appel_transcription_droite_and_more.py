# Generated by Django 4.2.1 on 2023-06-02 12:46

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('store', '0003_appel_transcription_droite_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='appel',
            name='transcription_droite',
        ),
        migrations.RemoveField(
            model_name='appel',
            name='transcription_gauche',
        ),
    ]