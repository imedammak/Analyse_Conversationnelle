# Generated by Django 4.2.2 on 2023-06-30 03:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('store', '0018_invitecode'),
    ]

    operations = [
        migrations.AddField(
            model_name='sentiment',
            name='converted_plot',
            field=models.TextField(blank=True, null=True),
        ),
    ]
