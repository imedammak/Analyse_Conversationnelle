# Generated by Django 4.2.2 on 2023-06-19 13:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('store', '0016_sentiment'),
    ]

    operations = [
        migrations.AddField(
            model_name='appel',
            name='left_percentage',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='appel',
            name='right_percentage',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
