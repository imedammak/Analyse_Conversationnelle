# Generated by Django 4.2.1 on 2023-06-07 21:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('store', '0012_delete_analyses'),
    ]

    operations = [
        migrations.AddField(
            model_name='appel',
            name='tags',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
    ]
