# Generated by Django 4.2.2 on 2023-06-29 11:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('store', '0017_appel_left_percentage_appel_right_percentage'),
    ]

    operations = [
        migrations.CreateModel(
            name='InviteCode',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('code', models.CharField(max_length=12)),
                ('used', models.BooleanField(default=False)),
            ],
        ),
    ]
