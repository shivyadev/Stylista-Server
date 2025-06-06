# Generated by Django 5.1.7 on 2025-03-24 14:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ai_models', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='UserInput',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user_id', models.CharField(max_length=100)),
                ('image_url', models.CharField(max_length=255)),
                ('usage', models.CharField(max_length=100)),
                ('gender', models.CharField(max_length=20)),
                ('outfits', models.JSONField(default=list)),
            ],
        ),
    ]
