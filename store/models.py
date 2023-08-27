from django.db import models
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.urls import reverse
from users.models import User
from django.http import HttpRequest  
import os
from urllib.parse import urlparse
from django.conf import settings
import whisper
from pydub import AudioSegment
from pydub.silence import split_on_silence
from whisper import load_model
import datetime
import numpy as np

import requests
import json

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=120, unique=True)
    address = models.CharField(max_length=220)
    created_date = models.DateField(auto_now_add=True)

    def __str__(self):
        return self.name

class InviteCode(models.Model):
    code = models.CharField(max_length=12)
    used = models.BooleanField(default=False)

class Appel(models.Model):
    record = models.URLField(max_length=200, null=True, blank=True)
    firstname = models.CharField(max_length=100, null=True, blank=True)
    total_duration = models.CharField(max_length=8, null=True, blank=True)
    to_number = models.CharField(max_length=100, null=True, blank=True)
    from_number = models.CharField(max_length=100, null=True, blank=True)
    call_id = models.CharField(max_length=100, null=True, blank=True)
    transcription = models.TextField(null=True, blank=True)
    tags = models.CharField(max_length=100, null=True, blank=True)
    left_percentage = models.FloatField(null=True, blank=True)  
    right_percentage = models.FloatField(null=True, blank=True)  

     
    def transcribe_audio(self, mp3_file_path):
            # Convert mp3 to wav
            wav_file_path = os.path.splitext(mp3_file_path)[0] + ".wav"
            sound = AudioSegment.from_mp3(mp3_file_path)
            sound.export(wav_file_path, format="wav")

            # Load whisper model
            model_size = 'small'
            model = load_model(model_size)

            # Split the audio file into segments based on silence
            segments = split_on_silence(sound, min_silence_len=500, silence_thresh=-50, keep_silence=500)
            total_duration_left = 0
            total_duration_right = 0
            # Open the result file in write mode with UTF-8 encoding
            with open('TEST.txt', 'w', encoding='utf-8') as file:
                prev_duration = 0  # Track the cumulative duration of previous segments
                # Transcribe each segment with Whisper and identify the position of the segment
                for i, segment in enumerate(segments):
                    # Separate the segment into left and right channels
                    left_channel = segment.split_to_mono()[0]
                    right_channel = segment.split_to_mono()[1]

                    # Calculate the decibel level difference between the two channels
                    pan = left_channel.dBFS - right_channel.dBFS

                    # Determine the position of the segment
                    if pan > 0:
                        position = 'Right'
                        total_duration_right += len(segment)

                    else:
                        position = 'Left'
                        total_duration_left += len(segment)


                    # Export the segment to a temporary file
                    segment.export(f"segment{i}.wav", format="wav")

                    # Transcribe the segment with Whisper
                    result = model.transcribe(f"segment{i}.wav")

                    if result['segments']:
                        result_text = result['segments'][0]['text']
                    else:
                        result_text = ""

                    # Calculate the start time of the segment
                    start_time = datetime.timedelta(milliseconds=prev_duration)
                    start_time = str(start_time)[:-4]  # Remove milliseconds

                    # Write the segment and its transcription to the result file
                    try:
                        file.write(f"{start_time} - Position: {position}\n")
                        file.write(f"Transcription: {result_text}\n\n")
                    except UnicodeEncodeError:
                        # Handle UnicodeEncodeError and print a message
                        file.write(f"Segment {i} - {start_time} - Position: {position}\n")
                        file.write("Transcription: <Unable to display Arabic characters>\n\n")
                        print("Unable to display Arabic characters in transcription.")

                    # Update the cumulative duration
                    prev_duration += len(segment)

            # Read the contents of the result file
            with open('TEST.txt', 'r', encoding='utf-8') as file:
                transcription = file.read()

            # Remove the temporary files
            for i in range(len(segments)):
                os.remove(f"segment{i}.wav")
                
                
            total_duration = total_duration_left + total_duration_right
            left_percentage = round((total_duration_left / total_duration) * 100, 2)  
            right_percentage = round((total_duration_right / total_duration) * 100, 2)  


            return transcription, left_percentage, right_percentage

   
    def save(self, *args, **kwargs):
        # Download the audio file
        audio_url = self.record
        response = requests.get(audio_url)
        if response.status_code == 200:
            # Get the file name from the URL
            file_name = os.path.basename(urlparse(audio_url).path)
            
            # Save the audio file locally
            file_path = os.path.join(settings.MEDIA_ROOT, file_name)
            with open(file_path, 'wb') as file:
                file.write(response.content)
            
            # Perform operations on the audio file (transcription, etc.)
            transcription = self.transcribe_audio(file_path)
            transcription, self.left_percentage, self.right_percentage = self.transcribe_audio(file_path)

            # Update the transcription field in the model
            self.transcription = transcription
        
        super().save(*args, **kwargs)

class Sentiment(models.Model):
    audio_file = models.FileField(upload_to='audio_files/')  # Store the original audio file
    converted_text = models.TextField(blank=True, null=True)  # Store the converted text
    pourcentage_positive = models.FloatField(default=0)
    pourcentage_negative = models.FloatField(default=0)
    pourcentage_neutral = models.FloatField(default=0)


    def convert_audio(self):
        # Convert the audio file to wav
        mp3_file_path = self.audio_file.path
        print(mp3_file_path)

        wav_file_path = os.path.splitext(mp3_file_path)[0] + ".wav"
        sound = AudioSegment.from_mp3(mp3_file_path)
        sound.export(wav_file_path, format="wav")

        # Load the Whisper model
        model_size = 'small'
        model = load_model(model_size)

        # Transcribe the left channel with Whisper
        sound = AudioSegment.from_wav(wav_file_path)
        left_channel = sound.split_to_mono()[0]

        # Convert AudioSegment to np.ndarray of float samples
        samples = np.array(left_channel.get_array_of_samples())
        samples = samples.astype(np.float32) / 32767.0  # Normalize the samples to the range [-1.0, 1.0]

        # Transcribe the audio samples
        transcription = model.transcribe(samples)
        self.converted_text = transcription["text"]

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        self.convert_audio()
        super().save(*args, **kwargs)  # Save the changes after conversion
        
