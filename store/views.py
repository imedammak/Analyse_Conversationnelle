from django.shortcuts import render, redirect
from django.views.generic import ListView
from django.contrib.auth.decorators import login_required
from .models import UserProfile, Appel
from .forms import UserProfileForm, AppelForm
from django.shortcuts import get_object_or_404
from django.db.models import Q
from django.http import JsonResponse
from django.conf import settings
from store.models import UserProfile, Appel
from django.http import Http404
from datetime import timedelta
from django.http import FileResponse
from django.views.generic import TemplateView
import datetime
from .forms import SentimentForm
import nltk
import whisper
from pydub import AudioSegment
from django.contrib.auth import get_user_model  
from django.db.models import Sum
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
import io
import urllib, base64
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import threading
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import os
import random
import string
from django.contrib import messages
from django.core.mail import send_mail
from django.views.decorators.csrf import csrf_exempt
from .models import InviteCode
import plotly.graph_objects as go
import plotly
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import nltk
import pickle
from .models import Sentiment

nltk.download('stopwords')


def tokenize_sentences(text):
    sentences = sent_tokenize(text)
    return sentences

nltk.download('punkt')
User = get_user_model()
TOPICS = {
    "price": {
        "labels": ["subscription", "budget", "business_model", "cost", "quote", "billing", "financing", "free", "investment", "monthly_amount", "package", "payment", "pricing", "price", "tariff", "your_budget","prices"],
        "color": "red"
    },
    "interest": {
        "labels": ["interest", "satisfaction", "information", "interesting", "useful", "need", "demonstration", "questions", "win", "no_inconvenience", "proposal", "decision", "new", "thinks", "replace", "remains_open", "want_to_see", "want_to_know", "meet_need", "go_fast", "want"],
        "color": "blue"
    },
    "discovery": {
        "labels": ["would_have_liked", "concerns", "cost", "wanted_to_talk_about", "people", "organized", "how_organized", "how_work", "in_terms_of_need", "in_terms_of_needs", "do_you_have", "do_you_know", "can_you", "purpose_of_call", "purpose_of_our_call", "better_understand", "agenda", "tell_me_about", "delay", "role", "size", "time_frame", "needs", "what_do_you_want_to_do", "already_have_tools", "already_have_a_tool", "what_do_you_have", "are_you_comfortable_with", "how_many_are_you", "how_are_you", "are_you_familiar", "tell_me_more", "what_do_you_use", "how_you", "what_data", "how_much_have_you", "minimum_order", "discovery"],
        "color": "green"
    },
    "next_steps": {
        "labels": ["send_me", "agree_on_date", "contact_info", "shared_screen", "send_documentation", "make_appointment", "no_need_to_call_back", "get_back_to_you", "ask_for_information", "tell_you", "remind_you", "call_back_on", "next_step", "my_details", "put_me_in_touch", "feel_free_to_come_back", "dont_hesitate", "we_will_come_back_to_you", "we_wont_come_back_to_you", "talk_again", "little_recap", "for_the_continuation", "can_move_forward", "when_available", "demo_meeting", "come_back_to_me", "will_come_back_to_you", "do_not_come_back_to_you", "if_any_questions", "find_slot", "second_call", "quick_recap", "invitation", "will_get_closer", "send_small_invitation", "send_email", "make_meeting", "call_back", "send_document"],
        "color": "purple"
    },
    "objections": {
        "labels": ["problem", "fear", "honesty", "personal", "prioritization", "tool", "offer", "workload", "thinking", "budget_closed", "expectations", "complexity", "not_interested", "time", "misunderstanding", "waiting", "solutions", "budget", "already_working"],
        "color": "orange"
    },
    "wow": {
        "labels": ["great", "impression", "cool", "bravo", "nice", "design", "congratulations", "powerful", "impressive", "nice_work", "clean", "very_clear", "magic", "crazy", "wow", "playful", "super", "excellent"],
        "color": "pink"
    }
}

def generate_invite_code():
    characters = string.ascii_letters + string.digits + string.punctuation
    code_length = 12
    code = ''.join(random.choice(characters) for _ in range(code_length))
    return code

@csrf_exempt
def ajax_generate_invite_code(request):
    if request.method == 'POST':
        # Verifier si l'utilisateur est un super utilisateur
        if not request.user.is_superuser:
            return JsonResponse({'error': 'Vous n\'êtes pas autorisé à générer un code d\'invitation.'})
            
        # Générer un nouveau code d'invitation
        invitation_code = generate_invite_code()
        invite_code = InviteCode.objects.create(code=invitation_code, used=False)
        invite_code.save()

        return JsonResponse({'invitation_code': invitation_code})
    else:
        return JsonResponse({'error': 'Invalid request'})

    

@csrf_exempt
def send_email(request):
    if request.method == "POST":
        email = request.POST.get("email")
        subject = request.POST.get("subject")
        message = request.POST.get("message")
        from_email = request.POST.get("from_email")

        send_mail(subject, message, from_email, [email], fail_silently=False)

        return JsonResponse({"success": True})
    else:
        return JsonResponse({"success": False})

def create_user(request):
    if request.method == 'POST':
        form = UserProfileForm(request.POST)
        password = request.POST.get('password')
        retype_password = request.POST.get('retype_password')
        invitation_code = request.POST.get('invitation_code')

        if form.is_valid():
            if password == retype_password:
                # Vérifier si le code d'invitation est valide
                try:
                    invite_code = InviteCode.objects.get(code=invitation_code)
                    if not invite_code.used:
                        user = User.objects.create_user(
                            username=form.cleaned_data['username'],
                            password=password,
                            email=form.cleaned_data['email']
                        )
                        UserProfile.objects.create(user=user, name=form.cleaned_data['name'], address=form.cleaned_data['address'])

                        # Marquez le code comme utilisé
                        invite_code.used = True
                        invite_code.save()

                        return redirect('user-list')
                    else:
                        form.add_error('invitation_code', 'This invitation code has already been used.')
                except InviteCode.DoesNotExist:
                    form.add_error('invitation_code', 'Invalid invitation code')
            else:
                form.add_error('password', 'Passwords do not match')
    else:
        form = UserProfileForm()

    context = {'form': form}
    return render(request, 'store/create-user.html', context)



def user_list(request):
    users = UserProfile.objects.all()
    return render(request, 'user-list.html', {'users': users})

def delete_user(request, user_id):
    if request.user.is_superuser:
        user_profile = UserProfile.objects.get(id=user_id)
        user = user_profile.user
        user_profile.delete()
        user.delete()
        return redirect('user-list')
    else:
        return HttpResponse('Unauthorized', status=401)
    

class UserProfileListView(ListView):
    model = UserProfile
    template_name = 'store/user-list.html'
    context_object_name = 'users'
    
def calculate_channel_percentage(transcription):
    total_time = len(transcription)
    left_time = sum(1 for line in transcription if line['channel'] == 'left')
    right_time = total_time - left_time

    left_percentage = (left_time / total_time) * 100 if total_time else 0
    right_percentage = (right_time / total_time) * 100 if total_time else 0

    return left_percentage, right_percentage


def store_json_to_db(file_path, progress_bar):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        for call in data['call_list']:
            tags = ", ".join(tag['name'] for tag in call.get('tags', []))
            call_id = call.get('call_id')
            if not Appel.objects.filter(call_id=call_id).exists():
                # Convert incall_duration into timedelta object and then format it into a string
                incall_duration = timedelta(seconds=call['incall_duration'])
                incall_duration_str = str(incall_duration)
                print(f"incall_duration: {call['incall_duration']}, converted: {incall_duration_str}")  # Debug print
                
                # Calculate the left and right channel percentages if the transcription exists
                if 'transcription' in call:
                    left_percentage, right_percentage = calculate_channel_percentage(call['transcription'])
                else:
                    left_percentage = right_percentage = 0

                appel = Appel(
                    record=call['record'],
                    firstname=call['user']['firstname'],
                    total_duration=incall_duration_str,  # Change this to incall_duration_str
                    to_number=call['to_number'],
                    from_number=call['from_number'],
                    call_id=call_id ,
                    tags=tags,
                    left_percentage=left_percentage,
                    right_percentage=right_percentage,
                )
                appel.save()
                progress_bar.update(1)
            else:
                print(f"An Appel with call_id {call_id} already exists.")
                continue

          
def create_appel(request):
    # Retrieve the progress value from the session
    progress = request.session.get('progress', 0)

    if request.method == 'POST':
        form = AppelForm(request.POST, request.FILES)
        if form.is_valid():
            files = request.FILES.getlist('files')
            total_files = len(files)

            # Create a progress bar
            progress_bar = tqdm(total=total_files, desc="Processing", unit="file")

            def process_files():
                for file in files:
                    file_path = os.path.join(settings.MEDIA_ROOT, file.name)
                    with open(file_path, 'wb') as destination:
                        for chunk in file.chunks():
                            destination.write(chunk)
                    store_json_to_db(file_path, progress_bar)
                    progress_bar.update(1)
                    
            progress = 0
            progress += 1



            # Start processing the files in a separate thread
            thread = threading.Thread(target=process_files)
            thread.start()
            # Wait for the thread to finish
            thread.join()
            

            return redirect('appel-list')
    else:
        form = AppelForm()

    context = {
        'form': form,
        'progress': progress  # Add the progress value to the context
    }
    return render(request, 'store/create-appel.html', context)


class AppelListView(ListView):
    model = Appel
    template_name = 'store/appel-list.html'
    context_object_name = 'appels'

    def get_queryset(self):
        queryset = super().get_queryset()
        search_query = self.request.GET.get('search')
        if search_query:
            queryset = queryset.filter(Q(firstname__icontains=search_query))
        return queryset

def transcription_view(request, pk):
    try:
        appel = Appel.objects.get(pk=pk)
    except Appel.DoesNotExist:
        raise Http404("Appel does not exist")

    if appel.transcription is None:
        return render(request, 'store/transcription_view.html', {'error': 'Transcription not found.'})
    else:
        transcription = appel.transcription.replace('Position: Right', 'Customer').replace('Position: Left', appel.firstname)
        transcription_lines = transcription.split('\n')

        formatted_transcription = []
        for line in transcription_lines:
            stripped_line = line.strip()
            line_topics = set()
            if stripped_line.startswith("Transcription:"):
                content = stripped_line.replace("Transcription:", "").strip()
                words = content.split()
                for word in words:
                    for topic, details in TOPICS.items():
                        if word.lower() in [label.lower() for label in details["labels"]]:
                            line_topics.add(details["color"])
                header = None
            else:
                content = None
                header = stripped_line

            formatted_transcription.append({"header": header, "content": content, "colors": line_topics})

        topics = [{"topic": topic, "color": details["color"]} for topic, details in TOPICS.items()]  # Convert the TOPICS dictionary to a list of dictionaries

        return render(request, 'store/transcription_view.html', {'transcription_lines': formatted_transcription, 'appel': appel, 'topics': TOPICS})

def analyse_list_view(request):
    return render(request, 'store/analyse_list.html')



model_S = load_model('D:/WorkSpace_Django/pfe2023/model/modelBest.h5')

def predict_sentiment(sentence):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([sentence])
    sequence = tokenizer.texts_to_sequences([sentence])
    data = pad_sequences(sequence, maxlen=10)

    prediction = model_S.predict(np.array(data))
    labels = ['neutral', 'negative', 'positive']

    # Convert the softmax outputs to percentages
    sentiment_percentages = prediction[0] * 100

    return sentiment_percentages.tolist()

@login_required(login_url='login')
def create_sentiment(request):
    if request.method == 'POST':
        form = SentimentForm(request.POST, request.FILES)
        if form.is_valid():
            sentiment = form.save()  # Save the form
            sentiment.convert_audio()  # Convert the audio and update the converted_text field
            converted_text = sentiment.converted_text  # Get the converted text
            sentiment.save()
            

            # Tokenize the converted text into sentences
            tokenized_sentences = tokenize_sentences(converted_text)

            # Initialise a counter for each sentiment
            sentiment_counts = {"negative": 0, "positive": 0, "neutral": 0}

            # Analyze the sentiment of each individual sentence and update the sentiment_counts
            for sentence in tokenized_sentences:
                sentiment_percentages = predict_sentiment(sentence)
                # Get the index of the maximum sentiment percentage
                max_sentiment_index = np.argmax(sentiment_percentages)
                # Update the corresponding sentiment count
                if max_sentiment_index == 0:
                    sentiment_counts["negative"] += 1
                elif max_sentiment_index == 1:
                    sentiment_counts["neutral"] += 1
                else:
                    sentiment_counts["positive"] += 1

            total_count = sum(sentiment_counts.values())
            sentiment_percentages = {sentiment: round((count / total_count) * 100, 2) for sentiment, count in sentiment_counts.items()}

            # Create a list with sentiments and their percentages
            sentiments = list(sentiment_percentages.keys())
            percentages = list(sentiment_percentages.values())
            
            sentiment.pourcentage_negative = sentiment_percentages['negative']
            sentiment.pourcentage_neutral = sentiment_percentages['neutral']
            sentiment.pourcentage_positive = sentiment_percentages['positive']
            sentiment.save()
            
            # Create a bar chart using Plotly
            fig = go.Figure(data=[go.Bar(x=sentiments, y=percentages, marker_color=['red', 'green', 'blue'], text=percentages, textposition='auto')])

            fig.update_layout(
                autosize=False,
                width=500,
                height=500,
                paper_bgcolor='rgba(0,0,0,0)',  # This will make the plot's paper (margin around the plot) transparent
                plot_bgcolor='rgb(242,244,244)',  # This will set the plot's background to grey
                xaxis_title="Sentiment",
                yaxis_title="Pourcentage (%)",
            )

            # Create a base64 string of the plot
            buf = io.BytesIO()
            fig.write_image(buf, format='png')
            buf.seek(0)
            string = base64.b64encode(buf.read())
            uri = urllib.parse.quote(string)

            # Save the URI to the Sentiment instance
            sentiment.converted_plot = uri
            sentiment.save()

            # Store the converted text and the plot data in the session
            request.session['converted_text'] = converted_text
            request.session['data'] = uri

            return render(request, 'store/create-sentiment.html', {'form': form, 'converted_text': converted_text, 'data': uri, 'sentiment_percentages': sentiment_percentages})

    else:
        form = SentimentForm()
        # Get the converted_text and sentiment_percentages from session if they exist
        converted_text = request.session.get('converted_text', None)
        data = request.session.get('data', None)

    return render(request, 'store/create-sentiment.html', {'form': form, 'converted_text': converted_text, 'data': data})