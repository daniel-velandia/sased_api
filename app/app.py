from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from googletrans import Translator
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)

CORS(app)

# Configuración de MongoDB
client = MongoClient("mongodb://admin:admin_password@mongo:27017/")
db = client["analisis_sentimientos"]
collection = db["resultados"]

def initialize_objects():
    analyzer = SentimentIntensityAnalyzer()
    translator = Translator()
    return analyzer, translator

def translate_text(text, translator):
    return translator.translate(text, src='es', dest='en').text

def analyze_sentiments(text, analyzer):
    return analyzer.polarity_scores(text)

def calculate_averages(scores):
    return {
        'compound_average': sum([s['compound'] for s in scores]) / len(scores) if scores else 0
    }

def process_criticisms(df, analyzer, translator):
    results = []
    teacher_scores = {}
    
    grouped = df.groupby(['profesor', 'materia'])
    
    for (teacher, subject), group in grouped:
        subject_results = []

        if teacher not in teacher_scores:
            teacher_scores[teacher] = {'scores': []}

        for criticism in group['crítica'].dropna():
            criticism_en = translate_text(criticism, translator)
            scores = analyze_sentiments(criticism_en, analyzer)

            subject_results.append({
                'compound': scores['compound'],
                'criticism': criticism
            })
            teacher_scores[teacher]['scores'].append(scores)

        subject_average = calculate_averages(subject_results)
        subject_average.update({'subject': subject, 'criticisms': subject_results})

        results.append({'teacher': teacher, 'subject': subject_average})

    final_results = []
    for teacher, data in teacher_scores.items():
        teacher_average = calculate_averages(data['scores'])
        teacher_average.update({
            'teacher': teacher,
            'subjects': [r['subject'] for r in results if r['teacher'] == teacher]
        })
        final_results.append(teacher_average)

    return final_results

def get_current_semester():
    month = datetime.now().month
    year = datetime.now().year
    semester = 1 if month <= 6 else 2
    return f"{year}-S{semester}"

def get_previous_semester():
    month = datetime.now().month
    year = datetime.now().year
    if month <= 6:
        return f"{year - 1}-S2"
    else:
        return f"{year}-S1"

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if file:
            try:
                df = pd.read_excel(file)
                analyzer, translator = initialize_objects()
                results = process_criticisms(df, analyzer, translator)
                
                current_semester = get_current_semester()
                collection.replace_one(
                    {"semester": current_semester},
                    {
                        "semester": current_semester,
                        "date": datetime.now(),
                        "results": results
                    },
                    upsert=True
                )
                
                previous_semester = get_previous_semester()
                previous_analysis = collection.find_one({"semester": previous_semester})
                
                return jsonify({
                    'current_semester_analysis': results,
                    'previous_semester_analysis': previous_analysis['results'] if previous_analysis else []
                })
            except Exception as e:
                return jsonify({'error': f'Error processing the file: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
