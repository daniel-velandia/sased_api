from flask import Flask, request, jsonify
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from googletrans import Translator

app = Flask(__name__)

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
        'compound_average': sum([s['compound'] for s in scores]) / len(scores) if scores else 0,
        'neg_average': sum([s['neg'] for s in scores]) / len(scores) if scores else 0,
        'neu_average': sum([s['neu'] for s in scores]) / len(scores) if scores else 0,
        'pos_average': sum([s['pos'] for s in scores]) / len(scores) if scores else 0
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
                'criticism': criticism,
                'neg': scores['neg'],
                'neu': scores['neu'],
                'pos': scores['pos']
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
                return jsonify(results)
            except Exception as e:
                return jsonify({'error': f'Error processing the file: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
