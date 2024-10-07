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
    try:
        return translator.translate(text, src='es', dest='en').text
    except Exception as e:
        print(f"Error al traducir: {e}")
        return text

def analyze_sentiments(text, analyzer):
    return analyzer.polarity_scores(text)

def process_criticisms(df, analyzer, translator):
    results = []
    grouped = df.groupby(['profesor', 'materia'])
    teacher_scores = {}

    for (teacher, subject), group in grouped:
        subject_results = []

        if teacher not in teacher_scores:
            teacher_scores[teacher] = {'neg': [], 'neu': [], 'pos': [], 'compound': []}

        for criticism in group['crítica'].dropna():
            criticism_en = translate_text(criticism, translator)
            scores = analyze_sentiments(criticism_en, analyzer)
            result = {
                'Compound': scores['compound'],
                'criticism': criticism,
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'positive': scores['pos']
            }
            subject_results.append(result)
            teacher_scores[teacher]['neg'].append(scores['neg'])
            teacher_scores[teacher]['neu'].append(scores['neu'])
            teacher_scores[teacher]['pos'].append(scores['pos'])
            teacher_scores[teacher]['compound'].append(scores['compound'])

        subject_average = {
            'subject': subject,
            'compound_average': sum([r['Compound'] for r in subject_results]) / len(subject_results) if subject_results else 0,
            'criticisms': subject_results
        }
        
        results.append({'teacher': teacher, 'subject': subject_average})

    final_results = []
    for teacher, scores in teacher_scores.items():
        average_scores = {
            'teacher': teacher,
            'compound_average': sum(scores['compound']) / len(scores['compound']) if scores['compound'] else 0,
            'negative_average': sum(scores['neg']) / len(scores['neg']) if scores['neg'] else 0,
            'neutral_average': sum(scores['neu']) / len(scores['neu']) if scores['neu'] else 0,
            'positive_average': sum(scores['pos']) / len(scores['pos']) if scores['pos'] else 0,
            'subjects': [r['subject'] for r in results if r['teacher'] == teacher]
        }
        final_results.append(average_scores)

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
                print(f"Error al leer el archivo o procesar datos: {e}")
                return jsonify({'error': f'Error processing the file: {str(e)}'}), 500
    except Exception as e:
        print(f"Error en la solicitud: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
