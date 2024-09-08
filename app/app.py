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
    scores_per_teacher = {col: {'neg': [], 'neu': [], 'pos': [], 'compound': []} for col in df.columns}
    
    for teacher in df.columns:
        teacher_results = []
        for criticism in df[teacher].dropna():
            criticism_en = translate_text(criticism, translator)
            scores = analyze_sentiments(criticism_en, analyzer)
            result = {
                'Compound': scores['compound'],
                'criticism': criticism,
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'positive': scores['pos']
            }
            teacher_results.append(result)
            scores_per_teacher[teacher]['neg'].append(scores['neg'])
            scores_per_teacher[teacher]['neu'].append(scores['neu'])
            scores_per_teacher[teacher]['pos'].append(scores['pos'])
            scores_per_teacher[teacher]['compound'].append(scores['compound'])
        
        average_scores = {
            'teacher': teacher,
            'compound_average': sum(scores_per_teacher[teacher]['compound']) / len(scores_per_teacher[teacher]['compound']) if scores_per_teacher[teacher]['compound'] else 0,
            'negative_average': sum(scores_per_teacher[teacher]['neg']) / len(scores_per_teacher[teacher]['neg']) if scores_per_teacher[teacher]['neg'] else 0,
            'neutral_average': sum(scores_per_teacher[teacher]['neu']) / len(scores_per_teacher[teacher]['neu']) if scores_per_teacher[teacher]['neu'] else 0,
            'positive_average': sum(scores_per_teacher[teacher]['pos']) / len(scores_per_teacher[teacher]['pos']) if scores_per_teacher[teacher]['pos'] else 0,
            'criticisms': teacher_results
        }
        results.append(average_scores)
    
    return results

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        df = pd.read_excel(file)
        analyzer, translator = initialize_objects()
        results = process_criticisms(df, analyzer, translator)
        
        return jsonify(results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
