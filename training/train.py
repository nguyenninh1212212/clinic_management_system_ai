import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from underthesea import word_tokenize
from unidecode import unidecode

def preprocess_text(text):
    text = unidecode(str(text).lower())
    return word_tokenize(text, format="text")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("1. Đang đọc dữ liệu...")
df = pd.read_csv(os.path.join(BASE_DIR, "dataset/training/training_data_clean.csv"))
df_unidecoded = df.copy()
df_unidecoded['text'] = df_unidecoded['text'].apply(lambda x: unidecode(str(x).lower()))
df_final = pd.concat([df, df_unidecoded], ignore_index=True)
df_final['processed_text'] = df_final['text'].apply(preprocess_text)
print("2. Đang xây dựng Model Pipeline...")
svm = LinearSVC(dual="auto")
clf = CalibratedClassifierCV(svm, cv=2) 
model = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', clf)
])

print("3. Đang huấn luyện...")
model.fit(df_final['processed_text'], df_final['label'])

print("4. Lưu mô hình (Export)...")
model_dir = os.path.join(BASE_DIR, "../models")
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, 'triage_svm_v4.pkl'))

print("Hoàn tất! Model đã được lưu tại: models/triage_svm_v4.pkl")