import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from underthesea import word_tokenize

def preprocess_text(text):
    return word_tokenize(str(text).lower(), format="text")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "dataset/symptom2diseaseT.csv")

print("1. Đang đọc dữ liệu...")
df = pd.read_csv(csv_path)
df['processed_text'] = df['text'].apply(preprocess_text)

print("2. Đang xây dựng Model Pipeline...")
svm = LinearSVC(dual="auto")
clf = CalibratedClassifierCV(svm, cv=2) 
model = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', clf)
])

print("3. Đang huấn luyện...")
model.fit(df['processed_text'], df['label'])

print("4. Lưu mô hình (Export)...")
# Đường dẫn lưu file .pkl cũng dùng đường dẫn tuyệt đối
model_dir = os.path.join(BASE_DIR, "../models")
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, 'triage_svm_v1.pkl'))

print("Hoàn tất! Model đã được lưu tại: models/triage_svm_v1.pkl")