import joblib
import numpy as np
import os

# 머신러닝 모델
from sklearn.ensemble import (
    RandomForestClassifier,
)

# 모델 평가 및 메트릭
from sklearn.metrics import (
    roc_auc_score,
)

# 모델 학습 및 튜닝
from sklearn.model_selection import train_test_split, cross_val_score

# 기타 클래스 호출
from src.datapreprocessor import DataPreprocessor
from src.modeltrainer import ModelTrainer
from src.modelevaluator import ModelEvaluator
from src.visualization import Visualization
from src.seedresetting import reset_seeds


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def get_auc(model, x_train, X_test):
    pred_probs = model.predict_proba(x_train)[:, 1]
    auc_train = roc_auc_score(y_train, pred_probs)

    pred_probs = model.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, pred_probs)

    print(f"train AUC: {auc_train:.4f}, test AUC: {auc_test:.4f}")

    return auc_train, auc_test

if __name__=="__main__":
    print("메인 시작")
    reset_seeds()

    # 현재 파일의 디렉토리 경로
    current_dir = os.path.dirname(__file__)
    
    # 0. 경로 및 변수 설정
    data_path = os.path.join(current_dir, 'data', 'data.csv')
    test_df_path = os.path.join(current_dir, 'data', 'test.csv')
    model_save_path = os.path.join(current_dir, 'models', 'customer_churn_model.pkl')
    test_pred_path = os.path.join(current_dir, 'data', 'test_pred.csv')

    # 1. 데이터 로드 및 전처리
    X_train, X_test, y_train, y_test = DataPreprocessor().preprocess_data(data_path, target_column="Churn")

    # 2. 모델 학습
    params = {
        'random_state':42,
        'max_depth': 8,
        'max_features': 'sqrt',
        'min_samples_leaf': 4,
        'min_samples_split': 8,
        'n_estimators': 192
    }
    
    model_rf = RandomForestClassifier(**params,).fit(X_train, y_train)

    auc_all = get_auc(model_rf,X_train, X_test)

    auc_train = auc_all[0]
    auc_test = auc_all[1]
    
    ModelTrainer().test()

    print("가우시안 NB 사용")
    
    # 가우시안 NB사용

    # Gaussian Naive Bayes 모델 초기화
    model_gnb = GaussianNB(priors=None, var_smoothing=1e-09)

    # 모델 학습
    model_gnb.fit(X_train, y_train)

    # 예측
    y_pred = model_gnb.predict(X_test)

    # 모델 평가
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)

    # 예측 (확률값)
    y_pred_proba = model_gnb.predict_proba(X_test)[:, 1]  # 이탈 확률
    y_pred_proba = y_pred_proba.round(4)

    # 테스트 데이터 로드
    test_df = pd.read_csv(test_df_path)
    test_df['Churn_Probability'] = y_pred_proba

    # 결합된 데이터 저장
    test_df.to_csv(test_pred_path, index=False)

    print("결합된 데이터가 성공적으로 저장되었습니다.")

    # 3. 모델저장
    joblib.dump(model_gnb, model_save_path)

    print("완료")