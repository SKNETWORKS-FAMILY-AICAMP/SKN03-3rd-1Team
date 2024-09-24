import joblib
import numpy as np
import os

# 머신러닝 모델
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    VotingClassifier
)
from sklearn.svm import SVC

# 모델 평가 및 메트릭
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
# 모델 학습 및 튜닝
from sklearn.model_selection import train_test_split, cross_val_score

# 기타 클래스 호출
from src.datapreprocessor import DataPreprocessor
from src.modeltrainer import ModelTrainer
from src.modelevaluator import ModelEvaluator
from src.seedresetting import reset_seeds


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def make_pred(model):
  # 예측 값 생성
  y_pred = model.predict(X_test)
  # No, Yes를 0, 1으로 변경
  y_pred = [0 if label == 'No' else 1 for label in y_pred]
  return y_pred

# 모델 평가 함수
def evaluate_model(y_test, y_pred):
  accuracy = accuracy_score(y_test, y_pred)
  conf_matrix = confusion_matrix(y_test, y_pred)
  class_report = classification_report(y_test, y_pred)

  print(f'Accuracy: {accuracy}')
  print('Confusion Matrix:')
  print(conf_matrix)
  print('Classification Report:')
  print(class_report)

  return

# 여러 모델 평가 함수
def evaluate_models(models, X_test, y_test):
    results = []

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
              
        # No, Yes를 0, 1으로 변경
        y_pred = [0 if label == 'No' else 1 for label in y_pred]
        
        # 성능 지표 계산
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', pos_label=1)
        recall = recall_score(y_test, y_pred, average='binary', pos_label=1)
        f1 = f1_score(y_test, y_pred, average='binary', pos_label=1)
        
        # 결과 저장
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
    return pd.DataFrame(results)



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

    y_test = [1 if label == 'Yes' else 0 for label in y_test]


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

    y_pred = make_pred(model_rf)
    evaluate_model(y_test, y_pred)
    
    # AdaBoost
    params = {
        'algorithm':'SAMME.R',
        'estimator':None,
        'learning_rate':1.0,
        'n_estimators':50,
        'random_state':42
    }
    model_adb = AdaBoostClassifier(**params).fit(X_train,y_train)

    y_pred = make_pred(model_adb)
    evaluate_model(y_test, y_pred)

    # 가우시안 NB
    model_gnb = GaussianNB(priors=None, var_smoothing=1e-09)
    model_gnb.fit(X_train, y_train)

    y_pred = make_pred(model_gnb)
    evaluate_model(y_test, y_pred)

    # 모델 : SVM
    model_svm = SVC(probability=True, random_state=42)  # SVM 모델 추가
    model_svm.fit(X_train, y_train)
    y_pred = make_pred(model_svm)
    evaluate_model(y_test, y_pred)

    # 모델 : Voting
    # VotingClassifier 초기화
    model_vt = VotingClassifier(
        estimators=[
        ('rf', model_rf),
        ('gnb', model_gnb),
        ('svm', model_svm),
        ('adb', model_adb)
        ],
        voting='soft',
        weights=[1,5,1,2]
    )  # 'hard' 또는 'soft' 투표 방식 선택 가능
    model_vt.fit(X_train, y_train)

    y_pred = make_pred(model_vt)
    evaluate_model(y_test, y_pred)

    # 예측 (확률값)
    y_pred_proba = model_gnb.predict_proba(X_test)[:, 1]  # 이탈 확률
    y_pred_proba = y_pred_proba.round(4)


    # 모델 종합 평가
    models = {
        'model_rf': model_rf,
        'model_adb': model_adb,
        'model_gnb': model_gnb,
        'model_svm': model_svm,
        'model_vt': model_vt,
    }
    results_df = evaluate_models(models, X_test, y_test)
    print(results_df)
    
    # 테스트 데이터 로드
    test_df = pd.read_csv(test_df_path)
    test_df['Churn_Probability'] = y_pred_proba

    # 결합된 데이터 저장
    test_df.to_csv(test_pred_path, index=False)

    print("결합된 데이터가 성공적으로 저장되었습니다.")

    # 3. 모델저장
    joblib.dump(model_gnb, model_save_path)

    print("완료")