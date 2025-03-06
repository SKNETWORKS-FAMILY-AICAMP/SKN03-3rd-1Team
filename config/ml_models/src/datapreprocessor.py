import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

class DataPreprocessor:
    def __init__(self, test_size = 0.2, random_state = 42) -> None:
        """
        DataPreprocessor 초기화
        """
        self.test_size = test_size
        self.random_state = random_state
    
    def load_data(self, filepath):
        try:
            data = pd.read_csv(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"파일을 {filepath} 경로에서 찾을 수 없습니다. ")
        except pd.errors.EmptyDataError:
            raise ValueError("파일이 비어있습니다.")
        except pd.errors.ParserError:
            raise ValueError("파일을 파싱하는데 에러가 발생하였습니다.")

        print("데이터 로드 완료")

        return data
    
    def drop_columns(self, data, columns_to_drop):
        data = data.drop(data.filter(regex='|'.join(columns_to_drop)).columns, axis=1)
        return data

    def encode_data(self, data):
        # 데이터 인코딩을 위해 범주형 컬럼에 데이터 할당(단, target인 Churn은 제외)
        columns = ['gender','SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges',
       'TotalCharges']
        numerical_columns = ['tenure', 'MonthlyCharges','TotalCharges']
        categorical_colums = list(set(columns)-set(numerical_columns))

        # categorical_colums에 원핫 인코딩 적용
        data = pd.get_dummies(data, columns=categorical_colums)
        
        # '_No', '_0', '_Female' 가 포함된 컬럼 삭제
        columns_to_drop = ['_No', '_0', '_Female']
        # 인코딩 이후 컬럼들 드랍
        data = self.drop_columns(data, columns_to_drop)
        return data
    
    def fill_missing_values(self, data):
        # 공백 문자열을 NaN으로 변환 후, 0으로 채워줌.
        data = data.replace(' ', pd.NA)
        data['TotalCharges'].fillna(0, inplace=True)
        return data
    
    def split_data(self, data, target_column):
        X = data.drop(columns=[target_column])
        y = data[target_column]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        return X_train, X_val, y_train, y_val 

    def preprocess_data(self, data_path,target_column):
        # X_train, X_val, y_train, y_val = self.load_data(data_path, target_column)
        # 1. 데이터 로드
        data = self.load_data(data_path)

        # 2. 데이터 전처리
        # 2-1.drop customerID
        columns_to_drop = ['customerID']
        data = self.drop_columns(data, columns_to_drop)
        # 2-2.데이터 인코딩
        data = self.encode_data(data)
        data.info()
        # 2-3.컬럼명의 공백(' ')을 언더바('_')으로 변환
        data.columns = data.columns.str.replace(' ', '_')
        # 2-4.결측치 채우기
        data = self.fill_missing_values(data)
        # 2-5.'TotalCharges' 컬럼을 float으로 변환
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

        # QuantileTransformer 객체 생성
        scaler = QuantileTransformer(output_distribution='normal')

        # 수치형 데이터 스케일링
        data[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(data[['tenure', 'MonthlyCharges', 'TotalCharges']])

        # 3. 데이터 스플릿
        X_train, X_val, y_train, y_val = self.split_data(data, target_column)

        return X_train, X_val, y_train, y_val