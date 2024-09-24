# 🤖SKN03-3rd-1Team : 옵티마이조🤖

👾 팀 소개

|     유혜린     |    박지용    |        송명신         |    정해린    |   진윤화   |
| :------------: | :----------: | :-------------------: | :----------: | :--------: |
|   @Lerini98    |   @J-push    |    @SongMyungshin     |  @junghl11   | @JinYunhwa |
| Project Leader | Manager Page | Front-end & User page | Manager Page |  Modeling  |

⌛ 개발 기간

2024-09-23 ~ 2024-09-24 (총 2일)

## 📌 프로젝트 목표

- 장고를 이용한 화면 개발 (가능하면, 관리자 분석 페이지 개발)

- 가입 고객 이탈 예측과 관련된 화면 개발

- 가입 고객 이탈 예측 모델 개발 및 결과에 대한 설명

## 📌프로젝트 소개

이탈 징후가 보이는 고객군에 대한 맞춤형 제안 서비스 제공

> 이탈 가능성이 높은 고객에게 특정 요금제나 혜택을 제안하여 이탈을 줄이는 방안을 설계

## 📎데이터셋 정보

[통신사 고객 이탈 데이터셋](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- 지난 달에 떠난 고객 (Churn 컬럼)

- 각 고객이 가입한 서비스

  - 전화, 다중 회선, 인터넷, 온라인 보안, 온라인 백업, 장치 보호, 기술 지원, TV 및 영화 스트리밍

- 고객 계정 정보

  - 고객이 된 기간, 계약, 지불 방법, 무지불 청구, 월별 요금 및 총 요금

- 고객의 인구 통계 정보
  - 성별, 연령대, 파트너 및 부양가족이 있는지 여부

## Flowchart

![alt text](image.png)

## Model

윤화님 여기에 작성 부탁드려용

## Error Report

### 관리자 페이지

1. 부트스트랩에서 가져온 템플릿이라 잘못 건들이면 모든 것이 적용이 안되는 경우 존재
   > 기존 템플릿을 건들이지 않고 바꿀 부분만 바꿔야 제대로 적용됨
2. 그래프 시각화 할 때, 템플릿에 있는 것으로 사용하려 했으나 적용 안되는 부분 발생
   > lineplot으로 그린 후 이미지로 표현하는 방법으로 대체
3. 읽어온 이미지를 x버튼으로 창닫기 안됨
   > 차트를 모달로 띄운 것이 아닌 이미지 자체로 띄워서 x버튼 생성 안됨 실패함

### 사용자 페이지

1. DDL파일을 로드하지 못한 에러 발생(ImportError: DLL load failed while importing \_cext: )
   > Microsoft의 Visual C++ Redistributable 패키지를 설치하여 문제 해결

## 한줄평

🐶 유혜린 : 결국 모델 학습엔 손을 못댄게 아쉽네요

🐲 박지용 : 모델을 학습시키라더니 제 자신을 더 학습시킨 것 같아 좋은 경험이었습니다

🦥 송명신 : 결국 모델 학습엔 손을 못댄게 정말 아쉽네요

🐹 정해린 :

🦝 진윤화 :
