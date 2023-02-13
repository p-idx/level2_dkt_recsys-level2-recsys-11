# Deep Knowledge Tracing
<img src="https://user-images.githubusercontent.com/55279227/218374419-3382bc6e-ae7f-46ae-af04-e20b88764544.JPG" width="800" height="500"/>

### **1-1. 프로젝트 소개**

- **Deep Knowledge Tracing**(이하 DKT)는 AI 모델을 활용하여 Knowledge Tracing을 하는 방법론이다. Knowledge Tracing은 유저의 지식 상태의 추이를 파악하여 다음 교육에 대한 추천 및 예측을 수행하는 task를 의미한다. 이때 딥러닝 기법을 활용해 보다 정확한 예측을 수행하고자 하는 것이 DKT이다. 과거 유저가 문제를 푼 데이터를 통해 개별 유저의 이해도와 수준을 학습하고, 이후 최적화된 문제를 추천함으로써 개인 맞춤화된 학습을 도울 수 있어 응용 기대효과가 높다.
    - 본 대회는 유저의 문제 정답 여부가 들어간 데이터를 학습하여 다음에 주어질 문제에 대한 정오답을 예측하는 이진분류 문제다. 이에 본 프로젝트에서는 RNN, Transformer, Boosting 및 GNN 기법을 활용하여 정오답 예측을 수행하였다.


### **1-2. 데이터 요약**

- I-scream 학생 교육 과정 데이터셋을 사용하며 파일은 csv 형태로 제공된다. 주어지는 정보는 다음과 같다.
    - userID : 유저의 고유번호
    - assessmentItemID : 문항의 고유번호
    - testId : 시험지의 고유번호
    - answerCode : 사용자의 정답 여부
    - Timestamp : 사용자가 문제를 풀기 시작한 시점
    - KnowledgeTag : 문항 분류 태그
    - Train_data와 test_data는 총 7,442명의 유저를 기준으로 90:10 비율로 나뉘어 744명의 사용자가 test_data에 속한다. 이 744명이 시간 상 마지막으로 푸는 문항의 answerCode는 -1 값이 삽입되어 있으며 해당 행에 대한 예측을 수행해야 한다.
- 총 데이터 행의 개수는 2,526,700개이고, Timestamp를 기준으로 정렬되어 있다.

### **1-3. 대회 평가 지표**

- 예측에 대한 평가는 AUC와 ACC 지표를 통해 이뤄진다. AUC를 우선으로 삼으며 AUC 점수가 같을 경우 ACC를 통해 순위를 매긴다.
    ![Untitled](https://user-images.githubusercontent.com/55279227/218373814-5cfadd55-8d48-459b-b4a0-d0f157c188b6.png)

    
- AUC(Area Under The ROC Curve)는 다양한 threshold에서 모델의 분류 성능을 나타내는 지표이다. 비율에 대한 값이기에 0부터 1 사이의 값을 가지며, threshold를 특정하지 않고 모델이 얼마나 잘 예측을 했는지 평가할 수 있다는 특징을 가진다.
- ACC(Accuracy)는 모델의 예측값과 실제값을 비교하여 모델이 옳게 예측한 비율을 나타내는 지표이다. Threshold를 0.5로 잡아 정오답을 판단한다.

### **1-4. 장비 정보**

- GPU: Tesla v100
- RAM: 88GB

### **1-5.** **개발환경**

- <img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white"><img src="https://img.shields.io/badge/vsc-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white"><img src="https://img.shields.io/badge/anaconda-44A833?style=for-the-badge&logo=anaconda&logoColor=white">
- <img src="https://img.shields.io/badge/pytorchlightning-792EE5?style=for-the-badge&logo=pytorchlightning&logoColor=white"><img src="https://img.shields.io/badge/w&b-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=white">


## 2. 프로젝트 기획

### Project Workflow
<img src="https://user-images.githubusercontent.com/55279227/218373655-da1cd584-1d72-458a-9523-d7e201f3ab24.jpg" width="750" height="400"/>

### 최종 AUC 스코어
<img src="https://user-images.githubusercontent.com/55279227/218373739-bbfb4e65-a1bf-4cc7-97a4-12b0584315cd.png" width="500" height="400"/>

- Riiid Kaggle 대회 SoTA 모델인 LastQuery 가 가장 높은 성능을 내었다.
<img src="https://user-images.githubusercontent.com/55279227/218374163-443a15bf-1e6d-4840-9dcd-609fd6369247.jpg" width="500" height="400"/>

## 3. 프로젝트 팀 구성 및 역할

- 양성훈: 프로젝트 계획 수립, Baseline 재구축, DKT 모델 구축
- 강수헌: EDA + feature engineering, CV 전략 검증, Catboost 성능 개선
- 김동건: EDA + feature engineering, 부스팅 모델 설계 및 실험
- 유상준: EDA + feature engineering, Catboost(+optuna), valid set 구축 전략
- 백승렬: EDA + feature engineering, Catboost 및 GCN 모델 튜닝, 설계 및 실험
