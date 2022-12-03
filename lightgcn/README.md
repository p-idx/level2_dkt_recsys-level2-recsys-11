# 동작환경

- cuda verion ==11.3
- torch==1.10.0
- torch-scatter==2.0.9
- torch-sparse==0.6.13
- torch-geometric==2.0.4

라이브러리 설치

```
pip uninstall torch, torch-sparse, torch-scatter, torch-geometric
pip3 install torch==1.10.0
pip install torch-scatter==2.0.9
pip install torch-sparse==0.6.13
pip install torch-geometric==2.0.4
```


# 파일

- install.sh : 관련 library 설치 스크립트 파일
- config.py : 설정 파일
- lightgcn/datasets.py : 데이터 로드 및 전처리 함수 정의
- lightgcn/model.py : 모델을 정의하고 manipulation 하는 build, train, inference관련 코어 로직 정의
- lightgcn/utils.py : 부가 기능 함수 정의
- train.py : 시나리오에 따라 데이터를 불러 모델을 학습하는 스크립트
- inference.py : 시나리오에 따라 학습된 모델을 불러 테스트 데이터의 추론값을 계산하는 스크립트
- evaluation.py : 저장된 추론값을 평가하는 스크립트


# 사용 시나리오

- install.sh 실행 : 라이브러리 설치(기존 라이브러리 제거 후 설치함)
- config.py 수정 : 데이터 파일/출력 파일 경로 설정 등
- train.py 실행 : 데이터 학습 수행 및 모델 저장
- inference.py 실행 : 저장된 모델 로드 및 테스트 데이터 추론 수행


# 12.03 변경 사항

- config.py 가서 사용할 FE폴더의 숫자를 수정해서 사용 
    - 데이터는 반드시 userID, assessmentItemID를 포함하고 있어야 함!
    - datasets.indexing_data, preprocess_data 에서 int로 label encoding된 값들은 str으로 변환해주어야 함
- validation(n=7440)이 datasets.data_separate() 에서 생성되게 수정
    - 베이스라인에서는 merged_train에서 랜덤하게 1000개 사용
