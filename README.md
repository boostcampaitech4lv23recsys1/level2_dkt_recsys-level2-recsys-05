# Deep Knowledge Tracing
본 프로젝트는 i-scream 데이터셋을 이용하여 DKT 모델을 구축하는 것입니다. 다만 학생 개개인의 지식 상태를 예측하는 것보다는, 주어진 문제를 맞추었을지 틀렸을지 예측하는 모델을 만드는 것이 목표입니다. 평가 지표는 AUROC 입니다.

## 팀원 소개
|이름|역할|
|----|---|
|[강민수](https://github.com/minsu0216)|EDA, Feature Engineering, Sequetial Recommendation Modeling|
|[김진명](https://github.com/tobe-honest)|EDA, Feature Engineering, ML / LightGCN experiments|
|[박경태](https://github.com/GT0122)|EDA, Feature Engineering, ML stacking, Sequential Modeling|
|[박용욱](https://github.com/oceanofglitta)|EDA, Transformers Experiments|

## 데이터 구조
| 이름 | 설명 |
| --- | --- |
| **userID** | 사용자 고유 번호  |
| **assessmentItemID** | 문항의 고유번호 |
| **testID** |  시험지의 고유번호 |
| **answerCode** | 사용자가 해당 문항을 맞췄는지 여부 (binary) |
| **Timestamp** | 사용자가 해당문항을 풀기 시작한 시점 |
| **KnowledgeTag** | 문항 당 하나씩 배정되는 태그로, 일종의 중분류 역할 |

## 폴더 구조
```bash
level2_dkt_recsys-level2-recsys-05
├── EDA                     # EDA 코드 
├── Feature Engineering     # Feature Engineering 코드
└── Model                   # 수행한 모델 
    ├── MLstack             # ML stacking
    ├── dkt                 # LSTM, BERT, Last Query 등 sequence 계열 모델
    ├── lightgcn            # LightGCN
    ├── lightgcn_lstm       # LightGCN + LSTM
    └── optuna              # Catboost
```

## 수행 결과
- Weighted Ensemble : Catboost(0.3333) + LightGBMRegressor(0.1666) + LSTM Attention(0.0833) + LightGCN(0.2333) + Bert(0.1833)  
- AUROC : 0.8276(public, 3th) -> 0.8454(private, 8th)