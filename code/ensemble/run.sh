python ensemble.py --ENSEMBLE_FILES lightgcn_best,lstmattn_best_sigmoid,tabnet805,catboost_clf_ms_data,lgbmregressor_ms_data --RESULT_PATH ./output/
0.5 - 0.3,0.075,0.075,0.075,0.075,0.1,0.075,0.075,0.075,0.075
# 진명 - catboost_clf_ms_data(0.82),lgbmregressor_ms_data(0.81),lightgcn_best(0.79, 거의 0.8)
# 용욱 - tabnet805(0.80)
# 경태 - stack(0.80),catboost_pkt(0.80),lstmattn_best_sigmoid(0.78)
# 민수 - model_lstm_best2(0.80),model_lstmattn_best2(0.80),model_bert_best2(0.81, 0.80)

# 실험 1
# 동일 비율

# 실험 2
# boosting - 0.5
# graph - 0.2
# sequence - 0.3
# 0.2,0.075,0.075,0.075,0.075,0.2,0.075,0.075,0.075,0.075

# 실험 3
# lightgcn lstmattn_best_sigmoid tabnet805 catboost_clf_ms_data lgbmregressor_ms_data
# 


best_of_best-
catboost_clf_ms_data-lgbmregressor_ms_data-lightgcn_best-lstmattn_best_sigmoid
catboost_clf_ms_data-lightgcn_best-model_bert_best2-sw-0.5-0.2-0.3-
catboost_clf_ms_data-lgbmregressor_ms_data-lightgcn_best-model_bert_best2-aw-
aw

catboost_clf_ms_data - 1/3*0.25+1/3*0.5+0.25*1/3
lgbmregressor_ms_data - 1/3*0.25+1/3*0.25
lstmattn_best_sigmoid - 1/3*0.25
lightgcn_best - 1/3*0.25+1/3*0.2+0.25*1/3
model_bert_best2 - 1/3*0.3+0.25*1/3