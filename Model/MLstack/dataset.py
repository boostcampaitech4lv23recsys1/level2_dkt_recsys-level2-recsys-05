import pandas as pd

from sklearn.model_selection import train_test_split

def load_data(args) :
    train = pd.read_csv(args.file_name)
    test = pd.read_csv(args.test_file_name)

    data = pd.concat([train, test])
    test = data[data.answerCode == -1]
    data = data[data.answerCode >= 0]

    Item2Vec = {v:k for k, v in enumerate(sorted(data.assessmentItemID.unique()))}
    test2Vec = {v:k for k, v in enumerate(sorted(data.testId.unique()))}
    tag2Vec = {v:k for k, v in enumerate(sorted(data.KnowledgeTag.unique()))}

    data['assessmentItemID'] = data['assessmentItemID'].apply(lambda x : Item2Vec[x])
    test['assessmentItemID'] = test['assessmentItemID'].apply(lambda x : Item2Vec[x])
    data['testId'] = data['testId'].apply(lambda x : test2Vec[x])
    test['testId'] = test['testId'].apply(lambda x : test2Vec[x])
    data['KnowledgeTag'] = data['KnowledgeTag'].apply(lambda x : tag2Vec[x])
    test['KnowledgeTag'] = test['KnowledgeTag'].apply(lambda x : tag2Vec[x])

    columns = data.columns
    columns = columns.drop('answerCode')
    columns = columns.drop('Timestamp')
    X, y = data[data.answerCode >= 0][columns], data[data.answerCode >= 0]['answerCode']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    return X_train, X_test, y_train, y_test, test[columns], columns