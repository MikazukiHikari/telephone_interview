import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score


class Model:
    def __init__(self):
        self.SEED = 1012
        self.model_GBC_S = GradientBoostingClassifier(random_state=self.SEED,
                                                      learning_rate=0.1,
                                                      loss='deviance',
                                                      max_depth=4,
                                                      max_features=0.1,
                                                      min_samples_leaf=100,
                                                      n_estimators=200)

    def train(self, X: pd.DataFrame, y: pd.DataFrame):
        _X = X.drop('ID', axis=1)

        train_acc_list, valid_acc_list = [], []
        KF_5 = StratifiedKFold(
            n_splits=5, random_state=self.SEED, shuffle=True)
        KF_5.get_n_splits(_X, y)
        for train_index, test_index in KF_5.split(_X, y):

            trainX_split, trainY_split = _X.iloc[train_index], y.iloc[train_index]

            validX_split, validY_split = _X.iloc[test_index], y.iloc[test_index]

            __X, _y = SMOTE(random_state=42).fit_resample(
                trainX_split, trainY_split)
            self.model_GBC_S.fit(__X, _y)

            threshold = 0.5
            trainY_pred = (self.model_GBC_S.predict_proba(trainX_split)[
                           :, 1] > threshold).astype('float')
            train_acc = accuracy_score(trainY_split,         # 計算訓練資料準確度
                                       trainY_pred)

            high_priority_prob = self.model_GBC_S.predict_proba(validX_split)[:, 1]
            validY_pred = (high_priority_prob > threshold).astype('float')
            valid_acc = accuracy_score(validY_split,         # 計算驗證資料準確度
                                       validY_pred)

            train_acc_list.append(train_acc)
            valid_acc_list.append(valid_acc)

    def pred(self, pX: pd.DataFrame):
        _pX = pX.drop('ID', axis=1)

        result = self.model_GBC_S.predict_proba(_pX)
        result = pd.concat([pX.loc[:, 'ID'],
                            pd.DataFrame({'prob': result[:, 1]})], axis=1)
        return result
