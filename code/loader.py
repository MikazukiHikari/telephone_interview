import pandas as pd
import numpy as np
import os

class Loader:
    def read_record(self, exam, eco, tele_history, tele_pred):
        """ 
            Read csv files, if there's any missing, report it and exit process.
        """
        if not (exam and tele_history and tele_pred and eco):
            return -1
        if exam:
            self.exam = pd.read_csv(exam)
            new_year = self.exam.sort_values('篩檢日期', ascending=False)['篩檢日期'].head(1).values[0]
            # old_year = self.exam.sort_values('篩檢日期', ascending=False)['篩檢日期'].tail(1).values[0]
            self.latest_year = int(str(new_year)[0:3])
            # self.oldest_year = int(str(old_year)[0:3])
            # self.total_year = self.latest_year - self.oldest_year + 1
        if eco:
            self.eco = pd.read_csv(eco)
        if tele_history:
            self.tele_history = pd.read_csv(tele_history)
            new_year = self.tele_history.sort_values('trace_time', ascending=False)['trace_time'].head(1).values[0]
            old_year = self.tele_history.sort_values('trace_time', ascending=False)['trace_time'].tail(1).values[0]
            self.tel_total_year = int(str(new_year)[0:3]) - int(str(old_year)[0:3]) + 1
        if tele_pred:
            self.tele_pred = pd.read_csv(tele_pred)
            self.tele_pred_orgin = self.tele_pred.copy()
        
        return 0

    def preprocessing(self):
        """
            Regularize data, fill NaN with -1:
            - column: use list to record last four years column name(ex.[111, 110, 109, 108])
            - record: drop incorrect data, merge `eco & edu` into list
            - history: separate `invite_items`, calculate examination times according 
                    to its `trace_time`, pad missing area by address in record
            - tele: separate `invite_items`, pad missing area by address in record
        """
        # year_fobtDate = []
        # year_edu = []
        # year_symp = []
        # for i in range(4):
        #     year = int(self.last) - i
        #     year_fobtDate.append('ccs_' + str(year) + '_fobtDate')
        #     year_edu.append('ccs_' + str(year) + '_edu')
        #     year_symp.append('ccs_' + str(year) + '_symp') 
        
        self.exam = self.exam[['ID', '年齡', '現住行政區', '篩檢日期']]
                            #    year_fobtDate[3], year_edu[3], year_symp[3],
                            #    year_fobtDate[2], year_edu[2], year_symp[2],
                            #    year_fobtDate[1], year_edu[1], year_symp[1],
                            #    year_fobtDate[0], year_edu[0], year_symp[0]]]
        """preserve only '區'."""
        self.exam['現住地行政區'] = self.exam['現住地行政區'].str.extract("(?:台南市|臺南市)(.+區)")[0]
        self.exam['examin_year'] = self.exam['篩檢日期'].astype(str).str[:3]
        # self.exam[[year_edu[3], year_symp[3], year_edu[2], year_symp[2], year_edu[1], year_symp[1], year_edu[0], year_symp[0]]] = self.exam[[
        #     year_edu[3], year_symp[3],
        #     year_edu[2], year_symp[2],
        #     year_edu[1], year_symp[1],
        #     year_edu[0], year_symp[0]]].apply(pd.to_numeric, errors='coerce')
        # self.exam.fillna(-1, inplace=True)
        # self.exam[[year_edu[3], year_symp[3], year_edu[2], year_symp[2], year_edu[1], year_symp[1], year_edu[0], year_symp[0]]] = self.exam[[
        #     year_edu[3], year_symp[3],
        #     year_edu[2], year_symp[2],
        #     year_edu[1], year_symp[1],
        #     year_edu[0], year_symp[0]]].astype(int)

        """ overwrite incorrect edu code to -1 """

        # self.exam.loc[self.exam[year_edu[3]] > 7, year_edu[3]] = -1
        # self.exam.loc[self.exam[year_edu[2]] > 7, year_edu[2]] = -1
        # self.exam.loc[self.exam[year_edu[1]] > 7, year_edu[1]] = -1
        # self.exam.loc[self.exam[year_edu[0]] > 7, year_edu[0]] = -1

        """ take valid state only """

        self.tele_history = self.tele_history[[
            'ID', 'area', 'invite_items', 'trace_time', 'state']]
        self.tele_history.fillna(-1, inplace=True)
        self.tele_history.loc[:, 'state'] = self.tele_history[['state']].apply(
            pd.to_numeric, errors='coerce')
        self.tele_history.dropna(inplace=True)
        self.tele_history.loc[:, 'state'] = self.tele_history[[
            'state']].astype(int)

        """ extract invite_items, drop !3 && !4 (3, 4 for mam)"""

        self.tele_history.loc[:, 'invite_items'] = self.tele_history[[
            'invite_items']].astype(str)
        self.tele_history.loc[self.tele_history['invite_items'].str.contains(
            '3'), 'invite_items'] = '3'
        self.tele_history.loc[self.tele_history['invite_items'].str.contains(
            '4'), 'invite_items'] = '4'
        self.tele_history.loc[:, 'invite_items'] = self.tele_history[[
            'invite_items']].apply(pd.to_numeric, errors='coerce')
        self.tele_history[['invite_items']] = self.tele_history[[
            'invite_items']].astype(int)
        self.tele_history = self.tele_history.drop(self.tele_history[(
            self.tele_history['invite_items'] != 3) & (self.tele_history['invite_items'] != 4)].index)

        """ fill area according to exam['現住行政區'] """

        df_merge = pd.merge(
            self.tele_history.loc[self.tele_history['area'] == -1, ['ID', 'area']], self.exam[['ID', '現住行政區']], on='ID', how='left')

        self.tele_history.loc[self.tele_history['area'] == -1,
                              'area'] = df_merge['現住行政區']
        self.tele_history.dropna(inplace=True)

        """ concat trace_time to year """

        self.tele_history[['trace_time']].astype(str)
        self.tele_history.loc[self.tele_history['trace_time'].str.len() > 5, 'trace_time'] = [
            i[0:3] for i in self.tele_history.loc[self.tele_history['trace_time'].str.len() > 5, 'trace_time']]
        self.tele_history.loc[:, 'trace_time'] = self.tele_history[[
            'trace_time']].apply(pd.to_numeric, errors='coerce')
        self.tele_history.dropna(inplace=True)
        self.tele_history.loc[:, 'trace_time'] = self.tele_history[[
            'trace_time']].astype(int)
        
        """ drop some error record. (ex. 1.10E+11) """
        self.tele_history = self.tele_history.drop(
            self.tele_history[(self.tele_history['trace_time'] < 100)].index)
        # self.tele_history = self.tele_history.drop(
        #     self.tele_history[(self.tele_history['trace_time'] < 109) | (self.tele_history['trace_time'] > 111)].index)

        """ groupby ['ID', 'trace_time'] """
        key = ['ID', 'trace_time']
        tmp_df = []
        """ Get max 'state' """
        tmp = self.tele_history.groupby(key)['state'].agg('max')
        tmp.name = 'state'
        tmp_df.append(tmp)
        """ Get max 'invite_items' """
        tmp = self.tele_history.groupby(key)['invite_items'].agg('max')
        tmp.name = 'invite_items'
        tmp_df.append(tmp)
        """ Get last 'area' """
        tmp = self.tele_history.groupby(key)['area'].last()
        tmp.name = 'tel_area'
        tmp_df.append(tmp)
        """ concat then reset_index """
        self.tele_history = pd.concat(tmp_df, axis=1).reset_index()

        """ append edu & eco """

        self.eco.fillna(-1, inplace=True)
        self.eco[['edu', 'eco']] = self.eco[[
            'EDUCATION', 'ECONMIC']].astype(int)
        self.eco = self.eco.drop(['EDUCATION', 'ECONMIC'], axis=1)

        df_merge = pd.merge(self.exam, self.eco, on='ID', how='left')
        df_merge.fillna(-1, inplace=True)
        df_merge[['edu', 'eco']] = df_merge[['edu', 'eco']].astype(int)
        df_merge.fillna(-1, inplace=True)
        self.exam = df_merge
        
        # self.exam['edu'] = self.exam[[year_edu[3], year_edu[2],
        #                               year_edu[1], year_edu[0], 'edu']].max(axis=1)
        # self.exam['symp'] = self.exam[[year_symp[3],
        #                                year_symp[2], year_symp[1], year_symp[0]]].max(axis=1)
        # self.exam = self.exam.drop([
        #     year_edu[3], year_edu[2], year_edu[1], year_edu[0]
        # ], axis=1)

        # """ set dtype = category """

        # self.exam['eco'] = pd.Categorical(
        #     self.exam['eco'], categories=[-1, 0, 1, 2, 3])
        # self.exam['edu'] = pd.Categorical(
        #     self.exam['edu'], categories=[-1, 1, 2, 3, 4, 5, 6, 7])
        # self.exam['symp'] = pd.Categorical(
        #     self.exam['symp'], categories=[-1, 0, 1])

        self.tele_pred = self.tele_pred[['ID', '未篩含首篩名單']]
        # self.tele_pred = self.tele_pred[['ID', 'area', 'invite_items']]
        # self.tele_pred.fillna(-1, inplace=True)

        # """ extract invite_items, and mark with -1 if it's not ccs """

        # self.tele_pred['invite_items'] = self.tele_pred['invite_items'].astype(
        #     str)
        # self.tele_pred.loc[self.tele_pred['invite_items'].str.contains(
        #     '1'), 'invite_items'] = '1'
        # self.tele_pred.loc[self.tele_pred['invite_items'].str.contains(
        #     '2'), 'invite_items'] = '2'
        # self.tele_pred[['invite_items']] = self.tele_pred[[
        #     'invite_items']].apply(pd.to_numeric, errors='coerce')
        # self.tele_pred[['invite_items']] = self.tele_pred[[
        #     'invite_items']].astype(int)
        # self.tele_pred.loc[(self.tele_pred['invite_items'] != 1) & (
        #     self.tele_pred['invite_items'] != 2), 'invite_items'] = -1

        # """ fill area according to exam['addr'] """

        # df_merge = pd.merge(
        #     self.tele_pred.loc[self.tele_pred['area'] == -1, ['ID', 'area']], self.exam[['ID', 'addr']], on='ID', how='left')
        # self.tele_pred.loc[self.tele_pred['area'] == -1,
        #                    'area'] = df_merge['addr'].str.extract("(?:台南市|臺南市)(.+區)")[0]

    def to_training_set(self):
        df_merge = pd.merge(self.exam, self.tele_history, on='ID')
        # set 'key' ('ID', 'trace_time')
        key = ['ID', 'trace_time']
        tmp_df = []
        # Set columns from self.exam.
        """Get max 'age'."""
        tmp = df_merge.groupby(key)['年齡'].agg('max')
        tmp.name = 'age'
        tmp_df.append(tmp)
        """Get latest '區'."""
        tmp = df_merge.sort_values('篩檢日期', ascending=True).groupby(key)['現住地行政區'].last()
        tmp.name = '現住地行政區'
        tmp_df.append(tmp)
        """Get the number of examining times"""
        tmp = df_merge.groupby(key)['篩檢日期'].agg('count')
        tmp.name = 'examining_times'
        tmp_df.append(tmp)
        """Get latest symptom'"""
        tmp = df_merge.sort_values('篩檢日期', ascending=True).groupby(key)['篩檢結果'].last()
        tmp.name = 'latest_symp'
        tmp_df.append(tmp)
        """Get latest 'edu'"""
        tmp = df_merge.groupby(key)['edu'].last()
        tmp.name = 'edu'
        tmp_df.append(tmp)
        """Get latest 'eco'"""
        tmp = df_merge.groupby(key)['eco'].last()
        tmp.name = 'eco'
        tmp_df.append(tmp)

        # Set columns from self.tele_history.
        """ Get max 'state' """
        tmp = df_merge.groupby(key)['state'].agg('max')
        tmp.name = 'state'
        tmp_df.append(tmp)
        """ Get max 'invite_items' """
        tmp = df_merge.groupby(key)['invite_items'].agg('max')
        tmp.name = 'invite_items'
        tmp_df.append(tmp)

        # Set the threshold about regular examination.
        df_merge['trace_minus_exam'] = df_merge['tel_year'] - df_merge.groupby(key)['examin_year'].transform('min')
        tmp = df_merge.groupby(key)['tel_minus_exam'].agg('max')
        tmp.name = 'threshold'
        tmp_df.append(tmp)

        # Combine total preprocessing data.
        df_merge = pd.concat(tmp_df, axis=1).reset_index()
        df_merge.rename(columns= {'現住行政區' : 'area'}, inplace=True)

        """ set dtype = category """

        df_merge['eco'] = pd.Categorical(
            df_merge['eco'], categories=[-1, 0, 1, 2, 3])
        df_merge['edu'] = pd.Categorical(
            df_merge['edu'], categories=[-1, 1, 2, 3, 4, 5, 6, 7])
        # last = int(self.last)
        # year_fobtDate = []
        # for i in range(4):
        #     year = last - i
        #     year_fobtDate.append('ccs_' + str(year) + '_fobtDate')

        # for i, v in df_merge.iterrows():
        #     if int(v['trace_time'] == last-2):
        #         times = 1
        #     if int(v['trace_time'] == last-1):
        #         times = 2
        #     if int(v['trace_time'] == last):
        #         times = 3

        #     if str(v[year_fobtDate[3]]) == '-1':
        #         times -= 1
        #     if (str(v[year_fobtDate[2]]) == '-1') & (int(v['trace_time']) >= last-1):
        #         times -= 1
        #     if (str(v[year_fobtDate[1]]) == '-1') & (int(v['trace_time']) >= last):
        #         times -= 1

        #     df_merge.loc[i, 'times'] = times

        # df_merge[['times']] = df_merge[['times']].astype(int)
        # df_merge = df_merge.drop(
        #     [year_fobtDate[3], year_fobtDate[2], year_fobtDate[1], year_fobtDate[0]], axis=1)

        first_ccs = df_merge.loc[df_merge['invite_items'] == 4]
        pro1_ccs_data = df_merge.drop(first_ccs.index)
        pro1_ccs_data.reset_index(inplace=True, drop=True)

        # if self.total_year % 2 == 0: 
        #     # total_year is 'even
        #     regular_threshold = self.total_year // 2 - 1
        # else: 
        #     # total_year is 'odd'
        #     regular_threshold = self.total_year // 2
        cond = (pro1_ccs_data['threshold'] % 2 == 0) # 'even'
        pro1_ccs_data['threshold'] = np.where(cond, pro1_ccs_data['threshold'] // 2 - 1, pro1_ccs_data['threshold'])
        pro1_ccs_data['threshold'] = np.where(~cond, pro1_ccs_data['threshold'] // 2, pro1_ccs_data['threshold'])

        regular_people = pro1_ccs_data.loc[pro1_ccs_data['examining_times'] >= pro1_ccs_data['threshold']]
        pro2_ccs_data = pro1_ccs_data.drop(regular_people.index)
        pro2_ccs_data.reset_index(inplace=True, drop=True)

        ma1 = pro2_ccs_data['state'] == 1
        ma2 = pro2_ccs_data['state'] == 10
        ma3 = pro2_ccs_data['state'] == 12
        useless_data = pro2_ccs_data[ma1 | ma2 | ma3]
        pro3_ccs_data = pro2_ccs_data.drop(useless_data.index)
        pro3_ccs_data.reset_index(inplace=True, drop=True)

        ma4 = pro3_ccs_data['state'] == 5
        strongly_reject = pro3_ccs_data[ma4]
        pro4_ccs_data = pro3_ccs_data.drop(strongly_reject.index)
        pro4_ccs_data.reset_index(inplace=True, drop=True)

        train_area = pro4_ccs_data['area'].value_counts().sort_index().keys()

        new_encode = {
            2: 1,
            3: 1,
            4: 0,
            6: 0,
            7: 0,
            8: 1,
            9: 1,
            11: 1,
        }
        pro4_ccs_data['state'] = pro4_ccs_data['state'].map(
            new_encode).astype(int)

        # data_dum = pd.get_dummies(pro4_ccs_data['gender'])
        # df_gender = pd.DataFrame(data_dum)
        # pro4_ccs_data = pd.concat([pro4_ccs_data, df_gender], axis=1)

        data_dum = pd.get_dummies(pro4_ccs_data['area'], prefix='a')
        df_area = pd.DataFrame(data_dum)
        pro4_ccs_data = pd.concat([pro4_ccs_data, df_area], axis=1)

        data_dum = pd.get_dummies(pro4_ccs_data['edu'], prefix='e')
        df_edu = pd.DataFrame(data_dum)
        pro4_ccs_data = pd.concat([pro4_ccs_data, df_edu], axis=1)

        data_dum = pd.get_dummies(pro4_ccs_data['eco'], prefix='eco')
        df_eco = pd.DataFrame(data_dum)
        pro4_ccs_data = pd.concat([pro4_ccs_data, df_eco], axis=1)

        X = pro4_ccs_data.drop(
            ['state', 'area', 'edu', 'eco', 'invite_items', 'trace_time'], axis=1)
        y = pro4_ccs_data[['state']]

        return X, y, train_area

    def to_predicting_set(self, train_area):
        # first_ccs = self.tele_pred.loc[self.tele_pred['invite_items'] == 2]
        first_ccs = self.tele_pred.loc[self.tele_pred['未篩含首篩名單']=='V']
        not_first_ccs = self.tele_pred.drop(first_ccs.index)
        not_first_ccs.reset_index(inplace=True, drop=True)
        not_first_ccs = not_first_ccs[['ID']]

        df_merge = pd.merge(self.exam, not_first_ccs, on='ID', how='right')
        # df_merge = df_merge.rename(columns={'addr': 'area'})
        # df_merge['area'] = df_merge['現住行政區'].str.extract("(?:台南市|臺南市)(.+區)")[0]

        # set 'key' ('ID')
        key = ['ID']
        tmp_df = []
        # Set columns from self.exam.
        """Get max 'age'."""
        tmp = df_merge.groupby(key)['年齡'].agg('max')
        tmp.name = 'age'
        tmp_df.append(tmp)
        """Get latest '區'."""
        tmp = df_merge.sort_values('篩檢日期', ascending=True).groupby(key)['現住地行政區'].last()
        tmp.name = '現住地行政區'
        tmp_df.append(tmp)
        """Get the number of examining times"""
        tmp = df_merge.groupby(key)['篩檢日期'].agg('count')
        tmp.name = 'examining_times'
        tmp_df.append(tmp)
        """Get latest 'symptom'"""
        tmp = df_merge.sort_values('篩檢日期', ascending=True).groupby(key)['篩檢結果'].last()
        tmp.name = 'latest_symp'
        tmp_df.append(tmp)
        """Get latest 'edu'"""
        tmp = df_merge.groupby(key)['edu'].last()
        tmp.name = 'edu'
        tmp_df.append(tmp)
        """Get latest 'eco'"""
        tmp = df_merge.groupby(key)['eco'].last()
        tmp.name = 'eco'
        tmp_df.append(tmp)

        """Set the threshold about regular examination."""
        df_merge['trace_minus_exam'] = self.latest_year - df_merge.groupby(key)['examin_year'].transform('min')
        tmp = df_merge.groupby(key)['tel_minus_exam'].agg('max')
        tmp.name = 'threshold'
        tmp_df.append(tmp)

        """Combine total preprocessing data."""
        df_merge = pd.concat(tmp_df, axis=1).reset_index()
        df_merge.rename(columns= {'現住行政區' : 'area'}, inplace=True)

        no_addr = df_merge.loc[df_merge['area'].isna()]
        no_screen_data = df_merge.loc[df_merge.isna().any(axis=1)]

        """concat no_screen_data and no_addr"""
        no_screen_data = pd.concat([no_screen_data, no_addr])
        df_merge.drop(no_screen_data.index, inplace=True)
        df_merge.reset_index(inplace=True, drop=True)

        # year_fobtDate = []
        # for i in range(4):
        #     year = int(self.last) - i
        #     year_fobtDate.append('ccs_' + str(year) + '_fobtDate')

        # for i, v in df_merge.iterrows():
        #     # 目前只針對4年份資料(ex.108-111 --> 4-1=3)
        #     times = 3

        #     if str(v[year_fobtDate[3]]) == '-1':
        #         times -= 1
        #     if (str(v[year_fobtDate[2]]) == '-1'):
        #         times -= 1
        #     if (str(v[year_fobtDate[1]]) == '-1'):
        #         times -= 1

        #     df_merge.loc[i, 'times'] = times

        # df_merge[['times']] = df_merge[['times']].astype(int)

        # df_merge = df_merge.drop(
        #     [year_fobtDate[3], year_fobtDate[2], year_fobtDate[1], year_fobtDate[0]], axis=1)

        cond = (df_merge['threshold'] % 2 == 0) # 'even'
        df_merge['threshold'] = np.where(cond, df_merge['threshold'] // 2 - 1, df_merge['threshold'])
        df_merge['threshold'] = np.where(~cond, df_merge['threshold'] // 2, df_merge['threshold'])

        regular_people = df_merge.loc[df_merge['examining_times'] >= df_merge['threshold']]
        pro1_ccs_data = df_merge.drop(regular_people.index)
        pro1_ccs_data.reset_index(inplace=True, drop=True)

        t = self.tele_history[['ID', 'state']]
        # keep latest 4 years tel_record (不會每年都有電訪紀錄，所以取近4年的紀錄)
        if self.tel_total_year >= 4:
            have_record = t[t['trace_time'] >= (self.latest_year-3)]
            no_record = t[~(t['trace_time'] >= (self.latest_year-3))]
        else:
            have_record = t

        # Each 'ID' keeps latest 'state'.
        have_record.sort_values('state', ascending=False, inplace=True)
        have_record.drop_duplicates(subset='ID', keep='first', inplace=True)
        pro1_ccs_data = pd.merge(pro1_ccs_data, have_record, on='ID', how='left')

        ma1 = pro1_ccs_data['state'] == 1
        ma2 = pro1_ccs_data['state'] == 10
        ma3 = pro1_ccs_data['state'] == 12
        useless_data = pro1_ccs_data[ma1 | ma2 | ma3]
        useless_data.reset_index(inplace=True, drop=True)
        pro2_ccs_data = pro1_ccs_data.drop(useless_data.index)
        pro2_ccs_data.reset_index(inplace=True, drop=True)

        ma4 = pro2_ccs_data['state'] == 5
        strongly_reject = pro2_ccs_data[ma4]
        pro3_ccs_data = pro2_ccs_data.drop(strongly_reject.index)
        pro3_ccs_data.drop(['state'], axis=1)
        pro3_ccs_data.reset_index(inplace=True, drop=True)

        """concat 'no_record' data."""
        pro3_ccs_data = pd.concat([pro3_ccs_data, no_record], axis=0)
        pro3_ccs_data.reset_index(inplace=True, drop=True)

        # data_dum = pd.get_dummies(pro3_ccs_data['gender'])
        # df_gender = pd.DataFrame(data_dum)
        # pro3_ccs_data = pd.concat([pro3_ccs_data, df_gender], axis=1)

        num_train_area = len(train_area)
        pro4_ccs_data = pro3_ccs_data.loc[pro3_ccs_data['area'].isin([train_area[0]]) == True]
        for i in range(1, num_train_area):
            pro4_ccs_data = pd.concat(
                [pro4_ccs_data, pro3_ccs_data.loc[pro3_ccs_data['area'].isin([train_area[i]]) == True]])

        addr_not_processed = pro3_ccs_data.drop(pro4_ccs_data.index)

        pro4_ccs_data['area'] = pd.Categorical(pro4_ccs_data['area'], categories=train_area)

        data_dum = pd.get_dummies(pro4_ccs_data['area'], prefix='a')
        df_area = pd.DataFrame(data_dum)
        pro4_ccs_data = pd.concat([pro4_ccs_data, df_area], axis=1)

        data_dum = pd.get_dummies(pro4_ccs_data['edu'], prefix='e')
        df_edu = pd.DataFrame(data_dum)
        pro4_ccs_data = pd.concat([pro4_ccs_data, df_edu], axis=1)

        data_dum = pd.get_dummies(pro4_ccs_data['eco'], prefix='eco')
        df_eco = pd.DataFrame(data_dum)
        pro4_ccs_data = pd.concat([pro4_ccs_data, df_eco], axis=1)

        return pro4_ccs_data.drop(['area', 'edu', 'eco'], axis=1), first_ccs, no_screen_data, regular_people, useless_data, strongly_reject, addr_not_processed

    def save(self, result, first, no_screen, regular, useless, reject, addr):
        pred_high = result.loc[result['prob'] > 0.41]
        pred_low = result.drop(pred_high.index)
        pred_high.reset_index(inplace=True, drop=True)
        pred_low.reset_index(inplace=True, drop=True)

        msg_id = pd.concat([regular['ID'], reject['ID']])
        first_id = first.loc[:, 'ID']
        pred_high_id = pred_high.loc[:, 'ID']
        pred_low_id = pred_low.loc[:, 'ID']
        useless_id = useless.loc[:, 'ID']
        addr_id = pd.concat([no_screen['ID'], addr['ID']])

        os.makedirs("./輸出名單", exist_ok=True)

        # 簡訊
        output_csv = self.tele_pred_orgin
        output_csv = pd.merge(output_csv, msg_id, on='ID', how='right')
        output_csv.to_csv('./輸出名單/簡訊名單.csv', index=False)  # 結果轉csv檔

        # 首篩
        output_csv = self.tele_pred_orgin
        output_csv = pd.merge(output_csv, first_id, on='ID', how='right')
        output_csv.to_csv('./輸出名單/首篩_電訪名單.csv', index=False)  # 結果轉csv檔

        # 高潛力
        output_csv = self.tele_pred_orgin
        output_csv = pd.merge(output_csv, pred_high_id, on='ID', how='right')
        output_csv.to_csv('./輸出名單/高潛力_電訪名單.csv', index=False)  # 結果轉csv檔

        # 低潛力
        output_csv = self.tele_pred_orgin
        output_csv = pd.merge(output_csv, pred_low_id, on='ID', how='right')
        output_csv.to_csv('./輸出名單/低潛力_電訪名單.csv', index=False)  # 結果轉csv檔

        # 「電話錯誤、空號、死亡名單」會多一個欄位紀錄state
        output_csv = self.tele_pred_orgin
        output_csv = pd.merge(output_csv, useless_id, on='ID', how='right')
        output_csv = pd.concat([output_csv, useless['state']], axis=1)
        output_csv.to_csv('./輸出名單/電話錯誤、空號、死亡名單.csv', index=False)  # 結果轉csv檔

        # 模型無法處理
        output_csv = self.tele_pred_orgin
        output_csv = pd.merge(output_csv, addr_id, on='ID', how='right')
        output_csv.to_csv('./輸出名單/模型無法處理名單.csv', index=False)  # 結果轉csv檔
