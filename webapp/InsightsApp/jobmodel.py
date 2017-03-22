import re
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction import text
from scipy.stats import percentileofscore
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
import json
import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import io

MODEL_NUMBER = 2

def readJson(filename):
    with open(os.path.join('InsightsApp/static/', filename), 'r') as f:
        s = json.load(f)
    return s

my_additional_stop_words = readJson('stopwords.json')
predictors = readJson('predictors.json')


all_predictors = predictors['Knowledge']+predictors['Skills']
stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)


listing_train_normalized_score_df = pd.read_csv('InsightsApp/static/listing_train_normalized_score.csv')
listing_train_for_query_df = pd.read_csv('InsightsApp/static/listing_train_for_query.csv')



def readModels(modelnum=MODEL_NUMBER):
    modelPath = '_'.join(["InsightsApp/static/models", str(modelnum)])
    v = joblib.load(os.path.join(modelPath, "tfidf.vect"))
    m = {}
    for predictor in all_predictors:
        m[predictor] = joblib.load(os.path.join(modelPath, predictor + ".sgd"))
    return v, m


tfidf_vect, best_estimator = readModels(modelnum=MODEL_NUMBER)


def writeJson(data, filename, mode='w'):
    with io.open(os.path.join('InsightsApp/static/', filename), mode, encoding='utf8') as f:
        s = json.dumps(data, indent=4, ensure_ascii=False)
        f.write(unicode(s))

# writeJson(predictors, 'predictors.json')
# writeJson(list(my_additional_stop_words), 'stopwords.json')


class my_prediction:

    def __init__(self,query_df,tfidf_vect,models):
        self.query_df = query_df
        self.tfidf_vect = tfidf_vect
        self.models = models
        self.clean_text()
        self.vectorizer()

    def clean_text(self):
        self.query_df['clean_text'] = self.query_df['text'].apply(lambda x: re.sub(r'\s',' ', x).lower())

    def vectorizer(self):
        self.query_matrix = self.tfidf_vect.transform(self.query_df['clean_text'])
        return self

    def predict_score(self):
        self.score_df = self.query_df.copy()
        for i, predictor in enumerate(self.models.keys()):
            prediction = self.models[predictor].predict(self.query_matrix)
            self.score_df[predictor]=prediction
        return self.score_df


def cal_percentile(s,col):
    score = percentileofscore(listing_train_for_query_df[col].values, s, kind='mean')
    return score


def normalize_score(df, all_predictors):
    df_normalized = df.copy()
    for col in all_predictors:
        df_normalized[col] = df[col].apply(lambda x: cal_percentile(x, col))
    return df_normalized


def get_nearest_jobs(query_score, cols=all_predictors, c_cluster=None):
    recommended_jobs = defaultdict(dict)
    if c_cluster:
        subset_train = listing_train_for_query_df.loc[listing_train_for_query_df['Career Cluster'] == c_cluster, :]
        nn = NearestNeighbors(n_neighbors=50, algorithm='brute', metric='euclidean', n_jobs=1)
        nn.fit(subset_train[cols])
        nn_index = nn.kneighbors(query_score[cols], return_distance=True)
        for i, index in enumerate(nn_index[1][0]):
            [job, cluster] = subset_train.iloc[index][['Occupation', 'Career Cluster']]
            recommended_jobs[i]['job'] = job
            recommended_jobs[i]['career cluster'] = c_cluster
            recommended_jobs[i]['distance'] = nn_index[0][0][i]
        return recommended_jobs
    else:
        nn = NearestNeighbors(n_neighbors=50, algorithm='brute', metric='euclidean', n_jobs=1)
        nn.fit(listing_train_for_query_df[cols])
        nn_index = nn.kneighbors(query_score[cols], return_distance=True)
        for i, index in enumerate(nn_index[1][0]):
            [job, cluster] = listing_train_for_query_df.iloc[index][['Occupation', 'Career Cluster']]
            recommended_jobs[i]['job'] = job
            recommended_jobs[i]['career cluster'] = cluster
            recommended_jobs[i]['distance'] = nn_index[0][0][i]
        return recommended_jobs


def dict_to_list(score_dict, threshold=50):
    table_content = []
    for i in score_dict:
        k = {}
        k['name'] = i[0]
        k['score'] = int(i[1])
        if k['score'] >= threshold:
            table_content.append(k)
    return table_content


def order_jobs(jobs_dict, top_k=10, threshold=2):
    r_jobs = []
    for i, v in enumerate(jobs_dict.values()[0:top_k]):
        jobs = {}
        jobs['name'] = v['career cluster']
        jobs['score'] = v['job']
        if v['distance'] < threshold and jobs not in r_jobs:
            r_jobs.append(jobs)
    return r_jobs


def query_predict(query_text, career_cluster):
    query_result = []
    query_text = re.sub('[^a-zA-Z ]', ' ', query_text)
    query_point = pd.DataFrame([{'text': query_text}])
    query_predicted_score = my_prediction(query_point, tfidf_vect, best_estimator).predict_score()
    query_normalized_score = normalize_score(query_predicted_score, all_predictors)
    skills_score_dict = query_normalized_score[predictors['Skills']].to_dict('records')
    skills_score_dict = sorted(skills_score_dict[0].items(), key=lambda (k, v): v, reverse=True)[0:10]
    skills_ordered = dict_to_list(skills_score_dict)
    knowledge_score_dict = query_normalized_score[predictors['Knowledge']].to_dict('records')
    knowledge_score_dict = sorted(knowledge_score_dict[0].items(), key=lambda (k, v): v, reverse=True)[0:10]
    knowledge_ordered = dict_to_list(knowledge_score_dict)
    query_result.append({u'tablename': u'Skills',
                         u'tablecontents': skills_ordered,
                         u'tablecategory': u'info'})
    query_result.append({u'tablename': u'Knowledge',
                         u'tablecontents': knowledge_ordered,
                         u'tablecategory': u'warning'})
    if len(skills_ordered) or len(knowledge_ordered):
        recommended_jobs = get_nearest_jobs(query_predicted_score, cols=all_predictors, c_cluster=career_cluster)
        recommended_jobs_ordered = order_jobs(recommended_jobs)
    else:
        recommended_jobs_ordered = []
    query_result.append({u'tablename': u'Recommended Jobs and Career Clusters',
                         u'tablecontents': recommended_jobs_ordered,
                         u'tablecategory': u'success'})
    return query_result


def reparse(text, debug=False):
    global tfidf_vect, best_estimator
    category=[u'info', u'warning', u'success', u'danger']
    if debug:
        test_path = 'InsightsApp/test_case/'
        test_cases = [f for f in os.listdir(test_path) if f[-4:] == '.txt']
        print test_cases
        for m in range(1, MODEL_NUMBER + 1):
            tfidf_vect, best_estimator = readModels(modelnum=m)
            for f in test_cases:
                with open(os.path.join(test_path, f), 'r') as case:
                    case_text = case.read()
                    case_text = re.sub('{.*?}', ' ', case_text)
                    case_text = re.sub('[^a-zA-Z ]', ' ', case_text)
                    case_text = re.sub('\\b\\w{1,2}\\b|\\b\\w{20,}\\b', ' ', case_text)
                    results = query_predict(case_text, career_cluster=None)
                    writeJson({'input': case_text,
                               'model': m,
                               'results': results},
                              filename='results.log',
                              mode='a')
    results = query_predict(text, career_cluster=None)
    return results

