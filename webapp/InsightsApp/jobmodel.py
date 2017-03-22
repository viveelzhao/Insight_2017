import re
import pandas as pd
#import sys
#import os
#import json
#import string
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.externals import joblib
from sklearn.feature_extraction import text
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction import text
#from sklearn.cross_validation import train_test_split
#from sklearn.pipeline import Pipeline
#from sklearn.grid_search import GridSearchCV
#from sklearn import linear_model
from scipy.stats import percentileofscore
#from sklearn.metrics.pairwise import pairwise_distances
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

predictors = {'Knowledge':   ['Administration and Management',
                              'Biology',
                              'Building and Construction',
                              'Chemistry',
                              'Clerical',
                              'Communications and Media',
                              'Computers and Electronics',
                              'Customer and Personal Service',
                              'Design',
                              'Economics and Accounting',
                              'Education and Training',
                              'Engineering and Technology',
                              'English Language',
                              'Fine Arts',
                              'Food Production',
                              'Foreign Language',
                              'Geography',
                              'History and Archeology',
                              'Law and Government',
                              'Mathematics_Knowledge',
                              'Mechanical',
                              'Medicine and Dentistry',
                              'Personnel and Human Resources',
                              'Philosophy and Theology',
                              'Physics',
                              'Production and Processing',
                              'Psychology',
                              'Public Safety and Security',
                              'Sales and Marketing',
                              'Sociology and Anthropology',
                              'Telecommunications',
                              'Therapy and Counseling',
                              'Transportation'],
                   'Skills': ['Active Learning',
                              'Active Listening',
                              'Complex Problem Solving',
                              'Coordination',
                              'Critical Thinking',
                              'Equipment Maintenance',
                              'Equipment Selection',
                              'Installation',
                              'Instructing',
                              'Judgment and Decision Making',
                              'Learning Strategies',
                              'Management of Financial Resources',
                              'Management of Material Resources',
                              'Management of Personnel Resources',
                              'Mathematics',
                              'Monitoring',
                              'Negotiation',
                              'Operation Monitoring',
                              'Operation and Control',
                              'Operations Analysis',
                              'Persuasion',
                              'Programming',
                              'Quality Control Analysis',
                              'Reading Comprehension',
                              'Repairing',
                              'Science',
                              'Service Orientation',
                              'Social Perceptiveness',
                              'Speaking',
                              'Systems Analysis',
                              'Systems Evaluation',
                              'Technology Design',
                              'Time Management',
                              'Troubleshooting',
                              'Writing']}

all_predictors = predictors['Knowledge']+predictors['Skills']

my_additional_stop_words = set(['hr', 'self', 'send resume', 'send', 'resume', 'day', 'hours', 'nj', 'currently',
                                'topics', 'interested', 'project', 'city', 'email', 'rep', 'staff', 'intern',
                                'college', 'la', 'need', 'location', 'hiring', 'annual', 'people', 'currently',
                                'www', 'yrs', 'contact', 'opportunities', 'county', 'exp', 'seeking', 'person',
                                'com', 'cdl', 'nh', 'help', 'great', 'plus', 'li', 'li li', 'li  li', 'he', 'eoe',
                                'span', 'monthly', 'va',
                                'pa', 'industry', 'projects', 'candidate', 'year', 'nbsp', 'oh', 'employment',
                                'rn', 'salary', 'lpn', 'br', 'br br', 'ny', 'ne', 'jobs', 'positions', 'nj', 'keywords',
                                'need', 'current', 'tn', 'ca', 'fl', 'va', 'td', 'rep', 'st', 'th', 'seeks', 'rd',
                                'daily',
                                'resumes', 'eh', 'id', 'id category', 'good', 'news category', 'pm', 'am', 'looking',
                                'apply',
                                'req', 'new', 'ft', 'tr', 'years', 'ms', 'org', 'pt', 'myownjobmatch',
                                'myownjobmatch category',
                                'york', 'tx', 'mo', 'hour', 'al', 'wanted', 'needed', 'category', 'position',
                                'yourmembership', 'yourmembership category', 'bls', 'hrs', 'cv', 'bc', 'mso',
                                'ed', 'ky', 'week', 'days', 'hrs', 'sc', 'ks', 'ascp', 'able', 'availabe', 'internship',
                                'job', 'area', 'letter', 'want', 'nyc', 'days', 'open', 'openings', 'nc', 'cnc',
                                'door door',
                                'hrs', 'http', 'title', 'acception applications', 'div', 'deg', 'omaha', 'los', 'cna',
                                'attn',
                                'px', 'mso', 'ks', 'cnc', 'nx', 'ref', 'skills', 'time event', 'fin', 'rel', 'incl',
                                'info', 'los angeles',
                                'category', 'opportunity', 'equal opportunity', 'il', 'equal', 'll', 'weel', 'ed',
                                'gigtown', 'div',
                                'div div', 'event', 'nc', 'ks', 'dev', 'ct', 'project'])
stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)

tfidf_vect = joblib.load("InsightsApp/static/models/tfidf.vect")
best_estimator={}
for predictor in all_predictors:
    best_estimator[predictor] = joblib.load("InsightsApp/static/models/"+predictor+".sgd")



class my_prediction:
    def __init__(self,query_df,tfidf_vect,models):
        self.query_df = query_df
        self.tfidf_vect = tfidf_vect
        #self.svd = svd
        self.models = models
        self.clean_text()
        self.vectorizer()
        #self.reduce_dimension()

    def clean_text(self):
        self.query_df['clean_text'] = self.query_df['text'].apply(lambda x: re.sub(r'\s',' ', x).lower())

    def vectorizer(self):
        self.query_matrix = self.tfidf_vect.transform(self.query_df['clean_text'])
        return self

   # def reduce_dimension(self):
    #    self.query_reduced_matrix = self.svd.transform(self.query_matrix)
     #   return self.query_reduced_matrix

    def predict_score(self):
        self.score_df = self.query_df
        for i, predictor in enumerate(self.models.keys()):
            prediction = self.models[predictor].predict(self.query_matrix)
            self.score_df[predictor]=prediction
        return self.score_df

listing_train_normalized_score_df = pd.read_csv('InsightsApp/static/listing_train_normalized_score.csv')
listing_train_for_query_df = pd.read_csv('InsightsApp/static/listing_train_for_query.csv')
print 'loading files ready.'


def cal_percentile(s,col):
    score = percentileofscore(listing_train_for_query_df[col].values, s, kind='mean')
    return score

def normalize_score(df,all_predictors):
    for col in all_predictors:
        df[col]=df[col].apply(lambda x: cal_percentile(x,col))
    return df


def get_nearest_jobs(query_normalized_score, cols=all_predictors, c_cluster=None):
    recommended_jobs = defaultdict(dict)
    # print c_cluster
    if c_cluster:
        # print "......"
        # print c_cluster
        subset_train = listing_train_for_query_df.loc[listing_train_for_query_df['Career Cluster'] == c_cluster, :]
        nn = NearestNeighbors(n_neighbors=50, algorithm='brute', metric='cosine', n_jobs=1)
        nn.fit(subset_train[cols])
        nn_index = nn.kneighbors(query_normalized_score[cols], return_distance=True)
        for i, index in enumerate(nn_index[1][0]):
            [job, cluster] = subset_train.iloc[index][['Occupation', 'Career Cluster']]
            recommended_jobs[i]['job'] = job
            recommended_jobs[i]['career cluster'] = c_cluster
        return recommended_jobs

    else:
        nn = NearestNeighbors(n_neighbors=50, algorithm='brute', metric='cosine', n_jobs=1)
        nn.fit(listing_train_normalized_score_df[cols])
        nn_index = nn.kneighbors(query_normalized_score[cols], return_distance=True)
        for i, index in enumerate(nn_index[1][0]):
            [job, cluster] = listing_train_for_query_df.iloc[index][['Occupation', 'Career Cluster']]
            recommended_jobs[i]['job'] = job
            recommended_jobs[i]['career cluster'] = cluster
            # recommended_jobs[i]['similarity'] = 100- np.power(nn_index[0][0][i],1.0/66)*10
        return recommended_jobs


def dict_to_list(score_dict):
    table_content = []
    for i in score_dict:
        k = {}
        k['name'] = i[0]
        k['score'] = int(i[1])
        table_content.append(k)
    return table_content


def order_jobs(jobs_dict, top_k=10):
    r_jobs = []
    for i, v in enumerate(jobs_dict.values()[0:top_k]):
        jobs = {}
        jobs['name'] = v['career cluster']
        jobs['score'] = v['job']
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
    query_result.append({'tablename': 'Skills',
                         'tablecontents': skills_ordered,
                         'tablecategory': 'info'})
    query_result.append({'tablename': 'Knowledge',
                         'tablecontents': knowledge_ordered,
                         'tablecategory': 'warning'})

    recommended_jobs = get_nearest_jobs(query_predicted_score, cols=all_predictors, c_cluster=career_cluster)
    recommended_jobs_ordered = order_jobs(recommended_jobs)
    query_result.append({'tablename': 'Recommended Jobs and Career Clusters',
                         'tablecontents': recommended_jobs_ordered,
                         'tablecategory': 'success'})

    return query_result

def reparse(text):
    category=['info', 'warning', 'success', 'danger']
    results = query_predict(text, career_cluster=None)

    return results

