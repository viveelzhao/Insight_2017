# Insight Data Science Project

I worked on a consulting project with a job marketplace company to help them to build a better job classification system. Problems are first, jobs are not placed in the right industry buckets, and second, jobs are not ranked according to the skills required. For example, assuming our user is interested in data analyst position in Finance, but the website displays several customer representative positions in real estate on the first page. These are common problems for many job marketplace websites. 

The objective of this project therefore are two folders: first, I need to create career cluster (industry) label and second, to recommend similar jobs based on the skills required. The training data set were pulled from multiple tables in the PostgreSQL database. I have no access to user information. The input features are job titles and job descriptions. 

The algorithm analyzes the words in the job title and job description in free text form, extracts skills and knowledge required , and measures the score for each skill. It maps every job to ONET(https://www.onetonline.org) occupation to get career cluster label. It'll search for the nearest job based on the skills and scores.

To demonstrate the algorithm, I built www.skillessence.com where you can type in or paste a job desciption, then click analyze and see what you get. Please note, to protect proprietary job listings content for the consulting company that I worked with, only generic job titles from ONET are displayed.




