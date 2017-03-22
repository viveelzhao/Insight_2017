from flask import Flask, render_template, request
from InsightsApp import app, jobmodel



TEST_PARAMS = {'tabs': [{'content': 'This is your analysis 1', 'name': 'Analysis 1'},
                        {'content': 'This is your analysis 2', 'name': 'Analysis 2'},
                        {'content': 'This is your analysis 3', 'name': 'Analysis 3'},
                        {'content': 'This is your analysis 4', 'name': 'Analysis 4'}],
               'results': []}



@app.route('/')
@app.route('/about')
def about():
    input_text = ''
    return render_template('index.html', params=[input_text, TEST_PARAMS])


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    input_text = request.args.get('intro')
    TEST_PARAMS['results'] = jobmodel.reparse(input_text)
    return render_template('index.html', params=[input_text, TEST_PARAMS])


@app.route('/output_1', methods=['GET', 'POST'])
def post_result_1():
    result = 'this is the 1st analysis result'
    return result


@app.route('/output_2', methods=['GET', 'POST'])
def post_result_2():
    result = 'this is the 2st analysis result'
    return result


@app.route('/output_3', methods=['GET', 'POST'])
def post_result_3():
    result = 'this is the 3st analysis result'
    return result


@app.route('/output_4', methods=['GET', 'POST'])
def post_result_4():
    result = 'this is the 4st analysis result'
    return result
