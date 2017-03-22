from flask import render_template, request, session, g, redirect, url_for, make_response
from InsightsApp import app, jobmodel

TABS = [{'content': 'This is your analysis 1', 'name': 'Analysis 1'},
        {'content': 'This is your analysis 2', 'name': 'Analysis 2'},
        {'content': 'This is your analysis 3', 'name': 'Analysis 3'},
        {'content': 'This is your analysis 4', 'name': 'Analysis 4'}]

TABS = []


@app.route('/about/')
def about():
    return render_template('index.html')


@app.route('/')
def hello():
    session['input_text'] = ''
    session['tabs'] = TABS
    session['results'] = ''
    return redirect(url_for("about"))


@app.route('/submit/', methods=['GET', 'POST'])
def submit():
    session['input_text'] = request.form['intro']
    if request.form['submit'] == 'Analyze':
        if not session['input_text']:
            return redirect(url_for("hello"))
        if len(session['input_text'].split()) < 25:
            return redirect(url_for("about"))
        session['results'] = jobmodel.reparse(session['input_text'])
        session['tabs'] = TABS
        return redirect(url_for("about"))
    if request.form['submit'] == 'Cancel':
        return redirect(url_for("hello"))


@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for("hello"))


@app.route('/output_1/', methods=['GET', 'POST'])
def post_result_1():
    result = 'this is the 1st analysis result'
    return result


@app.route('/output_2/', methods=['GET', 'POST'])
def post_result_2():
    result = 'this is the 2st analysis result'
    return result


@app.route('/output_3/', methods=['GET', 'POST'])
def post_result_3():
    result = 'this is the 3st analysis result'
    return result


@app.route('/output_4/', methods=['GET', 'POST'])
def post_result_4():
    result = 'this is the 4st analysis result'
    return result
