

def read_test_params(param_file='InsightsApp/static/testparams.dat'):
    try:
        with open(param_file, 'rb') as f:
            test_params = pickle.load(f)
    except IOError:
        print('wrong')
        test_params = {}
    return test_params

TEST_PARAMS = read_test_params()


@app.route('/')
@app.route('/about')
def about():
    input_text = ''
    params = [input_text, TEST_PARAMS]
    return render_template('index.html', params=params)


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    input_text = request.args.get('intro')
    for tab in TEST_PARAMS['tabs']:
        tab['content'] = 'This is the analysis results for ' + input_text
    params = [input_text, TEST_PARAMS]
    return render_template('index.html', params=params)


@app.route('/output_1', methods=['GET', 'POST'])
def post_result_1():
    result = 'this is the 1st analysis result at '
    return result

@app.route('/output_2', methods=['GET', 'POST'])
def post_result_2():
    result = 'this is the 2st analysis result at '
    return result

@app.route('/output_3', methods=['GET', 'POST'])
def post_result_3():
    result = 'this is the 3st analysis result at '
    return result

@app.route('/output_4', methods=['GET', 'POST'])
def post_result_4():
    result = 'this is the 4st analysis result at '
    return result
