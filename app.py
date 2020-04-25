import flask
from flask import request, redirect, send_from_directory
from werkzeug.utils import secure_filename
from fastai.vision import *

app = flask.Flask(__name__, template_folder='templates', static_url_path='')
app.secret_key = os.urandom(24)

path = os.path.dirname(__file__)

allowed_extension = {'png', 'jpg', 'jpeg'}

clf = load_learner(path, 'model/hedhika-classifier.pkl')


@app.route('/uploads/<path:path>')
def send_image(path):
    return send_from_directory('uploads', path)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extension


@app.route('/', methods=['GET', 'POST'])
def main_page():
    if flask.request.method == 'POST':
        file = request.files['file']
        if 'file' not in request.files:
            prediction_text = 'No file part'
            return redirect(request.url)
        if file.filename == '':
            prediction_text = 'No file selected for uploading'
            return redirect(request.url)
        if file:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join('uploads', filename))
                return redirect(flask.url_for('prediction', filename=filename))
    return flask.render_template('index.html')


@app.route('/prediction/<filename>')
def prediction(filename):
    test_image = open_image(os.path.join('uploads', filename))
    preds, idx, output = clf.predict(test_image)
    predictions = dict({clf.data.classes[i]: round(to_np(p) * 100, 2) for i, p in enumerate(output) if p > 0.2})

    return flask.render_template('prediction.html', predictions=predictions, i_path=(os.path.join('uploads', filename)).replace(os.sep,"/"))


if __name__ == '__main__':
    app.run(debug=True)