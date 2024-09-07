from flask import Flask, render_template, request, redirect, url_for
import os
from markupsafe import Markup
from DetectStuttering import DetectStuttering


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the text input
        text_input = request.form['text_input']

        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save the uploaded file
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Call the stutter detection model
            stutter_result = DetectStuttering(file_path, text_input).predict_megamgem()
            print(stutter_result)
            stutter_result=stutter_result.replace('\n','<br>')
            # Render result template with the result
            return render_template('index.html', text_input=text_input, stutter_result=Markup(stutter_result))

    # Render the index template for GET requests or errors
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
