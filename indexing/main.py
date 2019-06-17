import os
from flask import Flask, render_template, request, url_for,json
import uuid
import run_query
import gen_caption
#from bs4 import BeautifulSoup
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/Uploads'


@app.route('/')
def abc():
	return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
			if request.form['caption'] != '':
				print("work caption")
				list_img = run_query.query(request.form['caption'])
				
				return render_template('index.html', array = list_img, caption = request.form['caption'] )
				#return render_template('index.html')
			else :
				file = request.files['file']
				extension = os.path.splitext(file.filename)[1]
				f_name = str(uuid.uuid4()) + extension
				file.save(os.path.join(app.config['UPLOAD_FOLDER'], f_name))
				#print(f_name)
				caption_query = gen_caption.return_caption('static/Uploads/' + f_name)

				list_img = run_query.query(caption_query)
				
				return render_template('index.html', array = list_img,anhupload=f_name,caption_query = caption_query)
			
if __name__ == "__main__":
	app.run()