import json
import torch
from flask import Flask, render_template, request
# from flask_uploads import UploadSet, configure_uploads,IMAGES
# from scipy.misc import imsave, imread, imresize
# from flask import Flask, request
# from serve import model_loader
# import numpy as np

# import io
# import json

# from torchvision import models
# import torchvision.transforms as transforms
# from PIL import Image
# from flask import Flask, jsonify, request


# import re
# #system level operations (like loading files)
# import sys
# #for reading operating system data
# import os

# # I've commented out the last import because it won't work in kernels, 
# # but you should uncomment it when we build our app tomorrow

# # create an instance of Flask
# app = Flask(__name__)

# # load our pre-trained model & function
# model = model_loader()
# model.eval()

# photos = UploadSet('photos', IMAGES)

# app.config['UPLOADED_PHOTOS_DEST'] = '.'
# configure_uploads(app, photos)

# def transform_image(image_bytes):
#     my_transforms = transforms.Compose([transforms.Resize(255),
#                                         transforms.CenterCrop(224),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize(
#                                             [0.485, 0.456, 0.406],
#                                             [0.229, 0.224, 0.225])])
#     image = Image.open(io.BytesIO(image_bytes))
#     return my_transforms(image).unsqueeze(0)

@app.route('/index')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")



# @app.route('/upload', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST' and 'photo' in request.files:
#         filename = photos.save(request.files['photo'])
#         os.rename('./'+filename,'./'+'output.png')

# 	print "debug"
# 	#read the image into memory
# 	img = image.load_img('./output.png', target_size=(224, 224))
# 	#compute a bit-wise inversion so black becomes white and vice versa
	
# 	tensor = transform_image(img)
# 	img=model(tensor)
# 	_, predicted = torch.max(img, 1)
# 	ans=(int)predicted
	
	
	
	
# # 	x = image.img_to_array(img)
# # 	#make it the right size
# # 	x = np.expand_dims(x, axis=0)
# # 	#imshow(x)
# # 	#convert to a 4D tensor to feed into our model
# # 	x = preprocess_input(x)
# # 	print "debug2"
# # 	#in our computation graph
# # 	with graph.as_default():
# # 		#perform the prediction
# # 		out = model.predict(x)
# # 		u = decode_predictions(out, top=3)[0]
#     	s1 = ans
# 	s2 = ans*100
# # 		s3 = u[1][1]
# # 		s4 = u[1][2]*100
# # 		s5 = u[2][1]
# # 		s6 = u[2][2]*100
# # 		print(s1,s2,s3)
# # 		print "debug3"
# 		#convert the response to a string
# 	return render_template("index2.html",s1 = s1, s2 = s2)


# if __name__ == "__main__":
# 	#decide what port to run the app in
# 	port = int(os.environ.get('PORT', 5000))
# 	#run the app locally on the givn port
# 	app.run(host='0.0.0.0', port=port)
# 	#optional if we want to run in debugging mode
# 	#app.run(debug=True)
