from flask import Flask, render_template, request
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


from keras.preprocessing.image import load_img 
from keras .preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 
from keras.applications.vgg16 import decode_predictions 
from keras.applications.vgg16 import VGG16

app = Flask(__name__)
app=Flask(__name__,template_folder='templates')
model = VGG16()

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/", methods=['POST'])
def predict() :
    imagefile= request.files["imagefile"]
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]
    
    classification = '%s (%.2f%%)' % (label[1], label[2]*100)
    
    return render_template('index.html',prediction=classification)










if __name__ == "__main__":
    # app.run(debug=True,host="localhost", port=5001)
    app.run(debug=True,host='0.0.0.0', port=5001)