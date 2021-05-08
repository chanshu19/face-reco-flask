from PIL import Image
from mtcnn.mtcnn import MTCNN
from numpy import reshape
from numpy import load
from numpy import expand_dims
from numpy import asarray
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

class FaceReco():
    
    def __init__(self,raw_image):
        self.image = raw_image
    # extract a single face from a given photograph
    def extract_face(self, filename, required_size=(160, 160)):
        # load image from file
        image = Image.open(filename)
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        pixels = asarray(image)
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(pixels)
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array

    # get the face embedding for one face
    def get_embedding(self, model, face_pixels):
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = model.predict(samples)
        return yhat[0]

    def recognise_name(self):
        # extract the face pixels
        facePixels = self.extract_face(self.image)
        # load the facenet model
        model = load_model('model/facenet_keras.h5')
        # compute the face embedding
        faceEmbeddings = self.get_embedding(model=model,face_pixels=facePixels)
        # reshape the embedding
        faceEmbeddings = reshape(faceEmbeddings,(1,128))

        # Now We need to to feed this embedding into classifier

        # load face embeddings
        data = load('registered_faces/5-celebrity-faces-embeddings.npz')
        trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        # normalize input vectors
        in_encoder = Normalizer(norm='l2')
        trainX = in_encoder.transform(trainX)
        testX = in_encoder.transform(testX)
        # label encode targets
        out_encoder = LabelEncoder()
        out_encoder.fit(trainy)
        trainy = out_encoder.transform(trainy)
        testy = out_encoder.transform(testy)
        # fit model
        model = SVC(kernel='linear', probability=True)
        model.fit(trainX, trainy)
        # predict face
        yhat_class = model.predict(faceEmbeddings)
        predict_names = out_encoder.inverse_transform(yhat_class)
        return predict_names
    




