from flask import Flask, jsonify,request, send_from_directory
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage
import os 
import uuid 
import logging
from PIL import Image 
import numpy as np
import io
from logging.handlers import RotatingFileHandler

from faceshine import data2image
from faceshine.tasks import TaskZeroBackground, \
    TaskImageColorizer, TaskLowLight, TaskEraseScratches, TaskSuperFace, TaskFaceSegmentation

appFaceShine = Flask(__name__)
api = Api(appFaceShine)

taskFaceSegmentation = TaskFaceSegmentation()
taskSuperFace = TaskSuperFace()
taskLowLight = TaskLowLight()
taskEraseScratches = TaskEraseScratches()
taskImageColorizer = TaskImageColorizer()
taskZeroBackground = TaskZeroBackground()

#Confugure the debug points
logging.basicConfig(level=logging.DEBUG)


#Configure Logging
log_handler = RotatingFileHandler('app.log',maxBytes=1000000,backupCount=3)
log_handler.setLevel(logging.INFO)
appFaceShine.logger.addHandler(log_handler)

os.makedirs('processed_images', exist_ok=True)
os.makedirs('temp_files', exist_ok=True)

#Utility function to create unique directories
def create_unique_temp_dir():
    unique_id = str(uuid.uuid4())
    temp_dir  = os.path.join('temp_files',unique_id)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    return temp_dir

def get_download_url(filename):
    base_url = os.getenv('BASE_URL','http://localhost:5000/')
    return f"{base_url}download/{filename}"

@appFaceShine.route('/downlaod/<filename>')
def download_file(filename):
    try:
        appFaceShine.logger.info(f"Attempting to send file: {filename}")
        file_path = os.path.join('processed_images', filename)
        if os.path.exists(file_path):
            appFaceShine.logger.info(f"File {filename} exists. Sending file.")
            return send_from_directory('processed_images', filename, as_attachment=True)
        else:
            appFaceShine.logger.error(f"File {filename} does not exist.")
            return jsonify({"error": f"File {filename} does not exist"}), 404
    except Exception as e:
        appFaceShine.logger.error(f"Error sending file: {e}")
        return jsonify({"error": f"Error sending file: {e}"}), 500
    

parser = reqparse.RequestParser()

parser.add_argument('image',
                    type=FileStorage,
                    location='files',
                    required=False,
                    help='Provide one image file')


class Index(Resource):
    def get(self):
        return {'It works!': 'AI Remote Procedure Machine'}
    
class FaceSegmentation(Resource):
    def post(self):
        appFaceShine.logger.info("Received request for face segmentation.")
        args = parser.parse_args()
        if not args['image']:
            appFaceShine.logger.error("No image provided in the request.")
            return jsonify({"error": "No image provided"}), 400

        temp_dir = create_unique_temp_dir()
        input_path = os.path.join(temp_dir, 'input_image.jpg')
        args['image'].save(input_path)

        appFaceShine.logger.info(f"Image saved to {input_path}")

        image = data2image(open(input_path, 'rb').read())
        appFaceShine.logger.info("Running face segmentation task.")
        prediction = taskFaceSegmentation.executeTask(image)
        appFaceShine.logger.info("Face segmentation completed.")

        # print("Type of prediction:", type(prediction))
        # if isinstance(prediction, np.ndarray):
        #     print("Shape of prediction:", prediction.shape)
        
        if prediction.ndim == 2:  # Grayscale image
            prediction_image = Image.fromarray(prediction.astype(np.uint8), mode='L')
        elif prediction.ndim == 3 and prediction.shape[2] == 3:  # RGB Image
            prediction_image = Image.fromarray(prediction.astype(np.uint8), mode='RGB')
        else:
            appFaceShine.logger.error("Unexpected format for prediction")
            return jsonify({"error": "Unexpected format for prediction"})

        # Save the processed image
        output_filename = f"{uuid.uuid4()}.jpg"
        output_path = os.path.join('processed_images', output_filename)
        prediction_image.save(output_path)
        appFaceShine.logger.info(f"Processed image saved to {output_path}")

        return jsonify({"url": get_download_url(output_filename)})


class SuperFace(Resource):
    def post(self):
        appFaceShine.logger.info("Received request for face segmentation.")
        args = parser.parse_args()
        if not args['image']:
            appFaceShine.logger.error("No image provided in the request.")
            return jsonify({"error": "No image provided"}), 400

        temp_dir = create_unique_temp_dir()
        input_path = os.path.join(temp_dir, 'input_image.jpg')
        args['image'].save(input_path)

        appFaceShine.logger.info(f"Image saved to {input_path}")

        image = data2image(open(input_path, 'rb').read())
        appFaceShine.logger.info("Running face segmentation task.")
        prediction = taskSuperFace.executeTask(image)
        appFaceShine.logger.info("Face segmentation completed.")

        # print("Type of prediction:", type(prediction))
        # if isinstance(prediction, np.ndarray):
        #     print("Shape of prediction:", prediction.shape)
        
        if prediction.ndim == 2:  # Grayscale image
            prediction_image = Image.fromarray(prediction.astype(np.uint8), mode='L')
        elif prediction.ndim == 3 and prediction.shape[2] == 3:  # RGB Image
            prediction_image = Image.fromarray(prediction.astype(np.uint8), mode='RGB')
        else:
            appFaceShine.logger.error("Unexpected format for prediction")
            return jsonify({"error": "Unexpected format for prediction"})

        # Save the processed image
        output_filename = f"{uuid.uuid4()}.jpg"
        output_path = os.path.join('processed_images', output_filename)
        prediction_image.save(output_path)
        appFaceShine.logger.info(f"Processed image saved to {output_path}")

        return jsonify({"url": get_download_url(output_filename)})


class ImageLowLight(Resource):
    def post(self):
        appFaceShine.logger.info("Received request for face segmentation.")
        args = parser.parse_args()
        if not args['image']:
            appFaceShine.logger.error("No image provided in the request.")
            return jsonify({"error": "No image provided"}), 400

        temp_dir = create_unique_temp_dir()
        input_path = os.path.join(temp_dir, 'input_image.jpg')
        args['image'].save(input_path)

        appFaceShine.logger.info(f"Image saved to {input_path}")

        image = data2image(open(input_path, 'rb').read())
        appFaceShine.logger.info("Handling Low light issue.")
        prediction = taskLowLight.executeTask(image)
        appFaceShine.logger.info("Light Enhanced.")

        # print("Type of prediction:", type(prediction))
        # if isinstance(prediction, np.ndarray):
        #     print("Shape of prediction:", prediction.shape)
        
        if prediction.ndim == 2:  # Grayscale image
            prediction_image = Image.fromarray(prediction.astype(np.uint8), mode='L')
        elif prediction.ndim == 3 and prediction.shape[2] == 3:  # RGB Image
            prediction_image = Image.fromarray(prediction.astype(np.uint8), mode='RGB')
        else:
            appFaceShine.logger.error("Unexpected format for prediction")
            return jsonify({"error": "Unexpected format for prediction"})

        # Save the processed image
        output_filename = f"{uuid.uuid4()}.jpg"
        output_path = os.path.join('processed_images', output_filename)
        prediction_image.save(output_path)
        appFaceShine.logger.info(f"Processed image saved to {output_path}")

        return jsonify({"url": get_download_url(output_filename)})


class EraseScratches(Resource):
    def post(self):
        appFaceShine.logger.info("Recieved request to remove scratches.")
        args = parser.parse_args()
        if not args['image']:
            appFaceShine.logger.error("No image provided in the request.")
            return jsonify({"error": "No image provided"}), 400

        temp_dir = create_unique_temp_dir()
        input_path = os.path.join(temp_dir, 'input_image.jpg')
        args['image'].save(input_path)

        appFaceShine.logger.info(f"Image saved to {input_path}")

        image = data2image(open(input_path, 'rb').read())
        appFaceShine.logger.info("Removed the Face Scratched.")
        prediction = taskEraseScratches.executeTask(image)
        appFaceShine.logger.info("Face Scraches removed.")

        # print("Type of prediction:", type(prediction))
        # if isinstance(prediction, np.ndarray):
        #     print("Shape of prediction:", prediction.shape)
        
        if prediction.ndim == 2:  # Grayscale image
            prediction_image = Image.fromarray(prediction.astype(np.uint8), mode='L')
        elif prediction.ndim == 3 and prediction.shape[2] == 3:  # RGB Image
            prediction_image = Image.fromarray(prediction.astype(np.uint8), mode='RGB')
        else:
            appFaceShine.logger.error("Unexpected format for prediction")
            return jsonify({"error": "Unexpected format for prediction"})

        # Save the processed image
        output_filename = f"{uuid.uuid4()}.jpg"
        output_path = os.path.join('processed_images', output_filename)
        prediction_image.save(output_path)
        appFaceShine.logger.info(f"Processed image saved to {output_path}")

        return jsonify({"url": get_download_url(output_filename)})


class ImageColorizer(Resource):
    def post(self):
        appFaceShine.logger.info("Recieved request to remove scratches.")
        args = parser.parse_args()
        if not args['image']:
            appFaceShine.logger.error("No image provided in the request.")
            return jsonify({"error": "No image provided"}), 400

        temp_dir = create_unique_temp_dir()
        input_path = os.path.join(temp_dir, 'input_image.jpg')
        args['image'].save(input_path)

        appFaceShine.logger.info(f"Image saved to {input_path}")

        image = data2image(open(input_path, 'rb').read())
        appFaceShine.logger.info("Removed the Face Scratched.")
        prediction = taskImageColorizer.executeTask(image)
        appFaceShine.logger.info("Face Scraches removed.")

        # print("Type of prediction:", type(prediction))
        # if isinstance(prediction, np.ndarray):
        #     print("Shape of prediction:", prediction.shape)
        
        if prediction.ndim == 2:  # Grayscale image
            prediction_image = Image.fromarray(prediction.astype(np.uint8), mode='L')
        elif prediction.ndim == 3 and prediction.shape[2] == 3:  # RGB Image
            prediction_image = Image.fromarray(prediction.astype(np.uint8), mode='RGB')
        else:
            appFaceShine.logger.error("Unexpected format for prediction")
            return jsonify({"error": "Unexpected format for prediction"})

        # Save the processed image
        output_filename = f"{uuid.uuid4()}.jpg"
        output_path = os.path.join('processed_images', output_filename)
        prediction_image.save(output_path)
        appFaceShine.logger.info(f"Processed image saved to {output_path}")

        return jsonify({"url": get_download_url(output_filename)})


class ZeroBackground(Resource):
    def post(self):
        appFaceShine.logger.info("Received request for Removing Background.")
        args = parser.parse_args()
        if not args['image']:
            appFaceShine.logger.error("No image provided in the request.")
            return jsonify({"error": "No image provided"}), 400

        temp_dir = create_unique_temp_dir()
        input_path = os.path.join(temp_dir, 'input_image.jpg')
        args['image'].save(input_path)

        appFaceShine.logger.info(f"Image saved to {input_path}")

        image = data2image(open(input_path, 'rb').read())
        appFaceShine.logger.info("Running face segmentation task.")
        prediction = taskZeroBackground.executeTask(image)
        appFaceShine.logger.info("Face segmentation completed.")

        # print("Type of prediction:", type(prediction))
        # if isinstance(prediction, np.ndarray):
        #     print("Shape of prediction:", prediction.shape)
        
        if prediction.ndim == 2:  # Grayscale image
            prediction_image = Image.fromarray(prediction.astype(np.uint8), mode='L')
        elif prediction.ndim == 3 and prediction.shape[2] == 3:  # RGB Image
            prediction_image = Image.fromarray(prediction.astype(np.uint8), mode='RGB')
        else:
            appFaceShine.logger.error("Unexpected format for prediction")
            return jsonify({"error": "Unexpected format for prediction"})

        # Save the processed image
        output_filename = f"{uuid.uuid4()}.jpg"
        output_path = os.path.join('processed_images', output_filename)
        prediction_image.save(output_path)
        appFaceShine.logger.info(f"Processed image saved to {output_path}")

        return jsonify({"url": get_download_url(output_filename)})


api.add_resource(FaceSegmentation, '/segment_face')
api.add_resource(SuperFace, '/super_face')
api.add_resource(EraseScratches, '/erase_scratches')
api.add_resource(ImageLowLight, '/enhance_light')
api.add_resource(ImageColorizer, '/colorize')
api.add_resource(ZeroBackground, '/zero_background')
api.add_resource(Index, '/')



# if __name__ == '__main__':
#     appFaceShine.logger.info("Server is up and running")
#     appFaceShine.run(host='0.0.0.0',port=5000,debug=True)


# from flask import Flask, jsonify, request, send_from_directory
# from flask_restful import Resource, Api, reqparse
# from werkzeug.datastructures import FileStorage
# import os
# import uuid
# import logging
# from PIL import Image
# import numpy as np
# from logging.handlers import RotatingFileHandler

# from faceshine import data2image
# from faceshine.tasks import TaskZeroBackground, \
#     TaskImageColorizer, TaskLowLight, TaskEraseScratches, TaskSuperFace, TaskFaceSegmentation

# appFaceShine = Flask(__name__)
# api = Api(appFaceShine)

# taskFaceSegmentation = TaskFaceSegmentation()
# taskSuperFace = TaskSuperFace()
# taskLowLight = TaskLowLight()
# taskEraseScratches = TaskEraseScratches()
# taskImageColorizer = TaskImageColorizer()
# taskZeroBackground = TaskZeroBackground()

# # Configure the debug points
# #logging.basicConfig(level=logging.DEBUG)

# # Configure Logging
# log_handler = RotatingFileHandler('app.log', maxBytes=1000000, backupCount=3)
# log_handler.setLevel(logging.INFO)
# appFaceShine.logger.addHandler(log_handler)

# # Ensure directories exist
# def ensure_directories():
#     os.makedirs('processed_images', exist_ok=True)
#     os.makedirs('temp_files', exist_ok=True)
#     appFaceShine.logger.info("Directories ensured: 'processed_images' and 'temp_files'.")

# ensure_directories()

# # Utility function to create unique directories
# def create_unique_temp_dir():
#     unique_id = str(uuid.uuid4())
#     temp_dir = os.path.join('temp_files', unique_id)
#     os.makedirs(temp_dir, exist_ok=True)
#     appFaceShine.logger.info(f"Created temporary directory: {temp_dir}")
#     return temp_dir

# def get_download_url(filename):
#     base_url = os.getenv('BASE_URL', 'http://localhost:5000/')
#     return f"{base_url}download/{filename}"

# @appFaceShine.route('/download/<filename>')
# def download_file(filename):
#     try:
#         appFaceShine.logger.info(f"Attempting to send file: {filename}")
#         file_path = os.path.join('processed_images', filename)
#         if os.path.exists(file_path):
#             appFaceShine.logger.info(f"File {filename} exists. Sending file.")
#             return send_from_directory('processed_images', filename, as_attachment=True)
#         else:
#             appFaceShine.logger.error(f"File {filename} does not exist.")
#             return jsonify({"error": f"File {filename} does not exist"}), 404
#     except Exception as e:
#         appFaceShine.logger.error(f"Error sending file: {e}")
#         return jsonify({"error": f"Error sending file: {e}"}), 500

# parser = reqparse.RequestParser()
# parser.add_argument('image', type=FileStorage, location='files', required=False, help='Provide one image file')

# class Index(Resource):
#     def get(self):
#         return {'It works!': 'AI Remote Procedure Machine'}

# class FaceSegmentation(Resource):
#     def post(self):
#         appFaceShine.logger.info("Received request for face segmentation.")
#         args = parser.parse_args()
#         if not args['image']:
#             appFaceShine.logger.error("No image provided in the request.")
#             return jsonify({"error": "No image provided"}), 400

#         temp_dir = create_unique_temp_dir()
#         input_path = os.path.join(temp_dir, 'input_image.jpg')
#         args['image'].save(input_path)

#         appFaceShine.logger.info(f"Image saved to {input_path}")

#         image = data2image(open(input_path, 'rb').read())
#         appFaceShine.logger.info("Running face segmentation task.")
#         prediction = taskFaceSegmentation.executeTask(image)
#         appFaceShine.logger.info("Face segmentation completed.")

#         print("Type of prediction:", type(prediction))
#         if isinstance(prediction, np.ndarray):
#             print("Shape of prediction:", prediction.shape)
        
#         if prediction.ndim == 2:  # Grayscale image
#             prediction_image = Image.fromarray(prediction.astype(np.uint8), mode='L')
#         elif prediction.ndim == 3 and prediction.shape[2] == 3:  # RGB Image
#             prediction_image = Image.fromarray(prediction.astype(np.uint8), mode='RGB')
#         else:
#             appFaceShine.logger.error("Unexpected format for prediction")
#             return jsonify({"error": "Unexpected format for prediction"})

#         # Save the processed image
#         output_filename = f"{uuid.uuid4()}.jpg"
#         output_path = os.path.join('processed_images', output_filename)
#         prediction_image.save(output_path)
#         appFaceShine.logger.info(f"Processed image saved to {output_path}")

#         return jsonify({"url": get_download_url(output_filename)})

# api.add_resource(FaceSegmentation, '/segment_face')

# Start the Flask app
#if __name__ == '__main__':
#    appFaceShine.logger.info("Server is up and running")
#    appFaceShine.run(host='0.0.0.0', port=5000)

