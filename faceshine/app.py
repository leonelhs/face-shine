###########################
#  ASYNCIO and WEBSOCKETS #
###########################


























#########################################
#  CELERY WORKER RUNNINGN ON TASK DEMAND#
#########################################



from flask import Flask, jsonify, request, send_from_directory
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage
import os
import uuid
import logging
from PIL import Image
import numpy as np
from logging.handlers import RotatingFileHandler
import redis
from faceshine import data2image
from faceshine.tasks import TaskZeroBackground, TaskImageColorizer, TaskLowLight, TaskEraseScratches, TaskSuperFace, TaskFaceSegmentation
from faceshine.celery_app import celery, create_app

# Initialize Flask app
appFaceShine = create_app()
api = Api(appFaceShine)

# Initialize tasks
taskFaceSegmentation = TaskFaceSegmentation()
taskSuperFace = TaskSuperFace()
taskLowLight = TaskLowLight()
taskEraseScratches = TaskEraseScratches()
taskImageColorizer = TaskImageColorizer()
taskZeroBackground = TaskZeroBackground()

# Configure Redis
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
queue_name = 'task_queue'

# Configure Logging
logging.basicConfig(level=logging.DEBUG)
log_handler = RotatingFileHandler('app.log', maxBytes=1000000, backupCount=3)
log_handler.setLevel(logging.INFO)
appFaceShine.logger.addHandler(log_handler)

os.makedirs('processed_images', exist_ok=True)
os.makedirs('temp_files', exist_ok=True)

# Utility function to create unique directories
def create_unique_temp_dir():
    unique_id = str(uuid.uuid4())
    temp_dir = os.path.join('temp_files', unique_id)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    return temp_dir

def get_download_url(filename):
    base_url = os.getenv('BASE_URL', 'http://localhost:5000/')
    return f"{base_url}download/{filename}"

@appFaceShine.route('/download/<filename>')
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
parser.add_argument('image', type=FileStorage, location='files', required=False, help='Provide one image file')

class Index(Resource):
    def get(self):
        return {'It works!': 'AI Remote Procedure Machine'}

class SegmentFace(Resource):
    def post(self):
        task_name = 'segment_face'
        appFaceShine.logger.info(f"Received request for {task_name}.")

        client_id = request.remote_addr
        ongoing_task = redis_client.get(f"ongoing_task:{client_id}")
        if ongoing_task is not None:
            appFaceShine.logger.info(f"Client {client_id} already has an ongoing task.")
            return jsonify({"error": "A task is already in progress. Please wait for it to complete."}), 400

        args = parser.parse_args()
        if not args['image']:
            appFaceShine.logger.error("No image provided in the request.")
            return jsonify({"error": "No image provided"}), 400

        temp_dir = create_unique_temp_dir()
        input_path = os.path.join(temp_dir, 'input_image.jpg')
        args['image'].save(input_path)
        appFaceShine.logger.info(f"Image saved to {input_path}")

        task_id = str(uuid.uuid4())
        task_data = {
            "task_id": task_id,
            "task_name": task_name,
            "input_path": input_path,
            "client_id": client_id
        }
        redis_client.lpush(queue_name, str(task_data))
        redis_client.set(f"ongoing_task:{client_id}", task_id)
        appFaceShine.logger.info(f"Task {task_id} added to queue for {task_name}.")

        process_image_task.apply_async(args=[task_data])  # Directly add the task to Celery

        return jsonify({"task_id": task_id, "status": "queued"})

class SuperFace(Resource):
    def post(self):
        task_name = 'super_face'
        appFaceShine.logger.info(f"Received request for {task_name}.")

        client_id = request.remote_addr
        ongoing_task = redis_client.get(f"ongoing_task:{client_id}")
        if ongoing_task is not None:
            appFaceShine.logger.info(f"Client {client_id} already has an ongoing task.")
            return jsonify({"error": "A task is already in progress. Please wait for it to complete."}), 400

        args = parser.parse_args()
        if not args['image']:
            appFaceShine.logger.error("No image provided in the request.")
            return jsonify({"error": "No image provided"}), 400

        temp_dir = create_unique_temp_dir()
        input_path = os.path.join(temp_dir, 'input_image.jpg')
        args['image'].save(input_path)
        appFaceShine.logger.info(f"Image saved to {input_path}")

        task_id = str(uuid.uuid4())
        task_data = {
            "task_id": task_id,
            "task_name": task_name,
            "input_path": input_path,
            "client_id": client_id
        }
        redis_client.lpush(queue_name, str(task_data))
        redis_client.set(f"ongoing_task:{client_id}", task_id)
        appFaceShine.logger.info(f"Task {task_id} added to queue for {task_name}.")

        process_image_task.apply_async(args=[task_data])  # Directly add the task to Celery

        return jsonify({"task_id": task_id, "status": "queued"})

class EraseScratches(Resource):
    def post(self):
        task_name = 'erase_scratches'
        appFaceShine.logger.info(f"Received request for {task_name}.")

        client_id = request.remote_addr
        ongoing_task = redis_client.get(f"ongoing_task:{client_id}")
        if ongoing_task is not None:
            appFaceShine.logger.info(f"Client {client_id} already has an ongoing task.")
            return jsonify({"error": "A task is already in progress. Please wait for it to complete."}), 400

        args = parser.parse_args()
        if not args['image']:
            appFaceShine.logger.error("No image provided in the request.")
            return jsonify({"error": "No image provided"}), 400

        temp_dir = create_unique_temp_dir()
        input_path = os.path.join(temp_dir, 'input_image.jpg')
        args['image'].save(input_path)
        appFaceShine.logger.info(f"Image saved to {input_path}")

        task_id = str(uuid.uuid4())
        task_data = {
            "task_id": task_id,
            "task_name": task_name,
            "input_path": input_path,
            "client_id": client_id
        }
        redis_client.lpush(queue_name, str(task_data))
        redis_client.set(f"ongoing_task:{client_id}", task_id)
        appFaceShine.logger.info(f"Task {task_id} added to queue for {task_name}.")

        process_image_task.apply_async(args=[task_data])  # Directly add the task to Celery

        return jsonify({"task_id": task_id, "status": "queued"})

class Colorize(Resource):
    def post(self):
        task_name = 'colorize'
        appFaceShine.logger.info(f"Received request for {task_name}.")

        client_id = request.remote_addr
        ongoing_task = redis_client.get(f"ongoing_task:{client_id}")
        if ongoing_task is not None:
            appFaceShine.logger.info(f"Client {client_id} already has an ongoing task.")
            return jsonify({"error": "A task is already in progress. Please wait for it to complete."}), 400

        args = parser.parse_args()
        if not args['image']:
            appFaceShine.logger.error("No image provided in the request.")
            return jsonify({"error": "No image provided"}), 400

        temp_dir = create_unique_temp_dir()
        input_path = os.path.join(temp_dir, 'input_image.jpg')
        args['image'].save(input_path)
        appFaceShine.logger.info(f"Image saved to {input_path}")

        task_id = str(uuid.uuid4())
        task_data = {
            "task_id": task_id,
            "task_name": task_name,
            "input_path": input_path,
            "client_id": client_id
        }
        redis_client.lpush(queue_name, str(task_data))
        redis_client.set(f"ongoing_task:{client_id}", task_id)
        appFaceShine.logger.info(f"Task {task_id} added to queue for {task_name}.")

        process_image_task.apply_async(args=[task_data])  # Directly add the task to Celery

        return jsonify({"task_id": task_id, "status": "queued"})

class ZeroBackground(Resource):
    def post(self):
        task_name = 'zero_background'
        appFaceShine.logger.info(f"Received request for {task_name}.")

        client_id = request.remote_addr
        ongoing_task = redis_client.get(f"ongoing_task:{client_id}")
        if ongoing_task is not None:
            appFaceShine.logger.info(f"Client {client_id} already has an ongoing task.")
            return jsonify({"error": "A task is already in progress. Please wait for it to complete."}), 400

        args = parser.parse_args()
        if not args['image']:
            appFaceShine.logger.error("No image provided in the request.")
            return jsonify({"error": "No image provided"}), 400

        temp_dir = create_unique_temp_dir()
        input_path = os.path.join(temp_dir, 'input_image.jpg')
        args['image'].save(input_path)
        appFaceShine.logger.info(f"Image saved to {input_path}")

        task_id = str(uuid.uuid4())
        task_data = {
            "task_id": task_id,
            "task_name": task_name,
            "input_path": input_path,
            "client_id": client_id
        }
        redis_client.lpush(queue_name, str(task_data))
        redis_client.set(f"ongoing_task:{client_id}", task_id)
        appFaceShine.logger.info(f"Task {task_id} added to queue for {task_name}.")

        process_image_task.apply_async(args=[task_data])  # Directly add the task to Celery

        return jsonify({"task_id": task_id, "status": "queued"})

class TaskStatus(Resource):
    def get(self, task_id):
        task_status = redis_client.get(f"task_status:{task_id}")
        if not task_status:
            return jsonify({"error": "Task ID not found"}), 404

        task_status = task_status.decode('utf-8')
        if task_status == "completed":
            task_result = redis_client.get(f"task_result:{task_id}").decode('utf-8')
            return jsonify({"task_id": task_id, "status": task_status, "url": task_result})
        else:
            return jsonify({"task_id": task_id, "status": task_status})

api.add_resource(Index, '/')
api.add_resource(SegmentFace, '/segment_face')
api.add_resource(SuperFace, '/super_face')
api.add_resource(EraseScratches, '/erase_scratches')
api.add_resource(Colorize, '/colorize')
api.add_resource(ZeroBackground, '/zero_background')
api.add_resource(TaskStatus, '/task_status/<task_id>')

@celery.task(bind=True)
def process_image_task(self, task_data):
    task_data = eval(task_data[1].decode('utf-8'))  # Decode and evaluate the task_data
    task_id = task_data['task_id']
    task_name = task_data['task_name']
    input_path = task_data['input_path']
    client_id = task_data['client_id']
    
    try:
        image = data2image(open(input_path, 'rb').read())
        if task_name == 'segment_face':
            result_image = taskFaceSegmentation.executeTask(image)
        elif task_name == 'super_face':
            result_image = taskSuperFace.executeTask(image)
        elif task_name == 'enhance_light':
            result_image = taskLowLight.executeTask(image)
        elif task_name == 'erase_scratches':
            result_image = taskEraseScratches.executeTask(image)
        elif task_name == 'colorize':
            result_image = taskImageColorizer.executeTask(image)
        elif task_name == 'zero_background':
            result_image = taskZeroBackground.executeTask(image)
        else:
            raise ValueError(f"Unknown task: {task_name}")
        
        # Save the processed image
        if result_image.ndim == 2:  # Grayscale image
            result_image = Image.fromarray(result_image.astype(np.uint8), mode='L')
        elif result_image.ndim == 3 and result_image.shape[2] == 3:  # RGB Image
            result_image = Image.fromarray(result_image.astype(np.uint8), mode='RGB')
        else:
            raise ValueError("Unexpected format for result image")
        
        output_filename = f"{uuid.uuid4()}.jpg"
        output_path = os.path.join('processed_images', output_filename)
        result_image.save(output_path)
        appFaceShine.logger.info(f"Task {task_id} completed and saved to {output_path}")

        # Update Redis with the completed status
        download_url = get_download_url(output_filename)
        redis_client.set(f"task_status:{task_id}", "completed")
        redis_client.set(f"task_result:{task_id}", download_url)
        redis_client.delete(f"ongoing_task:{client_id}")
        redis_client.lpush('completed_tasks', f"{task_name}:{download_url}")  # Store completed task info

    except Exception as e:
        appFaceShine.logger.error(f"Error processing task {task_id}: {e}")
        redis_client.set(f"task_status:{task_id}", "error")
        redis_client.delete(f"ongoing_task:{client_id}")

@appFaceShine.route('/completed_tasks', methods=['GET'])
def get_completed_tasks():
    completed_tasks = []
    while redis_client.llen('completed_tasks') > 0:
        task_info = redis_client.rpop('completed_tasks').decode('utf-8')
        task_name, download_url = task_info.split(':')
        completed_tasks.append({"task_name": task_name, "download_url": download_url})
    return jsonify(completed_tasks)






########################################
#  CELERY WORKER RUNNINGN CONTINOUSLY  #
########################################


# from flask import Flask, jsonify, request, send_from_directory
# from flask_restful import Resource, Api, reqparse
# from werkzeug.datastructures import FileStorage
# import os
# import uuid
# import logging
# from PIL import Image
# import numpy as np
# from logging.handlers import RotatingFileHandler
# import redis
# from faceshine import data2image
# from faceshine.tasks import TaskZeroBackground, TaskImageColorizer, TaskLowLight, TaskEraseScratches, TaskSuperFace, TaskFaceSegmentation
# from faceshine.celery_app import celery,create_app

# # Initialize Flask app
# appFaceShine = create_app()
# api = Api(appFaceShine)

# # Initialize tasks
# taskFaceSegmentation = TaskFaceSegmentation()
# taskSuperFace = TaskSuperFace()
# taskLowLight = TaskLowLight()
# taskEraseScratches = TaskEraseScratches()
# taskImageColorizer = TaskImageColorizer()
# taskZeroBackground = TaskZeroBackground()

# # Configure Redis
# redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
# queue_name = 'task_queue'

# # Configure Celery
# #appFaceShine.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
# #appFaceShine.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

# # celery = Celery(appFaceShine.name, broker=appFaceShine.config['CELERY_BROKER_URL'])
# # celery.conf.update(appFaceShine.config)

# # Configure Logging
# logging.basicConfig(level=logging.DEBUG)
# log_handler = RotatingFileHandler('app.log', maxBytes=1000000, backupCount=3)
# log_handler.setLevel(logging.INFO)
# appFaceShine.logger.addHandler(log_handler)

# os.makedirs('processed_images', exist_ok=True)
# os.makedirs('temp_files', exist_ok=True)

# # Utility function to create unique directories
# def create_unique_temp_dir():
#     unique_id = str(uuid.uuid4())
#     temp_dir = os.path.join('temp_files', unique_id)
#     if not os.path.exists(temp_dir):
#         os.makedirs(temp_dir)
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

# class SegmentFace(Resource):
#     def post(self):
#         task_name = 'segment_face'
#         appFaceShine.logger.info(f"Received request for {task_name}.")

#         client_id = request.remote_addr
#         ongoing_task = redis_client.get(f"ongoing_task:{client_id}")
#         if ongoing_task is not None:
#             appFaceShine.logger.info(f"Client {client_id} already has an ongoing task.")
#             return jsonify({"error": "A task is already in progress. Please wait for it to complete."}), 400

#         args = parser.parse_args()
#         if not args['image']:
#             appFaceShine.logger.error("No image provided in the request.")
#             return jsonify({"error": "No image provided"}), 400

#         temp_dir = create_unique_temp_dir()
#         input_path = os.path.join(temp_dir, 'input_image.jpg')
#         args['image'].save(input_path)
#         appFaceShine.logger.info(f"Image saved to {input_path}")

#         task_id = str(uuid.uuid4())
#         task_data = {
#             "task_id": task_id,
#             "task_name": task_name,
#             "input_path": input_path,
#             "client_id": client_id
#         }
#         redis_client.lpush(queue_name, str(task_data))
#         redis_client.set(f"ongoing_task:{client_id}", task_id)
#         appFaceShine.logger.info(f"Task {task_id} added to queue for {task_name}.")

#         return jsonify({"task_id": task_id, "status": "queued"})


# class SuperFace(Resource):
#     def post(self):
#         task_name = 'super_face'
#         appFaceShine.logger.info(f"Received request for {task_name}.")

#         client_id = request.remote_addr
#         ongoing_task = redis_client.get(f"ongoing_task:{client_id}")
#         if ongoing_task is not None:
#             appFaceShine.logger.info(f"Client {client_id} already has an ongoing task.")
#             return jsonify({"error": "A task is already in progress. Please wait for it to complete."}), 400

#         args = parser.parse_args()
#         if not args['image']:
#             appFaceShine.logger.error("No image provided in the request.")
#             return jsonify({"error": "No image provided"}), 400

#         temp_dir = create_unique_temp_dir()
#         input_path = os.path.join(temp_dir, 'input_image.jpg')
#         args['image'].save(input_path)
#         appFaceShine.logger.info(f"Image saved to {input_path}")

#         task_id = str(uuid.uuid4())
#         task_data = {
#             "task_id": task_id,
#             "task_name": task_name,
#             "input_path": input_path,
#             "client_id": client_id
#         }
#         redis_client.lpush(queue_name, str(task_data))
#         redis_client.set(f"ongoing_task:{client_id}", task_id)
#         appFaceShine.logger.info(f"Task {task_id} added to queue for {task_name}.")

#         return jsonify({"task_id": task_id, "status": "queued"})

# class EraseScratches(Resource):
#     def post(self):
#         task_name = 'erase_scratches'
#         appFaceShine.logger.info(f"Received request for {task_name}.")

#         client_id = request.remote_addr
#         ongoing_task = redis_client.get(f"ongoing_task:{client_id}")
#         if ongoing_task is not None:
#             appFaceShine.logger.info(f"Client {client_id} already has an ongoing task.")
#             return jsonify({"error": "A task is already in progress. Please wait for it to complete."}), 400

#         args = parser.parse_args()
#         if not args['image']:
#             appFaceShine.logger.error("No image provided in the request.")
#             return jsonify({"error": "No image provided"}), 400

#         temp_dir = create_unique_temp_dir()
#         input_path = os.path.join(temp_dir, 'input_image.jpg')
#         args['image'].save(input_path)
#         appFaceShine.logger.info(f"Image saved to {input_path}")

#         task_id = str(uuid.uuid4())
#         task_data = {
#             "task_id": task_id,
#             "task_name": task_name,
#             "input_path": input_path,
#             "client_id": client_id
#         }
#         redis_client.lpush(queue_name, str(task_data))
#         redis_client.set(f"ongoing_task:{client_id}", task_id)
#         appFaceShine.logger.info(f"Task {task_id} added to queue for {task_name}.")

#         return jsonify({"task_id": task_id, "status": "queued"})

# class Colorize(Resource):
#     def post(self):
#         task_name = 'colorize'
#         appFaceShine.logger.info(f"Received request for {task_name}.")

#         client_id = request.remote_addr
#         ongoing_task = redis_client.get(f"ongoing_task:{client_id}")
#         if ongoing_task is not None:
#             appFaceShine.logger.info(f"Client {client_id} already has an ongoing task.")
#             return jsonify({"error": "A task is already in progress. Please wait for it to complete."}), 400

#         args = parser.parse_args()
#         if not args['image']:
#             appFaceShine.logger.error("No image provided in the request.")
#             return jsonify({"error": "No image provided"}), 400

#         temp_dir = create_unique_temp_dir()
#         input_path = os.path.join(temp_dir, 'input_image.jpg')
#         args['image'].save(input_path)
#         appFaceShine.logger.info(f"Image saved to {input_path}")

#         task_id = str(uuid.uuid4())
#         task_data = {
#             "task_id": task_id,
#             "task_name": task_name,
#             "input_path": input_path,
#             "client_id": client_id
#         }
#         redis_client.lpush(queue_name, str(task_data))
#         redis_client.set(f"ongoing_task:{client_id}", task_id)
#         appFaceShine.logger.info(f"Task {task_id} added to queue for {task_name}.")

#         return jsonify({"task_id": task_id, "status": "queued"})

# class ZeroBackground(Resource):
#     def post(self):
#         task_name = 'zero_background'
#         appFaceShine.logger.info(f"Received request for {task_name}.")

#         client_id = request.remote_addr
#         ongoing_task = redis_client.get(f"ongoing_task:{client_id}")
#         if ongoing_task is not None:
#             appFaceShine.logger.info(f"Client {client_id} already has an ongoing task.")
#             return jsonify({"error": "A task is already in progress. Please wait for it to complete."}), 400

#         args = parser.parse_args()
#         if not args['image']:
#             appFaceShine.logger.error("No image provided in the request.")
#             return jsonify({"error": "No image provided"}), 400

#         temp_dir = create_unique_temp_dir()
#         input_path = os.path.join(temp_dir, 'input_image.jpg')
#         args['image'].save(input_path)
#         appFaceShine.logger.info(f"Image saved to {input_path}")

#         task_id = str(uuid.uuid4())
#         task_data = {
#             "task_id": task_id,
#             "task_name": task_name,
#             "input_path": input_path,
#             "client_id": client_id
#         }
#         redis_client.lpush(queue_name, str(task_data))
#         redis_client.set(f"ongoing_task:{client_id}", task_id)
#         appFaceShine.logger.info(f"Task {task_id} added to queue for {task_name}.")

#         return jsonify({"task_id": task_id, "status": "queued"})



# class TaskStatus(Resource):
#     def get(self, task_id):
#         task_status = redis_client.get(f"task_status:{task_id}")
#         if not task_status:
#             return jsonify({"error": "Task ID not found"}), 404

#         task_status = task_status.decode('utf-8')
#         if task_status == "completed":
#             task_result = redis_client.get(f"task_result:{task_id}").decode('utf-8')
#             return jsonify({"task_id": task_id, "status": task_status, "url": task_result})
#         else:
#             return jsonify({"task_id": task_id, "status": task_status})

# api.add_resource(Index, '/')
# api.add_resource(SegmentFace, '/segment_face')
# api.add_resource(SuperFace, '/super_face')
# api.add_resource(EraseScratches, '/erase_scratches')
# api.add_resource(Colorize, '/colorize')
# api.add_resource(ZeroBackground, '/zero_background')
# api.add_resource(TaskStatus, '/task_status/<task_id>')



# @celery.task(bind=True)
# def process_image_task(self, task_data):
#     task_data = eval(task_data[1].decode('utf-8'))  # Decode and evaluate the task_data
#     task_id = task_data['task_id']
#     task_name = task_data['task_name']
#     input_path = task_data['input_path']
#     client_id = task_data['client_id']
    
#     try:
#         image = data2image(open(input_path, 'rb').read())
#         if task_name == 'segment_face':
#             result_image = taskFaceSegmentation.executeTask(image)
#         elif task_name == 'super_face':
#             result_image = taskSuperFace.executeTask(image)
#         elif task_name == 'enhance_light':
#             result_image = taskLowLight.executeTask(image)
#         elif task_name == 'erase_scratches':
#             result_image = taskEraseScratches.executeTask(image)
#         elif task_name == 'colorize':
#             result_image = taskImageColorizer.executeTask(image)
#         elif task_name == 'zero_background':
#             result_image = taskZeroBackground.executeTask(image)
#         else:
#             raise ValueError(f"Unknown task: {task_name}")
        
#         # Save the processed image
#         if result_image.ndim == 2:  # Grayscale image
#             result_image = Image.fromarray(result_image.astype(np.uint8), mode='L')
#         elif result_image.ndim == 3 and result_image.shape[2] == 3:  # RGB Image
#             result_image = Image.fromarray(result_image.astype(np.uint8), mode='RGB')
#         else:
#             raise ValueError("Unexpected format for result image")
        
#         output_filename = f"{uuid.uuid4()}.jpg"
#         output_path = os.path.join('processed_images', output_filename)
#         result_image.save(output_path)
#         appFaceShine.logger.info(f"Task {task_id} completed and saved to {output_path}")

#         # Update Redis with the completed status
#         download_url = get_download_url(output_filename)
#         redis_client.set(f"task_status:{task_id}", "completed")
#         redis_client.set(f"task_result:{task_id}", download_url)
#         redis_client.delete(f"ongoing_task:{client_id}")
#         redis_client.lpush('completed_tasks', f"{task_name}:{download_url}")  # Store completed task info

#     except Exception as e:
#         appFaceShine.logger.error(f"Error processing task {task_id}: {e}")
#         redis_client.set(f"task_status:{task_id}", "error")
#         redis_client.delete(f"ongoing_task:{client_id}")

# # Background worker to process tasks from the queue
# @celery.task(bind=True)
# def background_worker(self):
#     while True:
#         task_data = redis_client.brpop(queue_name)
#         if task_data:
#             process_image_task.apply_async(args=[task_data])

# # Start the background worker
# def start_background_worker():
#     background_worker.apply_async()


# @appFaceShine.route('/completed_tasks', methods=['GET'])
# def get_completed_tasks():
#     completed_tasks = []
#     while redis_client.llen('completed_tasks') > 0:
#         task_info = redis_client.rpop('completed_tasks').decode('utf-8')
#         task_name, download_url = task_info.split(':')
#         completed_tasks.append({"task_name": task_name, "download_url": download_url})
#     return jsonify(completed_tasks)

