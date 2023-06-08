from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage

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
        args = parser.parse_args()
        stream = args['image'].read()
        image = data2image(stream)
        prediction = taskFaceSegmentation.executeTask(image)
        return jsonify(prediction.tolist())


class SuperFace(Resource):
    def post(self):
        args = parser.parse_args()
        stream_a = args['image'].read()
        image = data2image(stream_a)
        prediction = taskSuperFace.executeTask(image)
        return jsonify(prediction.tolist())


class ImageLowLight(Resource):
    def post(self):
        args = parser.parse_args()
        stream_a = args['image'].read()
        image = data2image(stream_a)
        prediction = taskLowLight.executeTask(image)
        return jsonify(prediction.tolist())


class EraseScratches(Resource):
    def post(self):
        args = parser.parse_args()
        stream_a = args['image'].read()
        image = data2image(stream_a)
        prediction = taskEraseScratches.executeTask(image)
        return jsonify(prediction.tolist())


class ImageColorizer(Resource):
    def post(self):
        args = parser.parse_args()
        stream_a = args['image'].read()
        image = data2image(stream_a)
        prediction = taskImageColorizer.executeTask(image)
        return jsonify(prediction.tolist())


class ZeroBackground(Resource):
    def post(self):
        args = parser.parse_args()
        stream_a = args['image'].read()
        image = data2image(stream_a)
        prediction = taskZeroBackground.executeTask(image)
        return jsonify(prediction.tolist())


api.add_resource(FaceSegmentation, '/segment_face')
api.add_resource(SuperFace, '/super_face')
api.add_resource(EraseScratches, '/erase_scratches')
api.add_resource(ImageLowLight, '/enhance_light')
api.add_resource(ImageColorizer, '/colorize')
api.add_resource(ZeroBackground, '/zero_background')
api.add_resource(Index, '/')
