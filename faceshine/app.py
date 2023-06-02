from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse

from faceshine import data2image
from .segmentation import TaskZeroBackground
from werkzeug.datastructures import FileStorage

appFaceShine = Flask(__name__)
api = Api(appFaceShine)

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


class ZeroBackground(Resource):
    def post(self):
        args = parser.parse_args()
        stream_a = args['image'].read()
        image = data2image(stream_a)
        prediction = taskZeroBackground.executeTask(image)
        return jsonify(prediction.tolist())


api.add_resource(ZeroBackground, '/zero_background')
api.add_resource(Index, '/')
