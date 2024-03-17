
# import a utility function for loading Roboflow models
from roboflow import Roboflow
# import HTTPClient for API verification
from inference_sdk import InferenceHTTPClient
# import supervision to visualize our results
import supervision as sv
# import OpenCV
import cv2


def image_detection(file_name):
    """
    Analyzes image for trash, copies the image and surrounds the highlighted trash with marker and prediction confidence
    then shows the image onscreen.

    :param file_name: Name of image file that you would like to analyze for trash (use .jpg in the same directory)
    :return: A json file with the prediction's class, coordinates and confidence. e.g. ({'predictions':
     [{'x': 388, 'y': 262, 'width': 758, 'height': 475, 'confidence': 0.4060649871826172,
    'class': 'garbage', 'class_id': 0, 'detection_id': 'c58b895d-6ecc-4a4d-9570-9103a5ae8ee8',
    'image_path': 'trash_image_7.jpg', 'prediction_type': 'ObjectDetectionModel'}],
    'image': {'width': '800', 'height': '500'}}
    """

    rf = Roboflow(api_key="UlcWc2CYeXZpIp0u5mzX")
    project = rf.workspace().project("garbage_detection-wvzwv")
    model = project.version(9).model

    result = model.predict(file_name, confidence=40, overlap=30).json()

    labels = [item["class"] for item in result["predictions"]]

    detections = sv.Detections.from_roboflow(result)

    label_annotator = sv.LabelAnnotator()
    bounding_box_annotator = sv.BoxAnnotator()

    image = cv2.imread(file_name)

    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    sv.plot_image(image=annotated_image)

    return result


print(image_detection('trash_image_7.jpg'))
