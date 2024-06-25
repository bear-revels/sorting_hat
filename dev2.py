from ultralyticsplus import YOLO, render_result

# load model
model_path = 'best.pt'
model = YOLO(model_path)

# set model parameters
model.overrides['conf'] = 0.20  # NMS confidence threshold
model.overrides['iou'] = 0.40  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# set image
image = 'zidane.jpg'

# perform inference
results = model.predict(image)

# observe results
print(results[0].boxes)
render = render_result(model=model, image=image, result=results[0])
render.show()