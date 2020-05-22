import json
import os

def convert(bbox, size=(1024,1024)):
    xmin = int((2*bbox[0]-bbox[2])*size[0]//2)
    xmax = int((2*bbox[0]+bbox[2])*size[0]//2)
    ymin = int((2*bbox[1]-bbox[3])*size[1]//2)
    ymax = int((2*bbox[1]+bbox[3])*size[1]//2)
    return [xmin, ymin, xmax, ymax]


result_json_path = os.path.join("result_test_v4_lh.json")
output_path = './results_json_test_v4_lh'

if not os.path.exists(output_path):
    os.mkdir(output_path)

results = []
with open(result_json_path, 'r') as f:
    results = json.load(f)

'''
{'frame_id': 1, 'filename': '/home/wtx8887/darknet/XRAY/test_JPEGImages/39504.jpg', 'objects': [{'class_id': 8, 'name': 'Fracture', 'relative_coordinates': {'center_x': 0.438479, 'center_y': 0.163755, 'width': 0.111669, 'height': 0.084431}, 'confidence': 0.588105}]}
'''
for result in results:
    output = []
    filename = result['filename'].split('/')[-1]
    filename = filename.split('.')[0] + '.json'
    temp = os.path.join(output_path, filename)
    objects = result['objects']
    for obj in objects:
        obj_struct = {}
        bbox = [obj['relative_coordinates']['center_x'], 
                obj['relative_coordinates']['center_y'], 
                obj['relative_coordinates']['width'], 
                obj['relative_coordinates']['height']]
        value = convert(bbox)
        value.append(obj['confidence'])
        key = obj['name']
        obj_struct[key] = value
        output.append(obj_struct)
    
    f = open(temp, 'w')
    json.dump(output, f)