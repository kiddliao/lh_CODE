import os
import json

def convert(bbox, size=(1024,1024)):
    xmin = int((2*bbox[0]-bbox[2])*size[0]//2)
    xmax = int((2*bbox[0]+bbox[2])*size[0]//2)
    ymin = int((2*bbox[1]-bbox[3])*size[1]//2)
    ymax = int((2*bbox[1]+bbox[3])*size[1]//2)
    w = xmax - xmin
    h = ymax - ymin
    return [xmin, ymin, w, h]

result_json_path = os.path.join("result_wtx_hand_label.json")
output_path = 'result_wtx_hand_label_cocoshape.json'

with open(os.path.join('annotations', 'newinstances_val2017.json'), 'r') as f:
    empty_test = json.load(f)
img_map = {}
for i in range(len(empty_test['images'])):
    img_map[empty_test['images'][i]['file_name']] = empty_test['images'][i]['id']

results = []
outputs = []
with open(result_json_path, 'r') as f:
    results = json.load(f)

for result in results:
    image_id = result['filename'].split('/')[-1]
    for obj in result['objects']:
        obj_struct = {}
        bbox_temp = [obj['relative_coordinates']['center_x'], 
                obj['relative_coordinates']['center_y'], 
                obj['relative_coordinates']['width'], 
                obj['relative_coordinates']['height']]
        bbox = convert(bbox_temp)
        score = obj['confidence']
        category_id = obj['class_id']+1
        obj_struct['image_id'] = img_map[image_id]
        obj_struct['bbox'] = bbox
        obj_struct['score'] = score
        obj_struct['category_id'] = category_id
        outputs.append(obj_struct)

with open(output_path, 'w') as f:
    json.dump(outputs, f)

