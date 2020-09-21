#coding:utf-8
from __future__ import print_function
import argparse, json, os, itertools, random, shutil, re


parser = argparse.ArgumentParser()

# Inputs
parser.add_argument('--input_scene_0hop_dir', default='../output/test/moformat_scenes_0hop',
    help="")
parser.add_argument('--input_question_1hop_file', default='../output/test/questions_1hop.json',
    help="")
parser.add_argument('--output_question_0hop_file', default='../output/test/questions_0hop.json',
    help="JSON file containing metadata about functions")
parser.add_argument('--output_question_01hop_file', default='../output/test/questions_01hop.json',
    help="JSON file containing metadata about functions")

COLORS = set(['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow'])
SIZES = set(['large', 'small'])
SHAPES = set(['cube', 'cylinder', 'sphere'])
MATERIALS = set(['metal', 'rubber'])
SYNS = {'ball':'sphere',
        'block':'cube',
        'big':'large',
        'tiny':'small',
        'matte':'rubber',
        'metallic':'metal',
        'shiny':'metal'}

def create_question_01hop_file(input_question_1hop_file, input_question_0hop_file, output_question_01hop_file):
    with open(input_question_1hop_file, 'r') as f:
        question_1hop = json.load(f)
    with open(input_question_0hop_file, 'r') as f:
        question_0hop = json.load(f)
    question_0hop['questions'] = question_0hop['questions'] + question_1hop['questions']
    with open(output_question_01hop_file, 'w') as fw:
        json.dump(question_0hop, fw)

def collect_questions_0hop(input_scene_0hop_dir, output_question_0hop_file): 
    # 1 hop question.json
    # 0 hop question.json
    # 01hop question.json

    synonyms_json = 'synonyms.json'
    with open(synonyms_json, 'r') as f:
        synonyms = json.load(f)

    questions = []
    for filename in os.listdir(input_scene_0hop_dir):
        scene_file = os.path.join(input_scene_0hop_dir, filename)
        with open(scene_file, 'r') as f:
            scene = json.load(f)
        obj_num = len(scene['objects'])
        for q,a in zip(scene['questions'], scene['answer']):
            program = []
            program.append({u'inputs': [], u'_output': range(obj_num), u'type': u'scene', u'value_inputs': []})
            inputs_idx = 0
            prevword = ''
            for word in re.split('[.? ]', q):
                if word == 'big' and prevword == 'How':
                    continue
                if word in SYNS:
                    # print (word, SYNS[word])
                    word = SYNS[word]
                if word in COLORS:
                    outputs = []
                    for obj_idx,obj in enumerate(scene['objects']):
                        if obj['color'] == word and obj_idx in program[-1]['_output']:
                            outputs.append(obj_idx)
                    # if len(outputs) == 0:
                    #     outputs = [program[-1]['_output'][0]]
                    program.append({'type': 'filter_color',
                                    'value_inputs': [word],
                                    'inputs':[inputs_idx],
                                    '_output':outputs})
                    inputs_idx += 1
                if word in SIZES:
                    outputs = []
                    for obj_idx,obj in enumerate(scene['objects']):
                        if obj['size'] == word and obj_idx in program[-1]['_output']:
                            outputs.append(obj_idx)
                    # if len(outputs) == 0:
                    #     outputs = [program[-1]['_output'][0]]
                    program.append({'type': 'filter_size',
                                    'value_inputs': [word],
                                    'inputs':[inputs_idx],
                                    '_output':outputs})
                    inputs_idx += 1
                if word in SHAPES:
                    outputs = []
                    for obj_idx,obj in enumerate(scene['objects']):
                        if obj['shape'] == word and obj_idx in program[-1]['_output']:
                            outputs.append(obj_idx)

                    program.append({'type': 'filter_shape',
                                    'value_inputs': [word],
                                    'inputs':[inputs_idx],
                                    '_output':outputs})
                    inputs_idx += 1
                if word in MATERIALS:
                    outputs = []
                    for obj_idx,obj in enumerate(scene['objects']):
                        if obj['material'] == word and obj_idx in program[-1]['_output']:
                            outputs.append(obj_idx)

                    program.append({'type': 'filter_material',
                                    'value_inputs': [word],
                                    'inputs':[inputs_idx],
                                    '_output':outputs})
                    inputs_idx += 1
                prevword = word
            if len(program[-1]['_output']) < 1:
                print(scene['objects'])
                print (q, program)
                print (dfdfd)
            program.append({u'inputs': [2], u'_output': program[-1]['_output'][0], u'type': u'unique', u'value_inputs': []})
            
            if a in COLORS:
                program.append({u'inputs': [3], u'_output': a, u'type': u'query_color', u'value_inputs': []})
            if a in SIZES:
                program.append({u'inputs': [3], u'_output': a, u'type': u'query_size', u'value_inputs': []})
            if a in SHAPES:
                program.append({u'inputs': [3], u'_output': a, u'type': u'query_shape', u'value_inputs': []})
            if a in MATERIALS:
                program.append({u'inputs': [3], u'_output': a, u'type': u'query_material', u'value_inputs': []})

            questions.append({
                'question':q,
                'answer':a,
                'image_index':scene['image_index'],
                'image_filename':scene['image_filename'],
                'program': program
            })
            # print(questions[-1])
        # break
    with open(output_question_0hop_file, 'w') as fw:
        json.dump({'questions':questions}, fw)

def b4_func(): 
    # scene_dir = '/raid/dataset/multiple-objects-gan/data/clevr/train_512/scenes_single_object/'
    scene_dir = '/raid/dataset/multiple-objects-gan/data/clevr/test_512/scenes_single_object_full_information4/'
    output_question_json = '/raid/dataset/multiple-objects-gan/data/clevr/test_512/test_questions_single_object_full_information4.json'

    synonyms_json = 'synonyms.json'
    with open(synonyms_json, 'r') as f:
        synonyms = json.load(f)

    questions = []
    for filename in os.listdir(scene_dir):
        scene_file = os.path.join(scene_dir, filename)
        with open(scene_file, 'r') as f:
            scene = json.load(f)
        obj_num = len(scene['objects'])
        for q,a in zip(scene['questions'], scene['answer']):
            program = []
            program.append({u'inputs': [], u'_output': range(obj_num), u'type': u'scene', u'value_inputs': []})
            inputs_idx = 0
            prevword = ''
            for word in re.split('[.? ]', q):
                if word == 'big' and prevword == 'How':
                    continue
                if word in SYNS:
                    # print (word, SYNS[word])
                    word = SYNS[word]
                if word in COLORS:
                    outputs = []
                    for obj_idx,obj in enumerate(scene['objects']):
                        if obj['color'] == word and obj_idx in program[-1]['_output']:
                            outputs.append(obj_idx)
                    # if len(outputs) == 0:
                    #     outputs = [program[-1]['_output'][0]]
                    program.append({'type': 'filter_color',
                                    'value_inputs': [word],
                                    'inputs':[inputs_idx],
                                    '_output':outputs})
                    inputs_idx += 1
                if word in SIZES:
                    outputs = []
                    for obj_idx,obj in enumerate(scene['objects']):
                        if obj['size'] == word and obj_idx in program[-1]['_output']:
                            outputs.append(obj_idx)
                    # if len(outputs) == 0:
                    #     outputs = [program[-1]['_output'][0]]
                    program.append({'type': 'filter_size',
                                    'value_inputs': [word],
                                    'inputs':[inputs_idx],
                                    '_output':outputs})
                    inputs_idx += 1
                if word in SHAPES:
                    outputs = []
                    for obj_idx,obj in enumerate(scene['objects']):
                        if obj['shape'] == word and obj_idx in program[-1]['_output']:
                            outputs.append(obj_idx)

                    program.append({'type': 'filter_shape',
                                    'value_inputs': [word],
                                    'inputs':[inputs_idx],
                                    '_output':outputs})
                    inputs_idx += 1
                if word in MATERIALS:
                    outputs = []
                    for obj_idx,obj in enumerate(scene['objects']):
                        if obj['material'] == word and obj_idx in program[-1]['_output']:
                            outputs.append(obj_idx)

                    program.append({'type': 'filter_material',
                                    'value_inputs': [word],
                                    'inputs':[inputs_idx],
                                    '_output':outputs})
                    inputs_idx += 1
                prevword = word
            if len(program[-1]['_output']) < 1:
                print(scene['objects'])
                print (q, program)
                print (dfdfd)
            program.append({u'inputs': [3], u'_output': program[-1]['_output'][0], u'type': u'unique', u'value_inputs': []})
            
            if a in COLORS:
                program.append({u'inputs': [4], u'_output': a, u'type': u'query_color', u'value_inputs': []})
            if a in SIZES:
                program.append({u'inputs': [4], u'_output': a, u'type': u'query_size', u'value_inputs': []})
            if a in SHAPES:
                program.append({u'inputs': [4], u'_output': a, u'type': u'query_shape', u'value_inputs': []})
            if a in MATERIALS:
                program.append({u'inputs': [4], u'_output': a, u'type': u'query_material', u'value_inputs': []})

            questions.append({
                'question':q,
                'answer':a,
                'image_index':scene['image_index'],
                'image_filename':scene['image_filename'],
                'program': program
            })
            # print(questions[-1])
        # break
    with open(output_question_json, 'w') as fw:
        json.dump({'questions':questions}, fw)

if __name__ == '__main__':
    args = parser.parse_args()
    collect_questions_0hop(args.input_scene_0hop_dir, args.output_question_0hop_file)
    create_question_01hop_file(args.input_question_1hop_file, args.output_question_0hop_file, args.output_question_01hop_file)
    # with open('../output/train_512/CLEVR_questions_single_object.json') as f:
    #     questions = json.load(f)['questions']
    
    # for i,q in enumerate(questions):
    #     if len(q['program']) != 5:
    #         continue
    #     if q['program'][-2]['type']!='unique':
    #         print (q['question'], q['answer'],q['image_filename'])
    #         for p in q['program']:
    #             print (p)
    #         print 
    #     if i >= 10000:
    #         break


