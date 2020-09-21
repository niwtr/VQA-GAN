#coding:utf-8
from __future__ import print_function
import argparse, json, os, itertools, random, shutil


parser = argparse.ArgumentParser()

# Inputs
parser.add_argument('--input_scene_1hop_dir', default='../output/train/moformat_scenes_1hop',
    help="")
parser.add_argument('--output_scene_0hop_dir', default='../output/train/moformat_scenes_0hop',
    help="JSON file containing metadata about functions")
parser.add_argument('--output_scene_01hop_dir', default='../output/train/moformat_scenes_01hop',
    help="JSON file containing metadata about functions")

def update_qa(input_scene_1hop_file, output_scene_0hop_file, output_scene_01hop_file, question_template, synonyms):
    scene = json.load(open(input_scene_1hop_file))

    # query_param_types
    # type: Color
    # name: <C>
    questions_1hop = scene['questions']
    answers_1hop = scene['answer']
    questions_0hop = []
    answers_0hop = []
    for obj in scene['objects']:
        # generate questions for each object
        for template in question_template:
            param_type_to_name = {p['type']:p['name'] for p in template['params']}
            # random choose question: question_text
            question_text = random.choice(template['text'])
            # determine the query_param_types
            query_param_names = set([template['constraints'][0]['params'][0]])
            # determine the color|shape|: replaceable_param_types
            # for param_type,name in param_type_to_name_list:
            replaceable_param_names = set(['<Z>', '<C>', '<M>', '<S>']) - query_param_names
            replaceable_param_names = list(replaceable_param_names)
            random.shuffle(replaceable_param_names)
            replaceable_param_names = set(replaceable_param_names[:2])

            for param_type in ['Size', 'Color', 'Material', 'Shape']:
                name = param_type_to_name[param_type]
                # print (replaceable_param_names)
                if name in replaceable_param_names:
                    question_text = question_text.replace(name, obj[param_type.lower()])
                else: 
                    if name == '<S>':
                        question_text = question_text.replace(name, 'thing')
                    else:
                        question_text = question_text.replace(name, '')
                    if name in query_param_names:
                        answer = obj[param_type.lower()]
                
            question_text = ' '.join(question_text.split())
            for synk in synonyms.keys():
                synv = random.choice(synonyms[synk])
                question_text = question_text.replace(synk, synv)
            questions_0hop.append(question_text)
            answers_0hop.append(answer)
    
    scene['questions'] = questions_0hop
    scene['answer'] = answers_0hop
    with open(output_scene_0hop_file, 'w') as fw:
        json.dump(scene, fw, indent=2)

    scene['questions'] = questions_0hop + questions_1hop
    scene['answer'] = answers_0hop + answers_1hop
    with open(output_scene_01hop_file, 'w') as fw:
        json.dump(scene, fw, indent=2)

def update_qa4(scene_file, output_scene_file, question_template, synonyms):
    scene = json.load(open(scene_file))

    # query_param_types
    # type: Color
    # name: <C>
    questions = scene['questions']
    answers = scene['answer']
    for obj in scene['objects']:
        # generate questions for each object
        template = random.choice(question_template)
        param_type_to_name = {p['type']:p['name'] for p in template['params']}
        # random choose question: question_text
        question_text = random.choice(template['text'])
        # determine the query_param_types
        query_param_names = set([template['constraints'][0]['params'][0]])
        # determine the color|shape|: replaceable_param_types
        # for param_type,name in param_type_to_name_list:
        replaceable_param_names = set(['<Z>', '<C>', '<M>', '<S>']) - query_param_names

        for param_type in ['Size', 'Color', 'Material', 'Shape']:
            name = param_type_to_name[param_type]
            # print (replaceable_param_names)
            if name in replaceable_param_names:
                question_text = question_text.replace(name, obj[param_type.lower()])
            else: 
                if name == '<S>':
                    question_text = question_text.replace(name, 'thing')
                else:
                    question_text = question_text.replace(name, '')
                if name in query_param_names:
                    answer = obj[param_type.lower()]
            
        question_text = ' '.join(question_text.split())
        for synk in synonyms.keys():
            synv = random.choice(synonyms[synk])
            question_text = question_text.replace(synk, synv)
        questions.append(question_text)
        answers.append(answer)
    scene['questions'] = questions
    scene['answer'] = answers
    with open(output_scene_file, 'w') as fw:
        json.dump(scene, fw, indent=2)

if __name__ == '__main__':
    args = parser.parse_args()
    synonyms_json = 'synonyms.json'
    with open(synonyms_json, 'r') as f:
        synonyms = json.load(f)
    question_template_file = 'zerohop_templates/zero_hop.json'
    with open(question_template_file, 'r') as f:
        question_template = json.load(f)
    
    scene_dir = args.input_scene_1hop_dir
    os.system('mkdir -p ' + args.output_scene_0hop_dir)
    os.system('mkdir -p ' + args.output_scene_01hop_dir)

    for filename in os.listdir(scene_dir):
        input_scene_1hop_file = os.path.join(scene_dir, filename)
        output_scene_0hop_file = os.path.join(args.output_scene_0hop_dir, filename)
        output_scene_01hop_file = os.path.join(args.output_scene_01hop_dir, filename)
        update_qa(input_scene_1hop_file, output_scene_0hop_file, output_scene_01hop_file, question_template, synonyms)    
