import os, argparse, json
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--scene_path',  required=True, type=str)
parser.add_argument('--question_path',  required=True, type=str)
parser.add_argument('--output_scene_dir',  required=True, type=str)
args = parser.parse_args()

class Dataset:
    def __init__(self, scene_file, question_file):
        self.scene_file = scene_file
        self.question_file = question_file
        self.scenes = None
        self.questions = None
        self.image2question = defaultdict(list)
        self.image2answer = defaultdict(list)

    def load_scene(self):
        print('loading scene.json...')
        scenes = json.load(open(self.scene_file))
        self.scenes = scenes['scenes']
        print('loading json done')
    
    def load_question(self):
        print('loading question.json...')
        questions = json.load(open(self.question_file))
        self.questions = questions['questions']
        for q in self.questions:
            self.image2question[q['image_filename']].append(q['question'])
            self.image2answer[q['image_filename']].append(q['answer'])
        print('loading json done')

def get_box(args):
    os.system('mkdir -p ' + args.output_scene_dir)
    dset = Dataset(args.scene_path, args.question_path) 
    dset.load_scene()
    dset.load_question()

    for _i, scene in enumerate(dset.scenes):
        objects = []
        for obj_idx, obj in enumerate(scene['objects']):
            dic = {}
            for k in ['material', 'size', 'shape', 'color']:
                dic[k] = obj[k]
            if 'obj_bbox' in scene.keys():
                dic['bbox'] = scene['obj_bbox'][str(obj_idx+1)]
            objects.append(dic)
        scene_dic = {
            'objects': objects,
            'image_index': scene['image_index'],
            'image_filename': scene['image_filename'],
            'split': scene['split'],
            'questions': dset.image2question[scene['image_filename']],
            'answer': dset.image2answer[scene['image_filename']]
        }
        output_scene_file = os.path.join(args.output_scene_dir, scene['image_filename'].split('.')[0]+'.json')
        with open(output_scene_file, 'w') as fw:
            json.dump(scene_dic, fw, indent=2)
        
if __name__ == "__main__":
    get_box(args)

        