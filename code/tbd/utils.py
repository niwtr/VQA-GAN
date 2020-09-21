import json as js
import torch
def invert_dict(d):
    """ Utility for swapping keys and values in a dictionary.
    
    Parameters
    ----------
    d : Dict[Any, Any]
    
    Returns
    -------
    Dict[Any, Any]
    """
    return {v: k for k, v in d.items()}


def load_vocab(f):
    vocab = js.load(open(f, 'r'))
    vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
    vocab['program_idx_to_token'] = invert_dict(vocab['program_token_to_idx'])
    vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    # our answers format differs from that of Johnson et al.
    answers = ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow',
               'cube', 'cylinder', 'sphere',
               'large', 'small',
               'metal', 'rubber',
               'no', 'yes',
               '0', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9']
    vocab['answer_idx_to_token'] = dict(zip(range(len(answers)), answers))
    vocab['answer_token_to_idx'] = dict(zip(answers, range(len(answers))))
    return vocab



def map_ans(answers):
    ''' Map the answers from the format Justin Johnson et al. use to our own format. '''
    ans_tensor = torch.LongTensor(answers.size())
    ans_tensor[answers == 15] = 0  # blue
    ans_tensor[answers == 16] = 1  # brown
    ans_tensor[answers == 18] = 2  # cyan
    ans_tensor[answers == 20] = 3  # gray
    ans_tensor[answers == 21] = 4  # green
    ans_tensor[answers == 25] = 5  # purple
    ans_tensor[answers == 26] = 6  # red
    ans_tensor[answers == 30] = 7  # yellow
    ans_tensor[answers == 17] = 8  # cube
    ans_tensor[answers == 19] = 9  # cylinder
    ans_tensor[answers == 29] = 10 # sphere
    ans_tensor[answers == 22] = 11 # large
    ans_tensor[answers == 28] = 12 # small
    ans_tensor[answers == 23] = 13 # metal
    ans_tensor[answers == 27] = 14 # rubber
    ans_tensor[answers == 24] = 15 # no
    ans_tensor[answers == 31] = 16 # yes
    ans_tensor[answers == 4] = 17  # 0
    ans_tensor[answers == 5] = 18  # 1
    ans_tensor[answers == 6] = 19  # 10 <- originally sorted by ASCII value, not numerical
    ans_tensor[answers == 7] = 20  # 2
    ans_tensor[answers == 8] = 21  # 3
    ans_tensor[answers == 9] = 22  # 4
    ans_tensor[answers == 10] = 23 # 5
    ans_tensor[answers == 11] = 24 # 6
    ans_tensor[answers == 12] = 25 # 7
    ans_tensor[answers == 13] = 26 # 8
    ans_tensor[answers == 14] = 27 # 9
    return ans_tensor
