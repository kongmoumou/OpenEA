from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from flask import Flask, session, request
from flask_cors import CORS
from openea.modules.args.args_hander import load_args
from openea.modules.load.kgs import read_kgs_from_folder
from main_from_args import get_model
import random
import numpy as np
import time

app = Flask(__name__)
CORS(app)
app.secret_key = "ea-key"

task_count = 0
model_dict = {}
dict1 = {}
dict2 = {}
def load_vec(filename):
    source_data = open(filename, 'r')
    source_data_dict = {}
    for i, line in enumerate(source_data):
        line = line.split(' ')
        word = line[0]
        word_vec_list = [eval(x) for x in line[1:-1]]
        source_data_dict[word] = np.array(word_vec_list)

    return source_data_dict



@app.route("/")
def hello_world():
    if 'num' not in session:
        session['num'] = random.randint(0, 10)
        return {'num': session['num'], 'new': True}
    else:
        return {'num': session['num']}


@app.route("/start", methods=['POST'])
def start():
    global dict1, dict2
    start = time.time()
    dict1 = load_vec('vec/zh_vec.txt')
    dict2 = load_vec('vec/en_vec.txt')
    end = time.time()
    print("加载词向量用时：{}s".format(end-start)) # 加载中英文词向量, 数据集为ZH_EN_15K.

    if model_dict.get(session.get('task_id')):
        return {
            'code': 0,
            'msg': 'model already init'
        }

    dataset = request.json.get('dataset', 'ZH_EN_15K')
    args = load_args('./args/bootea_args_15K.json')  # json 读参数
    args.training_data = args.training_data + dataset + '/'
    args.dataset_division = '721_5fold/1/'
    print(args.embedding_module)
    print(args)
    remove_unlinked = False
    if args.embedding_module == "RSN4EA":
        remove_unlinked = True
    kgs = read_kgs_from_folder(args.training_data, args.dataset_division, args.alignment_module, args.ordered,
                               remove_unlinked=remove_unlinked)
    model = get_model('BootEAPro')()
    model.set_args(args)
    model.set_kgs(kgs)
    model.init()
    # model.run()
    # model.test()
    # model.save()

    global task_count
    task_count = task_count + 1
    session['task_id'] = task_count
    model_dict[session['task_id']] = {
        'model': model,
        'run': model.run()
    }
    # 初始化第一轮迭代
    next(model_dict[session['task_id']]['run'])

    return {
        'code': 0,
        'msg': 'model init'
    }


@app.route("/iterate", methods=['POST'])
def iterate():
    global model_dict
    if model_dict.get(session.get('task_id')) is None:
        return {
            'code': 1,
            'msg': 'no model'
        }
    
    model = model_dict[session['task_id']]
    model_inst = model['model']
    model.setdefault('run', model_inst.run())
    run = model['run']

    result = next(run)

    return {
        **result,
    }

@app.route("/update", methods=['POST'])
def update():
    if model_dict.get(session.get('task_id')) is None:
        return {
            'code': 1,
            'msg': 'no model'
        }

    model = model_dict[session['task_id']]['model']
    labeled_align = request.json.get('labeled_align', [])
    model.labeled_align = set([(x, y) for x, y in labeled_align])

    return {
        'code': 0,
        'msg': 'update success',
    }


@app.route("/state", methods=['GET'])
def get_state():
    if 'task_id' not in session:
        return {
            'code': 1,
            'msg': 'no model'
        }
    else:
        return {
            'code': 0,
            'model': 'model exist'
        }

@app.route("/kgs", methods=['GET'])
def get_kgs():
    if model_dict.get(session.get('task_id')) is None:
        return {
            'code': 1,
            'msg': 'no model'
        }

    model = model_dict[session['task_id']]['model']
    kg1 = model.kgs.kg1
    kg2 = model.kgs.kg2

    return {
        'kg1': {
            'id': kg1.entities_id_dict,
            'name': kg1.entities_id_name_dict,
            'idx': model.ref_ent1,
        },
        'kg2': {
            'id': kg2.entities_id_dict,
            'name': kg2.entities_id_name_dict,
            'idx': model.ref_ent2,
        }
    }

@app.route("/sim", methods=['GET'])
def get_sim_by_ids():
    if model_dict.get(session.get('task_id')) is None:
        return {
            'code': 1,
            'msg': 'no model'
        }

    id1 = int(request.args.get('id1', 0))
    id2 = int(request.args.get('id2', 0))
    model = model_dict[session['task_id']]['model']
    sim_mat = model.sim_mat
    ref_ent1 = model.ref_ent1
    ref_ent2 = model.ref_ent2

    index1 = ref_ent1.index(id1)
    index2 = ref_ent2.index(id2)

    return {
        'sim': sim_mat[index1][index2].item()
    }

@app.route("/sem_sim", methods=['GET'])
def get_sem_sim_by_ids():
    if model_dict.get(session.get('task_id')) is None:
        return {
            'code': 1,
            'msg': 'no model'
        }
    def similarity(word1, word2):
        if word1 not in dict1.keys():
            print("向量1不存在")
            return None
        if word2 not in dict2.keys():
            print("向量2不存在")
            return None
        vec1 = dict1[word1]
        vec2 = dict2[word2]
        return float(np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))

    id1 = int(request.args.get('id1', 0))
    id2 = int(request.args.get('id2', 0))
    model = model_dict[session['task_id']]['model']
    name_ent1 = model.kgs.kg1.entities_id_name_dict[id1].split('/')[-1]
    name_ent2 = model.kgs.kg2.entities_id_name_dict[id2].split('/')[-1]
    return {
        'sim': similarity(name_ent1, name_ent2)
    }
