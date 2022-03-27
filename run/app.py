from flask import Flask, session
from openea.modules.args.args_hander import load_args
from openea.modules.load.kgs import read_kgs_from_folder
from main_from_args import get_model
import random

app = Flask(__name__)
app.secret_key = "ea-key"

task_count = 0
model_dict = {}

@app.route("/")
def hello_world():
    if 'num' not in session:
        session['num'] = random.randint(0, 10)
        return {'num': session['num'], 'new': True}
    else:
        return {'num': session['num']}


@app.route("/start", methods=['POST'])
def start():
    global model_dict
    if model_dict.get(session.get('task_id')):
        return {
            'code': 0,
            'msg': 'model already init'
        }

    args = load_args('./args/bootea_args_15K.json')  # json 读参数
    args.training_data = args.training_data + 'D_Y_15K_V1' + '/'
    args.dataset_division = '721_5fold/1/'
    print(args.embedding_module)
    print(args)
    remove_unlinked = False
    if args.embedding_module == "RSN4EA":
        remove_unlinked = True
    kgs = read_kgs_from_folder(args.training_data, args.dataset_division, args.alignment_module, args.ordered,
                               remove_unlinked=remove_unlinked)
    model = get_model(args.embedding_module)()
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
        'run': None,
    }
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
    run = model_inst.run()
    model['run'] = run

    result = next(run)

    return {
        'data': result
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
