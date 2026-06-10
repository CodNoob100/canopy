import os
import glob
import json

base_dir = os.path.expanduser('~/nb301_models')
for path in glob.glob(os.path.join(base_dir, '**/*.json'), recursive=True):
    try:
        with open(path, 'r') as f:
            data = f.read()
            if 'NormalFloatHyperparameter' in data:
                print('Found in:', path)
                data = data.replace('"name": "OptimizerSelector:sgd:learning_rate",', '"name": "OptimizerSelector:sgd:learning_rate", "lower": 0.0001, "upper": 1.0,')
                with open(path, 'w') as out:
                    out.write(data)
                print('Fixed!')
    except Exception as e:
        print(e)
