#!/usr/bin/env python3


import pandas
from pathlib import Path
import codecs

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

source = Path(__file__).parent.parent
model_path = source / "results"
test_task_2 = source / "data/subtask2_test.csv"


def label_data_with_snips_nlu_model(lang='en', save="", out='practice'):
    """ Label counterfactual training data 

    :param lang: abbreviate language name of model
    :param save: path name where model is saved
    :return: csv file
    :rtype: file
    """
    from snips_nlu import SnipsNLUEngine
    from snips_nlu.default_configs import CONFIG_EN
    from snips_nlu_metrics import compute_train_test_metrics, compute_cross_val_metrics
    import pickle
    import json

    model = model_path / "snips_semeval_2020_model_task2_{}".format(save)
    if Path(model).exists():
        print("\n--> Loading Snips model...")
        nlu_engine = SnipsNLUEngine.from_path(model)

        if out == 'evaluate':
            print("--> [EVALUATION] Start labeling with Snips model...")
            pd_data = pandas.read_csv(test_task_2)
            id_sent, pred = [], []
            for i, row in pd_data.iterrows():
                sentence = row['sentence']
                predict = nlu_engine.parse(sentence)
                antecedent, consequent = [], []
                for slot in predict['slots']:
                    if slot['slotName'] == "Consequent":
                        end_id = slot['range']['end']
                        start_id = slot['range']['start']
                        #if end_id != -1: end_id = end_id - 1
                        consequent = [start_id, end_id]
                    if slot['slotName'] == "Antecedent":
                        end_id = slot['range']['end']
                        start_id = slot['range']['start']
                        #if end_id != -1: end_id = end_id - 1
                        antecedent = [start_id, end_id] 
                if len(antecedent) == 0: antecedent = [-1, -1]
                if len(consequent) == 0: consequent = [-1, -1]
                sent_id = row['sentenceID']
                eval_out = (sent_id, antecedent[0], antecedent[1],
                             consequent[0], consequent[1])
                pred.append(eval_out)
                #print(predict)
                #print(predict['input'][antecedent[0]:antecedent[1]])
                print(eval_out)
                
            # antecedent_endid
            results = pandas.DataFrame(data=pred, 
                                        columns=["sentenceID", "antecedent_startid",
                                               "antecedent_endid", "consequent_startid", 
                                               "consequent_endid"])
            model_saved = model_path / \
            "snips_semeval_2020_evaluation_final_{}.csv".format(save)
            results.to_csv(model_saved, index=False)

            from datetime import datetime
            from zipfile import ZipFile
            dtime = datetime.now().strftime("%Y%m%d-%H%M%S")
            results_name = "snips_semeval_2020_evaluation_task2_{}_{}.zip".format(save, dtime)

            results.to_csv(model_saved, index=False)
            with ZipFile(model_path / results_name, 'w') as myzip: 
                myzip.write(str(model_saved), "subtask2.csv")

            print("--> [EVALUATION] End labeling and saving with Snips model...")

        elif out == 'practice':
            print("--> [PRACTICE] Start labeling with Snips model...")
            test_task_prac_1 = source / "data/task2-train.csv"
            pd_data = pandas.read_csv(test_task_prac_1)
            id_sent, pred = [], []
            for i, row in pd_data.iterrows():
                sentence = row['sentence']
                predict = nlu_engine.parse(sentence)
                antecedent, consequent = [], []
                for slot in predict['slots']:
                    if slot['slotName'] == "Consequent":
                        end_id = slot['range']['end']
                        if end_id != -1: end_id = end_id - 1
                        consequent = [slot['range']['start'], end_id]
                    if slot['slotName'] == "Antecedent":
                        end_id = slot['range']['end']
                        if end_id != -1: end_id = end_id - 1
                        antecedent = [slot['range']['start'], end_id] 
                if len(antecedent) == 0: antecedent = [-1, -1]
                if len(consequent) == 0: consequent = [-1, -1]
                sent_id = row['sentenceID']
                eval_out = (sent_id, antecedent[0], antecedent[1],
                             consequent[0], consequent[1])
                pred.append(eval_out)
                print(eval_out, row['antecedent_startid'], row['antecedent_endid'])
                
            # antecedent_endid
            results = pandas.DataFrame(data=pred, 
                                       columns=["sentenceID", "antecedent_startid",
                                               "antecedent_endid", "consequent_startid", 
                                               "consequent_endid"])
                
            model_saved = model_path / \
                "snips_semeval_2020_evaluation_practice_{}.csv".format(save)
            results.to_csv(model_saved, index=False)

            print("--> [EVALUATION] Start labeling and saving with Snips model...")
       
def label_data_with_rasa_nlu_model(lang='en', save="", out='practice'):
    """ Label counterfactual training data 

    :param lang: abbreviate language name of model
    :param save: path name where model is saved
    :return: csv file
    :rtype: file
    """
    from rasa.nlu.model import Interpreter
    from rasa.nlu.components import ComponentBuilder
    from zipfile import ZipFile
    from snips_nlu import SnipsNLUEngine
    import pickle
    import json

    model = model_path / "rasa_semeval_2020_model_task2_{}".format(save)
    if Path(model).exists():
        print("\n--> Loading Rasa model 1...")
        model1 = str(model / "nlu_20200515-042204")
        nlu_engine, nlu_engine2 = "", ""
        with codecs.open(model_path / "builder_task2_{}.pkl".format(save), "rb") as ant:
            builder = pickle.load(ant)
            nlu_engine = Interpreter.load(model1, builder)
            print("\n--> Loading Snips model 2...")
            #model2 = str(model / "nlu_20200515-185057")
            #nlu_engine2 = Interpreter.load(model2, builder)
            print("\n--> Loading Snips model 3...")
            model3 = str(model_path / "snips_semeval_2020_model_task2_ct_v5")
            nlu_engine3 = SnipsNLUEngine.from_path(model3)
            print("\n--> Loading Snips model 4...")
            model4 = model_path / "snips_semeval_2020_model_task2_ct_v6"
            nlu_engine4 = SnipsNLUEngine.from_path(model4) 
            print("\n--> Loading Snips model 5...")
            model5 = model_path / "snips_semeval_2020_model_task2_ct_v7"
            nlu_engine5 = SnipsNLUEngine.from_path(model5)  
            print("\n--> Loading Snips model 6...")
            model6 = str(model / "nlu_20200513-075312_desc")
            nlu_engine6 = Interpreter.load(model6, builder)                          
            #           

        if out == 'evaluate':
            print("--> [EVALUATION] Start labeling with Rasa model...")
            pd_data = pandas.read_csv(test_task_2)
            id_sent, pred = [], []
            count = 0
            for i, row in pd_data.iterrows():
                sentence = row['sentence']
                predict = nlu_engine.parse(sentence, time=3)
                antecedent, consequent = [], []
                #print("-- entities: ", predict['entities'])
                for slot in predict['entities']:
                    if slot['entity'] == "consequent":
                        consequent = [slot['start'], slot['end']]
                    if slot['entity'] == "antecedent":
                        antecedent = [slot['start'], slot['end']] 
                
                if len(antecedent) == 0 and len(consequent) == 0:
                    """
                    predict2 = nlu_engine2.parse(sentence)
                    antecedent, consequent = [], []
                    for slot in predict['entities']:
                        if slot['entity'] == "consequent":
                            consequent = [slot['start'], slot['end']]
                        if slot['entity'] == "antecedent":
                            antecedent = [slot['start'], slot['end']] 
                    """
                    if len(predict['entities']) == 0:
                        predict3 = nlu_engine3.parse(sentence)
                        antecedent, consequent = [], []
                        for slot in predict3['slots']:
                            if slot['slotName'] == "Consequent":
                                end_id = slot['range']['end']
                                start_id = slot['range']['start']
                                #if end_id != -1: end_id = end_id - 1
                                consequent = [start_id, end_id]
                            if slot['slotName'] == "Antecedent":
                                end_id = slot['range']['end']
                                start_id = slot['range']['start']
                                #if end_id != -1: end_id = end_id - 1
                                antecedent = [start_id, end_id]

                        if len(predict3['slots']) == 0:
                            predict4 = nlu_engine4.parse(sentence)
                            antecedent, consequent = [], []
                            for slot in predict4['slots']:
                                if slot['slotName'] == "Consequent":
                                    end_id = slot['range']['end']
                                    start_id = slot['range']['start']
                                    #if end_id != -1: end_id = end_id - 1
                                    consequent = [start_id, end_id]
                                if slot['slotName'] == "Antecedent":
                                    end_id = slot['range']['end']
                                    start_id = slot['range']['start']
                                    #if end_id != -1: end_id = end_id - 1
                                    antecedent = [start_id, end_id]
                            
                            if len(predict4['slots']) == 0:
                                predict5 = nlu_engine5.parse(sentence)
                                antecedent, consequent = [], []
                                for slot in predict5['slots']:
                                    if slot['slotName'] == "Consequent":
                                        end_id = slot['range']['end']
                                        start_id = slot['range']['start']
                                        #if end_id != -1: end_id = end_id - 1
                                        consequent = [start_id, end_id]
                                    if slot['slotName'] == "Antecedent":
                                        end_id = slot['range']['end']
                                        start_id = slot['range']['start']
                                        #if end_id != -1: end_id = end_id - 1
                                        antecedent = [start_id, end_id]
                            
                                if len(predict4['slots']) == 0:
                                    predict6 = nlu_engine6.parse(sentence)
                                    antecedent, consequent = [], []
                                    for slot in predict6['entities']:
                                        if slot['entity'] == "consequent":
                                            consequent = [slot['start'], slot['end']]
                                        if slot['entity'] == "antecedent":
                                            antecedent = [slot['start'], slot['end']]

                                    if len(predict6['entities']) == 0:
                                        count += 1
                                        print("count: ", count)
                                        print(predict4)

                else: print("-------- ok ! ")

                if len(antecedent) == 0: antecedent = [-1, -1]
                if len(consequent) == 0: consequent = [-1, -1]
                
                sent_id = row['sentenceID']
                eval_out = (sent_id, antecedent[0], antecedent[1],
                             consequent[0], consequent[1])
                pred.append(eval_out)
                # print(eval_out)
                
            print("no treated: ", count)
            # antecedent_endid
            results = pandas.DataFrame(data=pred, columns=["sentenceID", 
                                        "antecedent_startid","antecedent_endid", 
                                        "consequent_startid","consequent_endid"])
            model_saved = model_path / \
            "rasa_semeval_2020_evaluation_task2_final_{}.csv".format(save)
            results.to_csv(model_saved, index=False)
            
            from datetime import datetime
            dtime = datetime.now().strftime("%Y%m%d-%H%M%S")
            results_name = "rasa_semeval_2020_evaluation_{}_{}.zip".format(save, dtime)

            results.to_csv(model_saved, index=False)
            with ZipFile(model_path / results_name, 'w') as myzip: 
                myzip.write(str(model_saved), "subtask2.csv")

        elif out == 'practice':
            print("--> [PRACTICE] Start labeling with Rasa model...")
            test_task_prac_1 = source / "data/task2-train.csv"
            pd_data = pandas.read_csv(test_task_prac_1)
            id_sent, pred = [], []
            for i, row in pd_data.iterrows():
                sentence = row['sentence']
                predict = nlu_engine.parse(sentence)
                antecedent, consequent = [], []
                for slot in predict['slots']:
                    if slot['slotName'] == "consequent":
                        end_id = slot['range']['end']
                        if end_id != -1: end_id = end_id - 1
                        consequent = [slot['range']['start'], end_id]
                    if slot['slotName'] == "antecedent":
                        end_id = slot['range']['end']
                        if end_id != -1: end_id = end_id - 1
                        antecedent = [slot['range']['start'], end_id] 
                if len(antecedent) == 0: antecedent = [-1, -1]
                if len(consequent) == 0: consequent = [-1, -1]
                sent_id = row['sentenceID']
                eval_out = (sent_id, antecedent[0], antecedent[1],
                             consequent[0], consequent[1])
                pred.append(eval_out)
                print(eval_out, row['antecedent_startid'], row['antecedent_endid'])
                
            # antecedent_endid
            results = pandas.DataFrame(data=pred, 
                                       columns=["sentenceID", "antecedent_startid",
                                               "antecedent_endid", "consequent_startid", 
                                               "consequent_endid"])
            
            results_name = "rasa_semeval_2020_evaluation_practice_{}.zip".format(save)
            model_saved = model_path / "subtask2.csv"
                
            results.to_csv(model_saved, index=False)
            with ZipFile(model_path / results_name, 'w') as myzip: 
                myzip.write("subtask2.csv", model_saved.open())


if __name__ == "__main__":
    #label_data_with_snips_nlu_model(lang="en", out='evaluate', save="ct_v4")
    label_data_with_rasa_nlu_model(lang="en", out='evaluate', save="v1")