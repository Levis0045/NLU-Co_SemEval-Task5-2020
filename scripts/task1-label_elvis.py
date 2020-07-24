# /usr/bin/python3


import sklearn_crfsuite
from pathlib import Path
import pandas

from data_format import BuildSnipsDataTask1

source_data = Path(__file__).parent.parent / "data"
source_result = Path(__file__).parent.parent / "results"
test_task_1 = source_data / "subtask1_test.csv"


def label_crfsuite_model(lang, vers, make_data=False,
                         evaluate=True, out='practice'):
    import pickle
    from sklearn_crfsuite import metrics
    import stanza
    #stanza.download('en')
    nlp = stanza.Pipeline('en')

    task1 = BuildSnipsDataTask1(lang, vers=vers)
    if make_data:
        task1.build_feature_train()

    print("\n--> Build training data...")
    model_saved = source_result / \
        "crfsuite_semeval_2020_model_{}.crf".format(vers)

    if Path(model_saved).exists():
        print("--> Load CRF Engine...")
        crf = sklearn_crfsuite.CRF(model_filename=str(model_saved))

        if evaluate:
            print("--> Start evaluation CRF model with model data...\n")
            filename_train = source_data / \
                "crfsuite_semeval_2020_train_{}.pkl".format(vers)
            file_data = pickle.load(open(filename_train, 'rb'))
            X_test = file_data[2]
            y_test = file_data[3]
            y_pred = crf.predict(X_test)
            labels = list(crf.classes_)
            print(metrics.flat_classification_report(
                y_test, y_pred, labels=labels, digits=3
            ))
        if out == 'evaluate':
            print("--> [EVALUATION] Start labeling with CRF data...")
            pd_data = pandas.read_csv(test_task_1)
            id_sent, pred = [], []
            row_data = list(pd_data.iterrows())
            len_row_data = len(row_data)
            for i, row in row_data:
                print("\tProcessing row data %s / %s"%(i+1,len_row_data))
                sentence = row['sentence']
                sent_feature = task1.sent2features(sentence, nlp)
                predict = crf.predict_single(sent_feature)
                pred.append((row['sentenceID'], int(predict[0])))
                print(predict)
                results = pandas.DataFrame(
                    data=pred, columns=["sentenceID", "pred_label"])
            model_saved = source_result / \
            "crfsuite_semeval_2020_evaluation_final_{}.csv".format(vers)
            results.to_csv(model_saved, index=False)
        elif out == 'practice':
            print("--> [PRACTICE] Start labeling with CRF data...")
            test_task_prac_1 = source_data / "task1-train.csv"
            pd_data = pandas.read_csv(test_task_prac_1)
            id_sent, pred = [], []
            for i, row in pd_data.iterrows():
                sentence = row['sentence']
                sent_feature = task1.sent2features(sentence, nlp)
                predict = crf.predict_single(sent_feature)
                pred.append((row['sentenceID'], int(predict[0])))
                #print(predict[0], row['gold_label'])
                results = pandas.DataFrame(
                    data=pred, columns=["sentenceID", "pred_label"])
            model_saved = source_result / \
                "crfsuite_semeval_2020_evaluation_practice_{}.csv".format(vers)
            results.to_csv(model_saved, index=False)

        print("--> Successfully end all operations with CRF data...\n")


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

    model = source_result / "snips_semeval_2020_model_task1_{}".format(save)
    if Path(model).exists():
        print("\n--> Loading Snips model...")
        nlu_engine = SnipsNLUEngine.from_path(model)

        if out == 'evaluate':
            print("--> [EVALUATION] Start labeling with Snips model...")
            pd_data = pandas.read_csv(test_task_1)
            pred = []
            for i, row in pd_data.iterrows():
                sentence = row['sentence']
                sent_id = row['sentenceID']
                print(i, sentence, "dffffffffffffffffffffffffffffffffff")
                sent_parse = nlu_engine.parse(sentence, intents=["Counterfactual", "NoCounterfactual"])
                if sent_parse['intent']['intentName']  == "Counterfactual": 
                    pred.append((sent_id, 1))
                elif sent_parse['intent']['intentName'] == "NoCounterfactual":
                    pred.append((sent_id, 0))
                else: 
                    sent_parse = nlu_engine.parse(sentence, top_n=3, intents=["Counterfactual", "NoCounterfactual"])
                    if sent_parse[1]['intent']['intentName']  == "Counterfactual": 
                        pred.append((sent_id, 1))
                        print('NULL [1]- Counterfactual ')
                    elif sent_parse[1]['intent']['intentName'] == "NoCounterfactual":
                        pred.append((sent_id, 0))
                        #pred.append((sent_id, 0))
                        print('NULL [1]- NoCounterfactual ')
                
                print(sent_parse['intent']['intentName'])
            
            results = pandas.DataFrame(data=pred, 
                            columns=["sentenceID", "pred_label"])
            model_saved = source_result / \
            "snips_semeval_2020_evaluation_task1_final_{}.csv".format(save)
            results.to_csv(model_saved, index=False)
            
        elif out == 'practice':
            print("--> [PRACTICE] Start labeling with Snips model...")
            test_task_prac_1 = source_data / "task1-train.csv"
            pd_data = pandas.read_csv(test_task_prac_1)
            pred = []
            for i, row in pd_data.iterrows():
                sentence = row['sentence']
                sent_parse = nlu_engine.parse(sentence)
                if sent_parse['intent']['intentName'] == "Counterfactual": 
                    pred.append((row['sentenceID'], 1))
                elif sent_parse['intent']['intentName'] == "NoCounterfactual":
                    pred.append((row['sentenceID'], 0))
                else: print(sent_parse['intent']['intentName'])
                
                #print(predict[0], row['gold_label'])
            results = pandas.DataFrame(data=pred, 
                                columns=["sentenceID", "pred_label"])
            model_saved = source_result / \
            "snips_semeval_2020_evaluation_pratice_task1_{}.csv".format(save)
            results.to_csv(model_saved, index=False)
                


def label_data_with_rasa_nlu_model(lang='en', save="", out='practice'):
    """ Label counterfactual training data 

    :param lang: abbreviate language name of model
    :param save: path name where model is saved
    :return: csv file
    :rtype: file

    model_20200501-025838 = 0.58
    model_20200502-090721 = 0.48
    model_20200502-135337 = 
    """
    from rasa.nlu.model import Interpreter
    from rasa.nlu.components import ComponentBuilder
    import pickle
    import json

    model = source_result / "rasa_semeval_2020_model_task1_{}".format(save)
    if Path(model).exists():
        print("\n--> Loading Rasa model...")
        model = str(model / "nlu_20200509-063701")
        builder = ComponentBuilder(use_cache=True)  
        nlu_engine = Interpreter.load(model, builder)

        if out == 'evaluate':
            print("--> [EVALUATION] Start labeling with Rasa model...")
            pd_data = pandas.read_csv(test_task_1)
            pred = []
            for i, row in pd_data.iterrows():
                sentence = row['sentence']
                sent_id = row['sentenceID']
                sent_parse = nlu_engine.parse(sentence)
                if sent_parse['intent']['name']  == "counterfactual": 
                    pred.append((sent_id, 1))
                elif sent_parse['intent']['name'] == "no_counterfactual":
                    pred.append((sent_id, 0))
                else: print("ERROR__: ", sent_parse)
                
                print(sent_parse['intent']['name'], sent_parse['text'])
            
            results = pandas.DataFrame(data=pred, 
                            columns=["sentenceID", "pred_label"])
            model_saved = source_result / \
            "rasa_semeval_2020_evaluation_task1_final_{}.csv".format(save)
            results.to_csv(model_saved, index=False)
            
            from datetime import datetime
            from zipfile import ZipFile
            dtime = datetime.now().strftime("%Y%m%d-%H%M%S")
            results_name = "rasa_semeval_2020_evaluation_task1_{}_{}.zip".format(save, dtime)
                
            results.to_csv(model_saved, index=False)
            with ZipFile(source_result / results_name, 'w') as myzip: 
                myzip.write(str(model_saved), "subtask1.csv")

        elif out == 'practice':
            print("--> [PRACTICE] Start labeling with Rasa model...")
            test_task_prac_1 = source_data / "task1-train.csv"
            pd_data = pandas.read_csv(test_task_prac_1)
            pred = []
            for i, row in pd_data.iterrows():
                sentence = row['sentence']
                sent_parse = nlu_engine.parse(sentence)
                if sent_parse['intent']['intentName'] == "Counterfactual": 
                    pred.append((row['sentenceID'], 1))
                elif sent_parse['intent']['intentName'] == "NoCounterfactual":
                    pred.append((row['sentenceID'], 0))
                else: print(sent_parse['intent']['intentName'])
                
                #print(predict[0], row['gold_label'])
            results = pandas.DataFrame(data=pred, 
                                columns=["sentenceID", "pred_label"])
            model_saved = source_result / \
            "rasa_semeval_2020_evaluation_pratice_task1_{}.csv".format(save)
            results.to_csv(model_saved, index=False)
                




if __name__ == '__main__':
    #label_crfsuite_model("en", vers="v3_prev_next", make_data=False, out='evaluate')
    #label_data_with_nlu_model(save="v3_deprel", out="evaluate")
    label_data_with_rasa_nlu_model(save="v1", out="evaluate")
