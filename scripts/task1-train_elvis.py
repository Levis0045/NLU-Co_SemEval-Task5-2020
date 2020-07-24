# /usr/bin/python


import sklearn_crfsuite
from pathlib import Path
import codecs
from data_format import BuildSnipsDataTask1

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

source_data = Path(__file__).parent.parent / "data"
source_config = Path(__file__).parent.parent / "configs" / "config_5"
source_result = Path(__file__).parent.parent / "results"


def train_eval_crfsuite_model(lang, vers, make_data=False, final=False):
    import pickle

    task1 = BuildSnipsDataTask1(lang, vers=vers)
    if make_data: task1.build_feature_train()

    print("--> Build training data...")
    filename_train = source_data / "crfsuite_semeval_2020_train_{}.pkl".format(vers)
    model_saved  = source_result / "crfsuite_semeval_2020_model_{}.crf".format(vers)
    #if Path(filename_train).exists():
    file_data = pickle.load(open(filename_train, 'rb'))
    X_train = [x for i, x in enumerate(file_data[0]) if i>= 4000 and x[1] == '0']
    y_train = file_data[1]
    if final:
        t_x_train = file_data[2]
        t_y_train = file_data[3]
        for x in t_x_train: X_train.append(x)
        for x in t_y_train: y_train.append(x)

    print("--> Initialised CRF Engine...")
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.0010,
        c2=0.00001,
        verbose=True,
        linesearch="StrongBacktracking",
        epsilon=1e-6,
        #variance=0.1,
        #gamma=0.6,
        delta=1e-6,
        all_possible_states=True,
        max_iterations=400,
        all_possible_transitions=True,
        model_filename=str(model_saved)
    )
    print("--> Sentences lenght: ", len(X_train), "\n--> Language lenght: ", len(y_train))
    print("--> Start training CRF data...\n")
    crf.fit(X_train, y_train)
    print("--> End training CRF data...\n")


def train_eval_naivebayes_model(self):
    print("\n------- Build training data...\n")
    file_out = corpus+"/models/training_data.pkl"
    if Path(file_out).exists():
        file_data = pickle.load(open(file_out, 'rb'))
        self.X_train, self.y_train = file_data[0], file_data[1]
        self.X_test, self.y_test = file_data[2], file_data[3]
        self.alphabet = file_data[4]
    else: self.build_feature_train()

    model_file, model = corpus+"/models/model_naivebayes_data.pkl", {}
    if Path(model_file).exists(): 
        classifier = pickle.load(open(model_file, 'rb'))
    else: 
        print("\n------- Initialised NaiveBayesClassifier Engine...\n")
        featuresets_train = [(n, lang) for (n, lang) in zip(self.X_train, self.y_train)]
        featuresets_test = [(n, lang) for (n, lang) in zip(self.X_test, self.y_test)]
        shuffle(featuresets_train)
        shuffle(featuresets_test)
        
        train_set, test_set, dev_test = featuresets_train, featuresets_test, featuresets_train[60000:75000]

        print("Train lenght: ", len(train_set), 
                "\nTest lenght: ", len(test_set), 
                "\nDev Train lenght: ", len(dev_test))
        
        from nltk import NaiveBayesClassifier
        from nltk.probability import ELEProbDist
        from nltk.classify import apply_features, accuracy
        print("\n------- Start training NaiveBayesClassifier data...\n")
        classifier = NaiveBayesClassifier.train(train_set, estimator=ELEProbDist)
        print("----> Saving training NaiveBayesClassifier data...\n")
        file_out = open(model_file, "wb")
        pickle.dump(classifier, file_out)
        file_out.close()
    print("\n------- End training NaiveBayesClassifier data...\n")

    print("\n------- Start evaluating NaiveBayesClassifier data...\n")
    print('Test accuracy: ',  accuracy(classifier, test_set))
    print('Dev test accuracy: ', accuracy(classifier, dev_test))

    errors = []
    for (ft, tag) in dev_test:
        guess = classifier.classify(ft)
        name  = ft['sentence.lower()']
        if guess != tag: errors.append( (tag, guess, name) )
    for (tag, guess, name) in sorted(errors[:100]):
        print('correct={:<8} guess={:<8s} name={:<30}'.format(tag, guess, name))
    print("\n------- End evaluating NaiveBayesClassifier data...\n")


def train_eval_fasttext_model(vers):
    import codecs
    import pandas
    import re

    train_task_1 = source_data / "task1-train.csv"
    test_task_1 = source_data / "subtask1_test.csv"

    filename_train = source_data / "fasttext_semeval_2020_train_{}.txt".format(vers)
    filename_test = source_data / "fasttext_semeval_2020_test_{}.txt".format(vers)
    
    regex = r"[0-9_•​:#&$*?!/\-%µ@=+'~\[\]£””“»()<>|]"
    with codecs.open(filename_train, 'a', 'utf-8') as tp:
        datas = pandas.read_csv(train_task_1)
        datas = datas.sample(frac=1).reset_index(drop=True)
        for i, x in datas.loc[:10000, :].iterrows():
            #sent = re.sub(regex, r"", x['sentence'], re.I)
            sent = x['sentence']
            tp.write("__label__{} {}\n".format(x['gold_label'], sent))
        with codecs.open(filename_test, 'a', 'utf-8') as tp:    
            for i, x in datas.loc[10001:, :].iterrows():
                #sent = re.sub(regex, r"", x['sentence'], re.I)
                sent = x['sentence']
                tp.write("__label__{} {}\n".format(x['gold_label'], sent))    
    
    import fasttext
    model = fasttext.train_supervised(
        input=str(filename_train), 
        lr=0.001, 
        epoch=1000,
        wordNgrams=4,
        dim=400,
        loss='hs'
    )
    #model.save_model(str(source_result / "fasttext_semeval_2020.model"))
    print(model.test(str(filename_test), k=5))

    test_task_prac_1 = source_data / "task1-train.csv"
    pd_data = pandas.read_csv(test_task_prac_1)
    id_sent, pred = [], []
    for i, row in pd_data.iterrows():
        sentence = row['sentence']
        predict = model.predict(sentence)
        if predict[0][0] == "__label__0":
            pred.append((row['sentenceID'], 0))
        elif predict[0][0] == "__label__1":
            pred.append((row['sentenceID'], 1))
        #print(predict[0][0], row['gold_label'])
        results = pandas.DataFrame(
            data=pred, columns=["sentenceID", "pred_label"])
    model_saved = source_result / \
        "fasttext_semeval_2020_evaluation_practice_{}.csv".format(vers)
    results.to_csv(model_saved, index=False)


def train_eval_snips_nlu_model(lang='en', cross=False, save=''):
    """ Train snips data from all brat annotation object 

    :param lang: abbreviate language name 
    :param save: path where model will be save
    :return: None
    :rtype: None
    """
    from snips_nlu import SnipsNLUEngine
    from snips_nlu.default_configs import CONFIG_EN
    from snips_nlu_metrics import compute_train_test_metrics, compute_cross_val_metrics
    import pickle
    import json

    if cross:
        train_data_obj = BuildSnipsDataTask1(lang, cross=cross, vers=save)
        train_data = train_data_obj.build_snips_data_task1()
        print("--> Evaluating training data with Snips metrics...")
        filename_results = source_result / "snips_semeval_2020_evaluation_task1_{}.pkl".format(save)
        if not Path(filename_results).exists():
            tt_metrics = compute_train_test_metrics(train_dataset=train_data[0],
                                                test_dataset=train_data[1],
                                                engine_class=SnipsNLUEngine,
                                                include_slot_metrics=False)
            #print(tt_metrics)
            if not Path(filename_results).exists():
                print("--> Writing snips nlu metrics data to file...")
                with codecs.open(filename_results, 'wb') as metric:
                    pickle.dump(tt_metrics, metric)
                from datetime import datetime
                dmtime = "_{}_{}".format(save, datetime.now().strftime("%Y%m%d-%H%M%S"))
                name = "snips_semeval_2020_evaluation_task1{}.json".format(dmtime)
                filename_results_json = source_result / name
                with codecs.open(filename_results_json, 'w', "utf-8") as m_json:
                    json.dump(tt_metrics, m_json)

    else:
        filename_results = source_result / "snips_semeval_2020_model_task1_{}".format(save)
        train_data_obj = BuildSnipsDataTask1(lang, cross=cross, vers=save)
        train_data = train_data_obj.build_snips_data_task1()
        nlu_engine = SnipsNLUEngine(config=CONFIG_EN)
        print("--> Training patent data with Snips...")
        nlu_engine.fit(train_data)
        try:     
            print("--> Saving model trained with Snips (JOBLIB)...")
            filename_joblib = source_result / "snips_semeval_2020_model_task1_{}.pkl".format(save)            
            with codecs.open(filename_joblib, 'wb') as metric:
                pickle.dump(nlu_engine, metric)
        except: pass
        print("--> Saving model trained with Snips (SNIPS)...")
        try: nlu_engine.persist(filename_results)
        except: pass
        

def train_eval_rasa_nlu_model(lang='en', cross=False, save=''):
    """ Train snips data from all brat annotation object 

    :param lang: abbreviate language name 
    :param save: path where model will be save
    :rtype: None
    """
    from rasa.nlu.training_data import load_data
    from rasa.nlu.model import Trainer
    from rasa.nlu.components import ComponentBuilder
    from rasa.nlu import config
    from rasa.nlu.test import run_evaluation

    config_file = source_config / "config_rasa_converrt.yml"

    if cross:
        filename_results = source_result / "rasa_cross_semeval_2020_model_task1_{}".format(save)

        train_data_obj = BuildSnipsDataTask1(lang, cross=cross, vers=save)
        train_data = train_data_obj.build_rasa_data_task1()

        training_data = load_data(str(train_data[0]))
        builder = ComponentBuilder(use_cache=True)  
        trainer = Trainer(config.load(str(config_file)), builder)
        
        print("--> Training patent data with Rasa...")
        trainer.train(training_data, num_threads=8, n_jobs=-1, verbose=True)
        
        print("--> Saving model trained with Rasa (Rasa)...")
        model_directory = trainer.persist(filename_results)
        
        print("--> Evaluating training data with Rasa metrics (Cross-validation)...")
        import os
        from datetime import datetime
        filename_test = str(train_data[1])
        print(filename_test)
        dmtime = "test_{}_{}".format(save, datetime.now().strftime("%Y%m%d-%H%M%S"))
        out_test = source_result / "rasa_cross_evaluation_task1" / dmtime
        model_directory = sorted(filename_results.glob("nlu_*"), key=os.path.getmtime)[-1] 
        run_evaluation(filename_test, str(model_directory), output_directory=str(out_test))

    else:
        filename_results = source_result / "rasa_semeval_2020_model_task1_{}".format(save)
        train_data_obj = BuildSnipsDataTask1(lang, cross=cross, vers=save)
        train_file = train_data_obj.build_rasa_data_task1()

        training_data = load_data(train_file)
        builder = ComponentBuilder(use_cache=True)  
        trainer = Trainer(config.load(str(config_file)), builder)
        
        print("--> Training patent data with Rasa...")
        trainer.train(training_data, num_threads=8, verbose=True, n_jobs=-1, fixed_model_name="nlu")
        
        print("--> Saving model trained with Rasa (Rasa)...")
        model_directory = trainer.persist(filename_results)
        



if __name__ == '__main__':
    #train_eval_crfsuite_model(lang='en', vers="v3_prev_next", make_data=False, final=True)
    #train_eval_fasttext_model("v7")
    #train_eval_snips_nlu_model(lang='en', cross=True, save='v8')
    train_eval_rasa_nlu_model(lang='en', cross=True, save='v1')
