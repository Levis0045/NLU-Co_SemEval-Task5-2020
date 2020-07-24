#!/usr/bin/env python3


import pandas
from pathlib import Path
import codecs

from data_format import BuildSnipsDataTask2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

source_config = Path(__file__).parent.parent / "configs" / "config_5"
source_result = Path(__file__).parent.parent / "results"
source = Path(__file__).parent.parent / "data"


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
        train_data_obj = BuildSnipsDataTask2(lang, cross=cross, vers=save, add_entities=True)
        train_data = train_data_obj.build_snips_data_task2()
        print("--> Evaluating training data with Snips metrics...")
        filename_results = source_result / "snips_semeval_2020_evaluation_task2_{}.pkl".format(save)
        if not Path(filename_results).exists():
            tt_metrics = compute_train_test_metrics(train_dataset=train_data[0],
                                                test_dataset=train_data[1],
                                                engine_class=SnipsNLUEngine,
                                                include_slot_metrics=True)
            #print(tt_metrics)
            if not Path(filename_results).exists():
                with codecs.open(filename_results, 'wb') as metric:
                    pickle.dump(tt_metrics, metric)
                from datetime import datetime
                dmtime = "_{}_{}".format(save, datetime.now().strftime("%Y%m%d-%H%M%S"))
                name = "snips_semeval_2020_evaluation_task2{}.json".format(dmtime)
                filename_results_json = source_result / name
                with codecs.open(filename_results_json, 'w', "utf-8") as m_json:
                    json.dump(tt_metrics, m_json)

    else:
        filename_results = source_result / "snips_semeval_2020_model_task2_{}".format(save)
        train_data_obj = BuildSnipsDataTask2(lang, cross=cross, vers=save, add_entities=True)
        train_data = train_data_obj.build_snips_data_task2()
        #print(CONFIG_EN)
        nlu_engine = SnipsNLUEngine(CONFIG_EN)
        print("--> Training patent data with Snips...")
        nlu_engine.fit(train_data)
        """
        try:     
            print("--> Saving model trained with Snips (JOBLIB)...")
            filename_joblib = source_result / "snips_semeval_2020_model_task2_{}.pkl".format(save)            
            with codecs.open(filename_joblib, 'wb') as metric:
                pickle.dump(nlu_engine, metric)
        except: pass
        """
        print("--> Saving model trained with Snips (SNIPS)...")
        try: nlu_engine.persist(filename_results)
        except Exception as e: print("error saving the madel....{}".format(str(e)))


def train_eval_rasa_nlu_model(lang='en', cross=False, save=''):
    """ Train rasa data from all brat annotation object 

    :param lang: abbreviate language name 
    :param save: path where model will be save
    :return: None
    :rtype: None
    """
    from rasa.nlu.training_data import load_data
    from rasa.nlu.model import Trainer
    from rasa.nlu.components import ComponentBuilder
    from rasa.nlu import config
    from rasa.nlu.test import run_evaluation
    import pickle

    config_file = source_config / "config_rasa_bert.yml"

    if cross:
        train_data_obj = BuildSnipsDataTask2(lang, cross=cross, vers=save)
        train_data = train_data_obj.build_rasa_data_task2()
        filename_results = source_result / "rasa_cross_semeval_2020_model_task2_{}".format(save)
        if Path(filename_results).exists():
            training_data = load_data(str(train_data[0]))
            builder = ComponentBuilder(use_cache=True)  
            with codecs.open(source_result / "builder_task2_{}.pkl".format(save), "wb") as ant:
                pickle.dump(builder, ant)
            trainer = Trainer(config.load(str(config_file)), builder)
            print("\n--> Training patent data with Rasa (Cross-validation)...")
            trainer.train(training_data, num_threads=8, verbose=True)
            print("--> Saving model trained with Rasa (Cross-validation)...")
            model_directory = trainer.persist(filename_results)
            
        print("--> Evaluating training data with Rasa metrics (Cross-validation)...")
        import os
        from datetime import datetime
        filename_test = str(train_data[1])
        dmtime = "test_{}_{}".format(save, datetime.now().strftime("%Y%m%d-%H%M%S"))
        out_test = source_result / "rasa_cross_evaluation_task2" / dmtime
        model_directory = sorted(filename_results.glob("nlu_*"), key=os.path.getmtime)[-1] 
        print(out_test)
        run_evaluation(filename_test, str(model_directory), output_directory=str(out_test))

    else:
        filename_results = source_result / "rasa_semeval_2020_results_task2_{}".format(save)
        train_data_obj = BuildSnipsDataTask2(lang, cross=cross, vers=save)
        train_file = train_data_obj.build_rasa_data_task2()

        print("\n--> Training will use the file: {}...".format(str(train_file)))
        training_data = load_data(str(train_file))
        builder = ComponentBuilder(use_cache=True)  
        with codecs.open(source_result / "builder_task2_{}.pkl".format(save), "wb") as ant:
            pickle.dump(builder, ant)
        trainer = Trainer(config.load(str(config_file)), builder)
        print("\n--> Training patent data with Rasa...")
        trainer.train(training_data, num_threads=12, n_jobs=8, verbose=True, fixed_model_name="nlu")
        print("--> Saving model trained with Rasa...")
        model_directory = trainer.persist(filename_results)

        """
        print("--> Evaluating training data with Rasa metrics (Cross-validation)...")
        filename_test = str(train_data[1])
        out_test = source_result / "rasa_cross_evaluation" / "test_{}".format(save)
        model_directory = filename_results / "nlu_20200506-104931"
        run_evaluation(filename_test, str(model_directory), output_directory=str(out_test))
        """



if __name__ == "__main__":
    #train_eval_snips_nlu_model(lang='en', cross=True, save='ct_v8')
    train_eval_rasa_nlu_model(lang='en', cross=True, save='v1')
