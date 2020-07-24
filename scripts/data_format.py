#!/usr/bin/env python3


import pandas
from pathlib import Path
import codecs
import yaml
import re 
import chardet
#import nltk

source = Path(__file__).parent.parent / "data"
train_task_1 = source / "task1-train.csv"
train_task_2 = source / "task2-train.csv"


__all__ = ["BuildSnipsDataTask2", "BuildSnipsDataTask1"]


isascii = lambda s: len(s) == len(s.encode())

counterfactual_regex = r"(If|Since|Neither|And|That|But|Also|Had|In|Unless)\s.+\s(would|could|should|I|may|might)+('nt|'d|'m|'ve|still|be)? .+"
counterfactual_regex_c = re.compile(counterfactual_regex, re.I)

class BuildSnipsDataTask2():
    def __init__(self, lang='en', cross=False, 
                 vers="v3", add_entities=False):
        self.lang = lang
        self.cross = cross
        self.vers = vers
        self.add_entities = add_entities
        
    def build_entities_train(self, antecedent, consequent):
        entities, group_ant, group_conq = [], {}, {}
        group_ant['name'] = "semeval/antecedent"
        group_ant['type'] = 'entity'
        group_ant['matching_strictness'] = 0.6
        group_ant['values'] = [x for x in antecedent]
        group_conq['name'] = "semeval/consequent"
        group_conq['type'] = 'entity'
        group_conq['matching_strictness'] = 0.6
        group_conq['values'] = [x for x in consequent]
        entities.append(group_ant)
        entities.append(group_conq)

        filename_entities = source / "snips_semeval_2020_entities_{}.yaml".format(self.vers)
        if not Path(filename_entities).exists():
            with codecs.open(filename_entities, "w", encoding="utf8") as pt:
                yaml.dump_all(entities, pt)

        return entities

    def build_intent_train(self, data, ents_list=None, 
                            form="snips"):
        stream_group_ant, stream_group_conq = {}, {}
        stream_group_ant_conq, stream_results = {}, []
        antecedent_or_consequent, antece_conquent = [], []
        antecedent_ent, consequent_ent = [], []
        if form == "snips":
            for tr in data:
                if re.search(tr[1], tr[0]) and re.search(tr[2], tr[0]):
                    utter = re.sub(tr[1], "[Antecedent](%s)"%tr[1], tr[0])
                    utter = re.sub(tr[2], "[Consequent](%s)"%tr[2], utter)
                    antece_conquent.append((utter, 1))
                    antecedent_ent.append(tr[1])
                    consequent_ent.append(tr[2])
                if re.search(tr[1], tr[0]) and not re.search(tr[2], tr[0]):
                    utter = re.sub(tr[1], "[Antecedent](%s)"%tr[1], tr[0])
                    antecedent_or_consequent.append((utter, 1))
                    antecedent_ent.append(tr[1])     
                if not re.search(tr[1], tr[0]) and re.search(tr[2], tr[0]):
                    utter = re.sub(tr[2], "[Consequent](%s)"%tr[2], tr[0])
                    antecedent_or_consequent.append((utter, 1))
                    consequent_ent.append(tr[1])       

            stream_group_ant_conq['name'] = "counterfactual_antecedent_consequent"
            stream_group_ant_conq['type'] = 'intent'
            stream_group_ant_conq['slots'] = [
                {'entity':"semeval/antecedent",'name':'Antecedent'},
                {'entity':"semeval/consequent",'name':'Consequent'}
            ]
            stream_group_ant_conq['utterances'] = [x[0] for x in antece_conquent]
            stream_group_ant['name'] = "counterfactual_antecedent_or_consequent"
            stream_group_ant['type'] = 'intent'
            stream_group_ant['slots'] = [
                {'entity':"semeval/antecedent",'name':'Antecedent'}
            ]
            stream_group_ant['utterances'] = [x[0] for x in antecedent_or_consequent]
            stream_results.append(stream_group_ant_conq)
            #stream_results.append(stream_group_conq)
            stream_results.append(stream_group_ant)

            if self.add_entities:
                entities_list = self.build_entities_train(antecedent_ent, consequent_ent)
                stream_results.append(entities_list[0])
                stream_results.append(entities_list[1])
            return stream_results

        elif form == "rasa":
            for tr in data:
                if tr[4][0] != -1:
                    utter = re.sub(tr[1], "[%s](antecedent)"%tr[1], tr[0])
                    utter = re.sub(tr[2], "[%s](consequent)"%tr[2], utter)
                    utter = re.sub(r"('|\"| [\.-]){2,5}", "", utter)
                    utter = utter.replace(" �", "")
                    #utter = re.sub(r"(\[|\(|\]|\))(\[|\)|\])", r"\1", utter)
                    antece_conquent.append((utter, 1))
                else: 
                    utter = re.sub(tr[1], "[%s](antecedent)"%tr[1], tr[0])
                    utter = re.sub(r"('|\"| [\.-]){2,5}", "", utter)
                    utter = utter.replace(" �", "")
                    #utter = re.sub(r"(\[|\(|\]|\))(\[|\)|\]|\()", r"\1", utter)
                    antecedent_or_consequent.append((utter, 1))

            return antece_conquent, antecedent_or_consequent

    def build_snips_data_task2(self):
        """ Build snips data from all brat annotation object 

        :param lang: abbreviate language name 
        :return: Snips Dataset of all brat annotation object
        :rtype: snips_nlu.dataset.Dataset
        """
        import yaml
        import io
        from snips_nlu.dataset import Dataset
        import re
        from sklearn.model_selection import train_test_split

        print("--> Creating snips nlu data training...")
        stream_results = []
        pandas_train = pandas.read_csv(train_task_2.absolute())
        stream_group_ant, stream_group_conq, utterances = {}, {}, []
        stream_group_ant_conq = {}
        
        antecedent_list, consequent_list, entities = [], [], []
        for i, row in pandas_train.iterrows():
            sent = row['sentence']
            ant = row['antecedent']
            conq = row['consequent']
            utterances.append(((sent, ant, conq), 1))
            if ant != "{}": antecedent_list.append(ant)
            if conq != "{}": consequent_list.append(conq)

        entities_list = (antecedent_list, consequent_list)
        filename_train = source / "snips_semeval_2020_train_task2_cross_{}.yaml".format(self.vers)
        filename_test = source / "snips_semeval_2020_test_task2_cross_{}.yaml".format(self.vers)
        
        if self.cross:
            utter_train = [x[0] for x in utterances]
            utter_test  = [x[1] for x in utterances]
            train, test, label_train, label_test = train_test_split(utter_train, utter_test, 
                                                    test_size=0.2, random_state=42)
            
            if not Path(filename_train).exists():
                stream_results = self.build_intent_train(train, entities_list)
                print("--> Writing snips nlu TRAINING data to file...")
                with codecs.open(filename_train, "w", encoding="utf8") as pt:
                    yaml.dump_all(stream_results, pt)

            if not Path(filename_test).exists():
                stream_results = self.build_intent_train(test, entities_list)
                print("--> Writing snips nlu TESTING data to file...")
                with codecs.open(filename_test, "w", encoding="utf8") as pt:
                    yaml.dump_all(stream_results, pt) 
            
            json_dataset_train, json_dataset_test = [], [] 
            with codecs.open(filename_train, "r", encoding="utf8") as pt:
                data_counterfact = io.StringIO(pt.read().strip().replace('﻿',''))
                json_dataset_train = Dataset.from_yaml_files(self.lang, [data_counterfact]).json  
            with codecs.open(filename_test, "r", encoding="utf8") as pt:
                data_counterfact = io.StringIO(pt.read().strip().replace('﻿',''))
                json_dataset_test = Dataset.from_yaml_files(self.lang, [data_counterfact]).json   

            DATASET_JSON = (json_dataset_train, json_dataset_test)   
            return DATASET_JSON
        else:
            utter_train = [x[0] for x in utterances]
            #self.vers = "all_"+self.vers
            filename_train = source / "snips_semeval_2020_train_task2_{}.yaml".format(self.vers)
            if not Path(filename_train).exists():
                stream_results = self.build_intent_train(utter_train, entities_list)
                print("--> Writing snips nlu TRAINING data to file...")
                with codecs.open(filename_train, "w", encoding="utf8") as pt:
                    yaml.dump_all(stream_results, pt)
            
            json_dataset_train = []
            with codecs.open(filename_train, "r", encoding="utf8") as pt:
                data_counterfact = io.StringIO(pt.read().strip().replace('﻿',''))
                json_dataset_train = Dataset.from_yaml_files(self.lang, [data_counterfact]).json  
                return json_dataset_train
            
    def build_rasa_data_task2(self):
        """ Build rasa data from all brat annotation object 

        :param lang: abbreviate language name 
        :return: Snips Dataset of all brat annotation object
        :rtype: file names
        """
        import yaml
        import io
        from snips_nlu.dataset import Dataset
        import re
        from sklearn.model_selection import train_test_split
        
        stream_results = []
        stream_group_ant, stream_group_conq, utterances = {}, {}, []
        stream_group_ant_conq = {}
        antecedent_list, consequent_list, entities = [], [], []

        global_path = "rasa_semeval_2020_test_task2_*_{}.md".format(self.vers)
        if len(list(Path(source).glob(global_path))) == 0:
            print("\n--> Creating rasa nlu data training...")
        pandas_train = pandas.read_csv(train_task_2.absolute())
        for i, row in pandas_train.iterrows():
            sent = row['sentence']
            ant  = row['antecedent']
            conq = row['consequent']
            conq_id = (row['consequent_startid'], row['consequent_endid'])
            ant_id = (row['antecedent_startid'], row['antecedent_endid'])
            utterances.append(((sent, ant, conq, ant_id, conq_id), 1))
            if ant  != "{}": antecedent_list.append(ant)
            if conq != "{}": consequent_list.append(conq)

        if self.cross:
            utter_all = [x[0] for x in utterances]
            utter_labels_ant  = [x[1] for x in utterances]
            filename_train = source / "rasa_semeval_2020_train_task2_cross_{}.md".format(self.vers)
            filename_test = source / "rasa_semeval_2020_test_task2_cross_{}.md".format(self.vers)
            train, test, label_ant_train, label_ant_test = train_test_split(utter_all, utter_labels_ant, 
                                                                test_size=0.2, random_state=42)
            ants = []
            conqs = []
            for utterance in test:
                ants.append(utterance[1])
                conqs.append(utterance[2])
            entities_list = (ants, conqs)
            if not Path(source / "antecedents.txt").exists():
                with codecs.open(source / "antecedents.txt", "a") as ant:
                    for x in ants: ant.write(x+'\n')
            if not Path(source / "consequents.txt").exists():
                with codecs.open(source / "consequents.txt", "a") as con:
                    for x in conqs: con.write(x+'\n')    

            if not Path(source / filename_train).exists():
                stream_results = self.build_intent_train(train, entities_list, form="rasa")
                print("--> Writing rasa nlu TRAINING data to file...")
                from mdrec import MDRec
                md = MDRec(save_file=filename_train)
                md.rec("intent:counterfactual_antecedent_consequent", h=2)
                md.rec([x[0] for x in stream_results[0]])
                md.rec("intent:counterfactual_antecedent_or_consequent", h=2)
                md.rec([x[0] for x in stream_results[1]])
                stream_results = self.build_intent_train(test, entities_list, form="rasa")
                print("--> Writing rasa nlu TEST data to file...")
                from mdrec import MDRec
                md = MDRec(save_file=filename_test)
                md.rec("intent:counterfactual_antecedent_consequent", h=2)
                md.rec([x[0] for x in stream_results[0]])
                md.rec("intent:counterfactual_antecedent_or_consequent", h=2)
                md.rec([x[0] for x in stream_results[1]])

            if not Path(source / filename_test).exists():
                stream_results = self.build_intent_train(test, entities_list, form="rasa")
                print("--> Writing rasa nlu TESTING data to file...")
                md2 = MDRec(save_file=filename_test)

                md2.rec("intent:counterfactual_antecedent_consequent", h=2)
                md2.rec("lookup:antecedent", h=2)
                md2.rec("data/antecedents.txt")
                md2.rec("lookup:consequent", h=2)
                md2.rec("data/consequents.txt")
            
            return filename_train, filename_test

        else:
            filename_train = source / "rasa_semeval_2020_train_task2_main_{}.md".format(self.vers)
            if not Path(filename_train).exists():
                print("--> Writing rasa nlu TRAINING data to file...")
                utter_train = [x[0] for x in utterances]
                self.vers = "all_"+self.vers
                stream_results = self.build_intent_train(utter_train, entities_list, form="rasa")
        
                from mdrec import MDRec
                md = MDRec(save_file=filename_train)
                md.rec("intent:counterfactual_antecedent_consequent", h=2)
                md.rec([x[0] for x in stream_results[0]])
                md.rec("intent:counterfactual_antecedent", h=2)
                md.rec([x[0] for x in stream_results[1]])
                md.rec("lookup:antecedent", h=2)
                md.rec("data/antecedents.txt")
                md.rec("lookup:consequent", h=2)
                md.rec("data/consequents.txt")
            return filename_train
            
            
           

from collections import Counter
from random import shuffle



class BuildSnipsDataTask1():
    def __init__(self, lang, vers="", cross=None):
        self.train_sents = []
        self.test_sents  = []
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.vers = vers
        self.lang = lang
        self.alphabet = Counter()
        self.cross = cross

    def normalize_ponctuations(self, data):
        results = ''
        punctuation = {
            ',': ' <comma> ',
            '"': ' <quotation> ',
            ';': ' <semicolon> ',
            '!': ' <exclamation> ',
            '?': ' <question> ',
            '(': ' <paren_l> ',
            ')': ' <paren_r> ',
            '--': ' <hyphen_d> ',
            ':': ' <colon> ',
            '-': ' <hyphen_s> '
        }
        for pt, val in punctuation.items():
            results = data.replace(pt, val)
        return results

    def make_train_data(self):
        datas = pandas.read_csv(train_task_1)
        results, gold, no_gold = [], [], []
        oo, aa = 0, 0
        for i, data in datas.iterrows():
            #clean_data = self.clean_sentences(data["sentence"])
            label = data["gold_label"]
            results.append((data["sentence"], label))

        results = self.shuffle_xtime(results, 4)
        print("--> All data sample: ", len(results))
        three_quart = int(3*len(results)/4)
        one_quart = int(1*len(results)/4)
        print("--> Training sample: ", three_quart)
        print("--> Test sample: ", one_quart)
        for x, y in results[-one_quart:]: 
            for l in x.split(): 
                for w in l: self.alphabet[w] += 1
        #print([x[1] for x in results])
        return results[:three_quart], results[-one_quart:]        

    def shuffle_xtime(self, data, repeat):
        results, i = data, 1
        while i <= repeat: 
            shuffle(results)
            i+=1
        return results

    def clean_sentences(self, sentences):
        clean_results = []
        regex = r"[0-9_•​:#&$*?!/\-%µ@=+'~\[\]£””“»()<>|]"
        for sent in sentences.split(' '):
            clean_form = self.normalize_ponctuations(sent)
            clean_form = re.sub(regex, r"", clean_form, re.I)
            clean_results.append(clean_form)
        #shuffle(clean_results)    
        return " ".join(clean_results)

    def word2features(self, sent, nlp): 
        words = sent.split()
        lenword = len(words)
        middlew = int(lenword/2)
        threeword = int(lenword/3)
        encoding = chardet.detect(str.encode(sent))
        
        features = {
            'bias': 0.01,
            'sentence.lower()': sent.lower(),
            'sentence.len()': lenword,
            'sentence[-middleword:]': " ".join(words[-middlew:]),
            'sentence[threeword:]': " ".join(words[threeword:]),
            'sentence[:threeword]': " ".join(words[:threeword]),
            #'sentence[syntactic]': " ".join(rd_parser.parse(sent)),
            'sentence[:3]': " ".join(words[:2]),
            'sentence[-lenword]': words[-lenword],
            'encoding': encoding['encoding']
        }
        
        doc = nlp(sent)
        for doc_sent in doc.sentences:
            words = doc_sent.words
            for i, word in enumerate(words):
                wd = word.text
                features["count({}-{})".format(wd, str(i))] = len(wd)
                features["word({}-{})".format(wd, str(i))] = wd.lower()
                features["word-tag({}-{})".format(wd, str(i))] = word.upos
                features["word-lemma({}-{})".format(wd, str(i))] = word.lemma
                features['word.isupper({})'.format(wd)] = wd[0].isupper()
                features['word.istitle({})'.format(wd)] = wd.istitle()
                features['word.deprel({})'.format(wd)] = word.deprel
                #features['word.isdigit()'] = word.isdigit()
                #features['word.isascii()'] = isascii(word)
                try: 
                    if word.feats != None:
                        features["word.feats({})".format(wd)] = word.feats
                except: pass
                try: features["word.misc({})".format(wd)] = word.misc
                except: pass

                if i > 0: 
                    prwd = words[i-1]
                    features["prev-word({}-{})".format(prwd.text, str(i-1))] = prwd.text
                    features["prev-tag({}-{})".format(prwd.text, str(i-1))] = prwd.upos
                    features["prev-lemma({}-{})".format(prwd.text, str(i-1))] = prwd.lemma
                    features["prev-deprel({}-{})".format(prwd.text, str(i-1))] = prwd.deprel
                if i < len(words)-1:
                    prwd = words[i+1]
                    features["next-word({}-{})".format(prwd.text, str(i+1))] = prwd.text
                    features["next-tag({}-{})".format(prwd.text, str(i+1))] = prwd.upos
                    features["next-lemma({}-{})".format(prwd.text, str(i+1))] = prwd.lemma
                    features["next-deprel({}-{})".format(prwd.text, str(i+1))] = prwd.deprel
                    #features["startword({}-{})".format(word, str(i+1))] = "".join(word[2:])
                    #features["endword({}-{})".format(word, str(i+1))] = "".join(word[-2:])
                
        #if words[-lenword] in ['.','!','?']: features['EOS'] = True
        #else: features['EOS'] = False
        #print(features)
        return features

    def sent2features(self, sent, nlp):
        return [self.word2features(sent, nlp)]

    def sent2labels(self, sent):
        #print(sent[1])
        return [str(sent)]

    def build_feature_train(self):
        import stanza
        import pickle
        #stanza.download('en')
        nlp = stanza.Pipeline(self.lang) # This sets up a default neural pipeline in English
        train_sents, test_sents = self.make_train_data()
        if len(self.alphabet.keys()) == 0:
            for x, y in test_sents: 
                for l in x.split(): 
                    for w in l: self.alphabet[x] += 1
        print("\n--> Build train features from dataset...")
        self.X_train = [self.sent2features(s[0], nlp) for s in train_sents]
        self.y_train = [self.sent2labels(s[1]) for s in train_sents]
        print("--> Build test features from dataset...")
        #for x, y in zip(self.X_train, self.y_train): print(x, y)
        self.X_test = [self.sent2features(s[0], nlp) for s in test_sents]
        self.y_test = [self.sent2labels(s[1]) for s in test_sents]      

        data_saved = [self.X_train, self.y_train, self.X_test, self.y_test, self.alphabet]
        filename_train = source / "crfsuite_semeval_2020_train_{}.pkl".format(self.vers)
        file_out = open(filename_train, "wb")
        pickle.dump(data_saved, file_out)
        file_out.close()
        print('\n--> End build features from dataset...')
        return None

    def build_intent_train_task1(self, data, form="snips", split="train",
                                 ents_list=None, path_name=None):
        if form == "snips":
            #import stanza
            #stanza.download('en')
            #nlp = stanza.Pipeline('en') # This sets up a default neural pipeline in English
            #deprels = ['root', 'obj', 'mark']

            br = re.compile(r'[\[\)\(\]\s]+', re.M)
            st = re.compile(r"\s*([.;,:!'])\s*", re.M)

            counter_list, no_counter_list = [], []
            entities, lendata = {}, len(data)
            for i, tr in enumerate(data):
                print('\t-- Working on {} / {} '.format(i+1, lendata))
                #print(tr[1], tr[0])
                sent = re.sub(r'(\(|\)|\[|\])', '', tr[0])
                #doc = nlp(sent)
                #print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')
                new_sent = sent
                
                if tr[1] == 1: 
                    #new_sent = st.sub(r'\1 ', new_sent)
                    #new_sent = counterfactual_regex_c.sub(r'[counterfactual_struc]($1) ', new_sent)
                    #print(new_sent)
                    counter_list.append(new_sent)
                elif tr[1] == 0: 
                    #new_sent = br.sub(' ', new_sent)
                    #new_sent = st.sub(r'$1 ', new_sent)
                    #print(new_sent)
                    no_counter_list.append(new_sent)
            
            counter_list = counter_list
            no_counter_list = no_counter_list

            stream_results, slots = [], []
            stream_counter, stream_no_counter = {}, {}
            """
            for arg in deprels: 
                #arg = arg.replace(':', '-')
                slots.append({'entity':"semeval/%s"%arg, 'name':'%s'%arg.title()})
            """
            stream_counter['name'] = "Counterfactual"
            stream_counter['type'] = 'intent'
            #stream_counter['slots'] = [{'entity':"semeval/consequent",'name':'Consequent'}]
            stream_counter['slots'] = slots
            stream_counter['utterances'] = counter_list 
            stream_no_counter['name'] = "NoCounterfactual"
            stream_no_counter['type'] = 'intent'
            #stream_no_counter['slots'] = [{'entity':"semeval/consequent",'name':'Consequent'}]
            stream_no_counter['slots'] = slots
            if split == "train":
                no_counter_list = no_counter_list[:int(len(no_counter_list)/2)]
            stream_no_counter['utterances'] = no_counter_list
            stream_results.append(stream_counter)
            stream_results.append(stream_no_counter)
            return stream_results

        if form == "rasa":
            from mdrec import MDRec
            import json
            counter_list, no_counter_list = [], []
            self.vers = "all_"+self.vers
            filename_train = str(path_name)

            md = MDRec(save_file=filename_train)
            lendata = len(data)

            for i, tr in enumerate(data):
                print('\t-- Working on {} / {}'.format(i+1, lendata))
                sent = re.sub(r'(\[|\]|\)|\()', '', tr[0])
                sent = re.sub(r"(- '|\"|.')", "", sent)
                #sent = re.sub(r" [.]{2}", "", sent)
                if tr[1] == 1: counter_list.append(sent)
                elif tr[1] == 0: no_counter_list.append(sent)
                    
            # \[(If)\]\(counterfactual_mark\) (\w+ \w+)
            # \[(If|But|that)\]\(counterfactual_mark\)
            # print(len(counter_list), len(no_counter_list))
            md.rec("intent:counterfactual", h=2)
            md.rec(counter_list)
            md.rec("intent:no_counterfactual", h=2)
            if split == "train":
                no_counter_list = no_counter_list[:int(len(no_counter_list)/2)]
            else: no_counter_list = no_counter_list
            md.rec(no_counter_list)
            #md.rec("lookup:causal_mark", h=2)
            #md.rec("data/causal_mark.txt")
            md.rec("regex:counterfactual", h=2)
            md.rec([counterfactual_regex])
            md.rec("regex:counterfactual", h=2)
            md.rec(["\w+ \w+ That \w+ \w+", "(If|Had|Evenif) \w+ \w+ \w+"])
            return md

    def build_snips_data_task1(self):
        """ Build snips data from all brat annotation object 

        :return: Snips Dataset of all brat annotation object
        :rtype: snips_nlu.dataset.Dataset
        """
        import yaml
        import io
        from snips_nlu.dataset import Dataset
        import re
        from sklearn.model_selection import train_test_split

        print("--> Creating snips nlu data training...")
        stream_results = []
        pandas_train = pandas.read_csv(train_task_1.absolute())
        stream_counter, stream_no_counter, utterances = {}, {}, []
        stream_group_ant_conq = {}
        
        counter_list, no_counter_list, entities = [], [], []
        for i, row in pandas_train.iterrows():
            sent = row['sentence']
            gold = row['gold_label']
            utterances.append(((sent, gold), 1))

        filename_train = source / "snips_semeval_2020_train_task1_cross_{}.yaml".format(self.vers)
        filename_test = source / "snips_semeval_2020_test_task1_cross_{}.yaml".format(self.vers)
        
        if self.cross:
            utter_train = [x[0] for x in utterances]
            utter_test  = [x[1] for x in utterances]
            
            train, test, label_train, label_test = train_test_split(utter_train, utter_test, 
                                                            test_size=0.2, random_state=42)
            
            if not Path(filename_train).exists():
                stream_results = self.build_intent_train_task1(train, split="train")
                print("--> Writing snips nlu TRAINING data to file...")
                with codecs.open(filename_train, "w", encoding="utf8") as pt:
                    yaml.dump_all(stream_results, pt)
            
            if not Path(filename_test).exists():
                stream_results = self.build_intent_train_task1(test, split="test")
                print("--> Writing snips nlu TESTING data to file...")
                with codecs.open(filename_test, "w", encoding="utf8") as pt:
                    yaml.dump_all(stream_results, pt) 
            
            json_dataset_train, json_dataset_test = [], [] 
            with codecs.open(filename_train, "r", encoding="utf8") as pt:
                data_counterfact = io.StringIO(pt.read().strip().replace('﻿',''))
                json_dataset_train = Dataset.from_yaml_files(self.lang, [data_counterfact]).json  
            with codecs.open(filename_test, "r", encoding="utf8") as pt:
                data_counterfact = io.StringIO(pt.read().strip().replace('﻿',''))
                json_dataset_test = Dataset.from_yaml_files(self.lang, [data_counterfact]).json   

            DATASET_JSON = (json_dataset_train, json_dataset_test)   
            return DATASET_JSON
        else:
            utter_train = [x[0] for x in utterances]
            self.vers = "all_"+self.vers
            filename_train = source / "snips_semeval_2020_train_task1_main_{}.yaml".format(self.vers)
            
            if not Path(filename_train).exists():
                stream_results = self.build_intent_train_task1(utter_train)
                print("--> Writing snips nlu TRAINING data to file...")
                with codecs.open(filename_train, "w", encoding="utf8") as pt:
                    yaml.dump_all(stream_results, pt)
            
            json_dataset_train = []
            with codecs.open(filename_train, "r", encoding="utf8") as pt:
                data_counterfact = io.StringIO(pt.read().strip().replace('﻿',''))
                json_dataset_train = Dataset.from_yaml_files(self.lang, [data_counterfact]).json  
                return json_dataset_train

    def build_rasa_data_task1(self):
        """ Build rasa data from all brat annotation object 

        :return: Rasa Dataset of all brat annotation object
        :rtype: List(Path)
        """
        import yaml
        import io
        from snips_nlu.dataset import Dataset
        import re
        from sklearn.model_selection import train_test_split

        print("--> Creating rasa nlu data training...")
        stream_results = []
        pandas_train = pandas.read_csv(train_task_1.absolute())
        stream_counter, stream_no_counter, utterances = {}, {}, []
        stream_group_ant_conq = {}
        
        counter_list, no_counter_list, entities = [], [], []
        for i, row in pandas_train.iterrows():
            sent = row['sentence']
            gold = row['gold_label']
            utterances.append(((sent, gold), 1))

        if self.cross:
            filename_train = source / "rasa_semeval_2020_train_task1_cross_{}.md".format(self.vers)
            filename_test = source / "rasa_semeval_2020_test_task1_cross_{}.md".format(self.vers)

            utter_train = [x[0] for x in utterances]
            utter_test  = [x[1] for x in utterances]
            
            train, test, label_train, label_test = train_test_split(utter_train, utter_test, 
                                                                test_size=0.2, random_state=42)
            
            if not Path(filename_train).exists():
                mdrec = self.build_intent_train_task1(train, form="rasa", 
                                                        path_name=filename_train, split="train")
                print("--> Writing rasa nlu TRAINING data to file...")
            
            if not Path(filename_test).exists():
                mdrec = self.build_intent_train_task1(test, form="rasa", 
                                                            path_name=filename_test, split="test")
                print("--> Writing rasa nlu TESTING data to file...")

            return filename_train, filename_test
        else:
            utter_train = [x[0] for x in utterances]
            filename_train = source / "rasa_semeval_2020_train_task1_main_{}.md".format(self.vers)
            if not Path(filename_train).exists():
                mdrec = self.build_intent_train_task1(utter_train, form="rasa", path_name=filename_train)
                print("--> Writing rasa nlu TRAINING data to file...")
                return str(filename_train)
            else: return str(filename_train)