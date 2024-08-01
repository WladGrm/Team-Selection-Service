import os
import itertools
import torch

from sentence_transformers import SentenceTransformer, util

from docs.config import TZ_FILE_DIR_PATH
from docs.config import CODE_EMB_TENSOR_DIR_PATH
from docs.config import FORMULA_EMB_TENSOR_DIR_PATH
from docs.config import AREA_EMB_TENSOR_DIR_PATH
from docs.config import CODE_AREA_EMB_TENSOR_DIR_PATH

model = SentenceTransformer('cointegrated/rubert-tiny2')
model = model.to("cpu")

#######################################
# База поиска
#######################################

#База данных числовых Шифров специальностей

#Работает в гугл колабе, на локальной машине - выдает ошибку
#codes = torch.load(os.path.abspath("./docs/nlp_files/digits.pt"))
codes = ['01.01.01',
 '01.01.02',
 '01.01.07',
 '01.01.03',
 '01.01.04',
 '01.01.06',
 '01.01.05',
 '01.02.04',
 '01.02.06',
 '01.02.01',
 '01.01.09',
 '01.02.05',
 '01.02.08',
 '01.03.02',
 '01.03.03',
 '01.03.04',
 '01.03.01',
 '01.04.01',
 '01.04.05',
 '01.04.04',
 '01.04.02',
 '01.04.06',
 '01.04.03',
 '01.04.09',
 '01.04.07',
 '01.04.08',
 '01.04.13',
 '01.04.11',
 '01.04.10',
 '01.04.16',
 '01.04.20',
 '01.04.18',
 '01.04.17',
 '01.04.15',
 '01.04.14',
 '01.04.21',
 '02.00.21',
 '03.01.02',
 '03.01.01',
 '03.01.03',
 '03.01.05',
 '03.01.04',
 '03.01.07',
 '03.01.06',
 '03.01.08',
 '03.01.09',
 '03.02.02',
 '03.02.04',
 '03.02.01',
 '03.02.06',
 '03.02.03',
 '03.02.05',
 '03.02.11',
 '03.02.07',
 '03.02.10',
 '03.02.08',
 '03.02.12',
 '03.02.09',
 '03.03.02',
 '03.03.04',
 '03.02.14',
 '03.02.13',
 '03.03.03',
 '03.03.01',
 '03.03.05',
 '05.01.01',
 '03.03.06',
 '05.02.07',
 '05.02.04',
 '05.02.10',
 '05.02.08',
 '05.02.09',
 '05.02.02',
 '05.02.05',
 '05.02.13',
 '05.02.11',
 '05.02.18',
 '05.02.22',
 '05.02.23',
 '05.04.02',
 '05.04.13',
 '05.04.03',
 '05.04.12',
 '05.04.11',
 '05.04.06',
 '05.05.03',
 '05.05.04',
 '05.07.01',
 '05.05.06',
 '05.07.09',
 '05.07.05',
 '05.07.02',
 '05.07.07',
 '05.07.06',
 '05.07.03',
 '05.07.10',
 '05.08.05',
 '05.08.01',
 '05.08.04',
 '05.08.03',
 '05.08.06',
 '05.09.05',
 '05.09.01',
 '05.09.03',
 '05.09.02',
 '05.09.07',
 '05.11.01',
 '05.09.12',
 '05.09.10',
 '05.11.03',
 '05.11.13',
 '05.11.07',
 '05.11.06',
 '05.11.10',
 '05.11.08',
 '05.11.14',
 '05.11.15',
 '05.11.18',
 '05.11.16',
 '05.12.04',
 '05.11.17',
 '05.12.07',
 '05.12.14',
 '05.13.01',
 '05.13.06',
 '05.12.13',
 '05.13.05',
 '05.13.10',
 '05.13.12',
 '05.13.11',
 '05.13.18',
 '05.13.15',
 '05.13.17',
 '05.14.01',
 '05.13.19',
 '05.14.02',
 '05.13.20',
 '05.14.03',
 '05.14.04',
 '05.16.01',
 '05.16.02',
 '05.14.08',
 '05.14.12',
 '05.14.14',
 '05.16.06',
 '05.16.05',
 '05.16.04',
 '05.16.08',
 '05.16.09',
 '05.16.07',
 '05.17.01',
 '05.17.03',
 '05.17.06',
 '05.17.04',
 '05.17.02',
 '05.17.08',
 '05.18.01',
 '05.17.11',
 '05.17.07',
 '05.17.18',
 '05.18.15',
 '05.18.04',
 '05.18.12',
 '05.18.07',
 '05.18.05',
 '05.18.06',
 '05.18.17',
 '05.19.01',
 '05.19.05',
 '05.19.02',
 '05.19.04',
 '05.21.01',
 '05.20.02',
 '05.20.01',
 '05.20.03',
 '05.22.01',
 '05.22.06',
 '05.21.03',
 '05.21.05',
 '05.22.07',
 '05.22.10',
 '05.22.19',
 '05.22.13',
 '05.22.08',
 '05.22.14',
 '05.22.17',
 '05.23.05',
 '05.23.02',
 '05.23.03',
 '05.23.01',
 '05.23.04',
 '05.23.08',
 '05.23.16',
 '05.23.17',
 '05.23.11',
 '05.23.07',
 '05.23.19',
 '05.23.22',
 '05.25.05',
 '05.25.02',
 '05.25.03',
 '05.23.21',
 '05.23.20',
 '05.26.01',
 '05.26.02',
 '05.26.03',
 '05.26.06',
 '05.26.05',
 '05.27.01',
 '05.27.03',
 '05.27.06',
 '05.27.02',
 '06.01.05',
 '06.01.03',
 '06.01.04',
 '06.01.02',
 '06.01.01',
 '06.02.02',
 '06.02.04',
 '06.02.01',
 '06.02.03',
 '06.01.06',
 '06.01.07',
 '06.02.10',
 '06.02.09',
 '06.02.06',
 '06.02.05',
 '06.02.07',
 '06.02.08',
 '06.03.02',
 '06.03.03',
 '06.03.01',
 '10.01.01',
 '10.01.03',
 '10.01.02',
 '09.00.14',
 '10.01.09',
 '10.02.02',
 '10.02.01',
 '10.01.08',
 '10.01.10',
 '10.02.20',
 '10.02.03',
 '10.02.05',
 '10.02.19',
 '10.02.04',
 '10.02.14',
 '10.02.21',
 '13.00.08',
 '14.01.01',
 '14.01.05',
 '14.01.03',
 '14.01.06',
 '14.01.02',
 '14.01.04',
 '14.01.10',
 '14.01.08',
 '14.01.12',
 '14.01.09',
 '14.01.07',
 '14.01.11',
 '14.01.13',
 '14.01.17',
 '14.01.14',
 '14.01.15',
 '14.01.16',
 '14.01.22',
 '14.01.19',
 '14.01.18',
 '14.01.23',
 '14.01.20',
 '14.01.21',
 '14.01.25',
 '14.01.28',
 '14.01.24',
 '14.01.26',
 '14.01.29',
 '14.01.27',
 '14.02.02',
 '14.02.01',
 '14.02.03',
 '14.02.05',
 '14.02.04',
 '14.01.30',
 '14.03.01',
 '14.03.02',
 '14.03.03',
 '14.02.06',
 '14.03.06',
 '14.03.09',
 '14.03.05',
 '14.03.07',
 '14.03.04',
 '14.03.08',
 '14.03.10',
 '14.04.01',
 '14.04.02',
 '14.03.11',
 '25.00.36']

###############################################
# Общие Функции
################################################
#Возвращает массив шифров специальностей размером top_k
def get_semantic_search_id_list(query, corpus, codes, top_k):
    result = []
    search = list(itertools.chain.from_iterable(util.semantic_search(query, corpus, top_k = top_k)))
    for item in search:
        result.append(codes[item["corpus_id"]])
    return result

#Возвращает массив оценок специальностей в диапазоне от 0 до 10 размером top_k
def get_semantic_search_score(query, corpus, top_k):
    result = []
    search = list(itertools.chain.from_iterable(util.semantic_search(query, corpus, top_k = top_k)))
    for item in search:
        result.append(item["score"]*10)
    return result

###############################################
# Классы
################################################

#Класс, отвечающий за выделение полей ТЗ
class Parse:
    def __init__(self, file_path):
        self.path = file_path


    def parse_text_between(self, start_string, end_string):
        with open(self.path, 'r', encoding="utf-8") as file:
            content = file.read()

        start_index = content.find(start_string)
        if start_index == -1:
            return "Start string not found"

        start_index += len(start_string)
        end_index = content.find(end_string, start_index)
        if end_index == -1:
            return "End string not found"

        return content[start_index:end_index].strip()
    

    def parse_text_after_string(self, target_string):
        with open(self.path, 'r', encoding="utf-8") as file:
            content = file.read()

        target_index = content.find(target_string)
        if target_index == -1:
            return "Target string not found"

        return content[target_index + len(target_string):].strip()
    




#Класс - структура данных решения 
class Solution_NLP:
    def __init__(self, code_list, score_list ):
        #Список для хранения шифров релевантных специальностей
        self.code_list = code_list

        #Список для хранения результатов релевантных специальностей
        self.score_list = score_list


class NLP:
    def __init__(self):

        self.file_path = TZ_FILE_DIR_PATH

        self.mode_param = 6

        self.top_k = 5

        #Коды специальностей
        self.codes = codes

        #Шифр специальности - лист тензор-эмбеддинглв
        self.code_emb_tensor = torch.load(os.path.abspath(CODE_EMB_TENSOR_DIR_PATH))

        #Формула специальности - лист тензор-эмбеддингов
        self.formula_emb_tensor = torch.load(os.path.abspath(FORMULA_EMB_TENSOR_DIR_PATH))

        #Область исследований специальности - лист тензор-эмбеддингов
        self.area_emb_tensor = torch.load(os.path.abspath(AREA_EMB_TENSOR_DIR_PATH))

        #Шифр + Область исследований специальности - лист тензор-эмбеддингов
        self.code_area_emb_tensor = torch.load(os.path.abspath(CODE_AREA_EMB_TENSOR_DIR_PATH))


    def set_file_path(self, path):
        self.file_path = path

    def set_mode_param(self, mode_param):
        if (mode_param>=1 and mode_param<=6):
            self.mode_param = mode_param
        else:
            pass


    def set_top_k(self, top_k):
        if (top_k>=1 and top_k<=len(codes)):
            self.top_k = top_k
        else:
            pass
    

      
################################################## 
# Режимы поиска
##################################################
#1 -  Цель - Формула
#2 -  Объект+Имя+Техн характеристики - Области исследований + Шифр
#3 -  Назначение - Шифр
#4 -  Состав - Область исследований
#5 -  Технические характеристики - Области исследований
#6 -  Среднее всех вариантов сравнения   

    def solve(self):


        tz = Parse (self.file_path)

        tz_aim = tz.parse_text_between('Цель','Объект')
        tz_object = tz.parse_text_between('Объект', 'Содержание')
        tz_name = tz.parse_text_between('Название', 'Цель')
        tz_cont = tz.parse_text_between('Содержание', 'Технические требования')
        tz_tech = tz.parse_text_after_string('Технические требования')

        #Режим поиска
        match self.mode_param:
            case 1:
                tz_aim_emb_tensor = model.encode(tz_aim,convert_to_tensor=True)
                code_list_ = get_semantic_search_id_list(tz_aim_emb_tensor, self.formula_emb_tensor, codes, self.top_k)
                score_list_ = get_semantic_search_score(tz_aim_emb_tensor, self.formula_emb_tensor, self.top_k)
            
            case 2:
                tz_object_name_tech_emb_tensor = model.encode(tz_object+'[SEP]'+tz_name+'[SEP]'+tz_tech,convert_to_tensor=True)
                code_list_ = get_semantic_search_id_list(tz_object_name_tech_emb_tensor, self.code_area_emb_tensor, codes, self.top_k)
                score_list_ = get_semantic_search_score(tz_object_name_tech_emb_tensor, self.code_area_emb_tensor, self.top_k)

            case 3:
                tz_name_emb_tensor = model.encode(tz_name,convert_to_tensor=True)
                code_list_ = get_semantic_search_id_list(tz_name_emb_tensor, self.code_emb_tensor, codes, self.top_k)
                score_list_ = get_semantic_search_score(tz_name_emb_tensor, self.code_emb_tensor, self.top_k)

            case 4:
                tz_cont_emb_tensor = model.encode(tz_cont,convert_to_tensor=True)
                code_list_ = get_semantic_search_id_list(tz_cont_emb_tensor, self.area_emb_tensor, codes, self.top_k)
                score_list_ = get_semantic_search_score(tz_cont_emb_tensor, self.area_emb_tensor, self.top_k)

            case 5:
                tz_tech_emb_tensor = model.encode(tz_tech,convert_to_tensor=True)
                code_list_ = get_semantic_search_id_list(tz_tech_emb_tensor, self.area_emb_tensor, codes, self.top_k)
                score_list_ = get_semantic_search_score(tz_tech_emb_tensor, self.area_emb_tensor, self.top_k)


            case 6:
                tz_aim_emb_tensor = model.encode(tz_aim,convert_to_tensor=True)
                tz_object_name_tech_emb_tensor = model.encode(tz_object+'[SEP]'+tz_name+'[SEP]'+tz_tech,convert_to_tensor=True)
                tz_name_emb_tensor = model.encode(tz_name,convert_to_tensor=True)
                tz_cont_emb_tensor = model.encode(tz_cont,convert_to_tensor=True)
                tz_tech_emb_tensor = model.encode(tz_tech,convert_to_tensor=True)

                a = []
                b = []
                c = []
                d = []
                e = []

                for item in self.formula_emb_tensor:
                    a.append(torch.nn.functional.cosine_similarity(tz_aim_emb_tensor, item, dim=0))

                for item in self.code_area_emb_tensor:
                    b.append(torch.nn.functional.cosine_similarity(tz_object_name_tech_emb_tensor, item, dim=0))

                for item in self.code_emb_tensor:
                    c.append(torch.nn.functional.cosine_similarity(tz_name_emb_tensor, item, dim=0))

                for item in self.area_emb_tensor:
                    d.append(torch.nn.functional.cosine_similarity(tz_cont_emb_tensor, item, dim=0))

                for item in self.area_emb_tensor:
                    e.append(torch.nn.functional.cosine_similarity(tz_tech_emb_tensor, item, dim=0))
                

                full_mean_score = [((a[i]+b[i]+c[i]+d[i]+e[i])/5)*10 for i in range(len(a))]
                
                full_codes = self.codes

                full_mean_score, full_codes = (list(t) for t in zip(*sorted(zip(full_mean_score, full_codes), reverse = True)))

                code_list_ = full_codes[:self.top_k]

                score_list_ = []
                for item in full_mean_score:
                    score_list_.append(item.tolist())
                    
                score_list_ = score_list_[:self.top_k]


        return Solution_NLP(code_list_, score_list_)
    



             

