from src.genetic_algorithm.selector import Data
from src.genetic_algorithm.selector import Solver
from src.natural_processing.nlp     import NLP

from tkinter     import *
from tkinter     import ttk
from tkinter.ttk import Combobox

from docs.config import OUTPUT_FILE_DIR_PATH
from docs.config import TZ_FILE_DIR_PATH

class MainForm:
    # pass
    def __init__(self):
        # Список названий вариаций генетического алгоритма
        self.__GA_algorithms_name_list = [
            'Genetic Algorithm (GA)',
            'GA Genitor',
            'GA Punctuated Equilibrium',
            'GA Unfixed Population Size']

        # Соответствующие индексы названиям генетическим алгоритмам
        self.__GA_algorithms_index = [
            0,
            1, 
            2,
            3
        ]
        
        # Список названий текстовых сравнений
        self.__COMP_algorithms_name_list = [
            "Цель - Формула",
            "Объект - Области исследований + Шифр",
            "Назначение - Шифр",
            "Состав - Области исследований",
            "Технические характеристики - \n Области исследований",
            "Средний"
        ]

        # Лист возможных количеств возможных итераций-попыток
        self.__GA_possible_try_list = [i for i in range(1, 16)]

        # Лист количеств полоколений решения
        self.__GA_generation_count_list = [i*50 for i in range(1, 21)]

        # Лист возможных верхних границ
        self.__GA_competence_upper_limit_list = [i for i in range(0, 51)]

        # Лист вероятностей мутации
        self.__GA_probability_of_mutation_list = [0.00001*i for i in range(0, 1000)]

        # Литс количеств особей
        self.__GA_count_of_individuals_list = [2*i for i in range(25, 100)]

        # Сущность исходных данных Генетических алгоритмов
        self.__GA_init_data = Data()

        # Сущность решателя Генетически алгоритмом
        self.__GA_solver = Solver(data = self.__GA_init_data)

        # Главное окно
        self.mainWindow = Tk()

        # Состояние Флага ЛОГИРОВАНИЕ РЕШЕНИЯ В КОНСОЛЬ
        self.__var_logger_flag = IntVar()
        self.__var_logger_flag.set(1)

        # Имя файла для сохранения решения
        self.__output_file_path = StringVar()
        self.__output_file_path.set(OUTPUT_FILE_DIR_PATH)

        # Путь к файлу с подбираемым ТЗ
        self.__input_file_TZ_path = StringVar()
        self.__input_file_TZ_path.set(TZ_FILE_DIR_PATH)
        # self.__input_file_TZ_path.set(".\docs\\nlp_files\\tz.txt".format(os.path.dirname(__file__)))

        # Переменная хранения состояния кнопки "Решить"
        self.__solve_button_state = 'normal'

        # Настройка главного окна
        self.mainWindow.title('Приложение подбора проектных групп')
        self.mainWindow.geometry("450x400")
        self.mainWindow.resizable(False, False)
        # self.mainWindow.iconbitmap(default="favicon.ico")
    
        # Кнопка "Рассчитать"
        self.__solve_button = ttk.Button(
            self.mainWindow, 
            text = "Рассчитать", 
            command = self.__solve_button_push,
            state = self.__solve_button_state
        )

        # Выпадающий список генетических алгоритмов
        self.__GA_selection_combobox = Combobox(
            self.mainWindow,
            state="readonly", 
            textvariable = self.__GA_algorithms_index, 
            values = self.__GA_algorithms_name_list
        )
        self.__GA_selection_combobox.current(0)

        # Выпадающий список алгоритмов текстографического сравнения
        self.__COMP_selection_combobox = Combobox(
            self.mainWindow,
            state="readonly",
            textvariable = self.__COMP_algorithms_name_list,
            values = self.__COMP_algorithms_name_list
        )
        self.__COMP_selection_combobox.current(0)

        # Выпадающий список количества возможных итераций решения
        self.__GA_try_selection_combobox = Combobox(
            self.mainWindow,
            state="readonly",
            textvariable = self.__GA_possible_try_list,
            values = self.__GA_possible_try_list
        )
        self.__GA_try_selection_combobox.current(0)

        # Выпадающий список количеств полоколений решения
        self.__GA_generation_count_combobox = Combobox(
            self.mainWindow,
            state="readonly",
            textvariable = self.__GA_generation_count_list,
            values = self.__GA_generation_count_list
        )
        self.__GA_generation_count_combobox.current(1)

        # Выпадающий список верхних допустимыз границ решения
        self.__GA_competence_upper_limit_combobox = Combobox(
            self.mainWindow,
            state="readonly",
            textvariable = self.__GA_competence_upper_limit_list,
            values = self.__GA_competence_upper_limit_list
        )
        self.__GA_competence_upper_limit_combobox.current(5)

        # Выпадающий список количеств особей
        self.__GA_count_of_individuals_combobox = Combobox(
            self.mainWindow,
            state="readonly",
            textvariable = self.__GA_count_of_individuals_list,
            values = self.__GA_count_of_individuals_list
        )
        self.__GA_count_of_individuals_combobox.current(5)
    
        # Выпадающий список вероятностей мутации
        self.__GA_probability_of_mutation_combobox = Combobox(
            self.mainWindow,
            state="readonly",
            textvariable = self.__GA_probability_of_mutation_list,
            values = self.__GA_probability_of_mutation_list
        )
        self.__GA_probability_of_mutation_combobox.current(1)

        # Чекбокс флага логирования
        self.__logger_check = ttk.Checkbutton(
            variable=self.__var_logger_flag,
            onvalue=1, 
            offvalue=0,
            text="Логирование в консоль")

        # Поле ввода пути сохранения файла
        self.__output_file_path_entry = Entry(
            textvariable = self.__output_file_path
        )

        # Поле ввода пути к ТЗ
        self.__input_file_TZ_path_entry = Entry(
            textvariable = self.__input_file_TZ_path
        )

        # Сепаратор первый сверху
        self.__separator_obj_first = ttk.Separator(self.mainWindow, orient="horizontal")
        self.__separator_obj_second = ttk.Separator(self.mainWindow, orient="horizontal")

        # Подписи
        self.__label_ga_selection_combobox = Label(text="Генетический алгоритм для решения")
        self.__label_try_selection_combobox = Label(text="Количество попыток подсчета")
        self.__label_output_entry = Label(text="Путь сохранения результата")
        self.__label_input_TZ_path = Label(text="Путь к техническому заданию")
        self.__label_texts_compare_type = Label(text="Параметр сравнения текстов")
        self.__label_generation_count = Label(text="Количество поколений")
        self.__label_count_of_individuals = Label(text="Количество особей")
        self.__label_probability_of_mutation = Label(text="Вероятность мутации")
        self.__label_competence_upper_limit = Label(text="Верхняя допустимая граница решения")

        # Отображение элементов интерфейса 
        self.__label_input_TZ_path.place(x = 10, y = 10)
        self.__input_file_TZ_path_entry.place(x = 13, y = 35, width = 420)

        self.__label_output_entry.place(x = 10, y = 60)
        self.__output_file_path_entry.place(x = 13, y = 85, width = 420)

        self.__separator_obj_second.place(x = 10, y = 112, width = 250, relwidth=0.4)

        self.__label_texts_compare_type.place(x = 10, y = 120)
        self.__COMP_selection_combobox.place(x = 230, y = 120, width = 205) 

        self.__label_ga_selection_combobox.place(x = 10, y = 145)
        self.__GA_selection_combobox.place(x = 230, y = 145, width = 205)

        self.__label_try_selection_combobox.place(x = 10, y = 170)
        self.__GA_try_selection_combobox.place(x = 230, y = 170, width = 205)

        self.__label_generation_count.place(x = 10, y = 195)
        self.__GA_generation_count_combobox.place(x = 230, y = 195, width = 205)

        self.__label_count_of_individuals.place(x = 10, y = 220)
        self.__GA_count_of_individuals_combobox.place(x = 230, y = 220, width = 205)

        self.__label_probability_of_mutation.place(x = 10, y = 245)
        self.__GA_probability_of_mutation_combobox.place(x = 230, y = 245, width = 205)

        self.__label_competence_upper_limit.place(x = 10, y = 270)
        self.__GA_competence_upper_limit_combobox.place(x = 230, y = 270, width = 205)

        self.__separator_obj_first.place(x = 10, y = 360, width = 250, relwidth=0.4)

        self.__solve_button.place(x = 180, y = 368, width = 260)
        # self.__solve_button.place(x = 350, y = 220, width = 90)     # TODO: Вариант 2
        self.__logger_check.place(x = 10, y = 370)

    def __solve_button_push(self):
        """
        Событие: нажатие кнопки "Решить"
        """
        # ------- Часть обработки естественного языка -------

        nlp_module = NLP()
        # Установить путь к ТЗ
        nlp_module.set_file_path(self.__input_file_TZ_path.get())
        # Установить метод сравнения 
        nlp_module.set_mode_param(self.__COMP_selection_combobox.current() + 1)

        # TODO: Отладка
        # TODO: Добавить элемент выбора
        nlp_module.set_top_k(4)

        # Запуск обработки естественного языка
        result = nlp_module.solve()

        # TODO: Отладка
        print(result.code_list)
        print(result.score_list)

        # ------- Часть работы генетического алгоритма -------

        # Установить текущий алгоритм для решения
        self.__GA_solver.set_current_algorithm(self.__GA_selection_combobox.current() + 1)
        # Инициализация выбранного алгоритма
        self.__GA_solver.init_alg()
        # Установить количество попыток решения
        self.__GA_solver.set_try_count(
            self.__GA_possible_try_list[self.__GA_try_selection_combobox.current()])
        # Установить файл вывода работы алгоритма
        self.__GA_solver.set_output_file_name(self.__output_file_path.get())
        # Установить количество поколений решения
        self.__GA_solver.set_count_of_generations(
            self.__GA_generation_count_list[self.__GA_generation_count_combobox.current()])
        # Установить количество особей генетического алгоритма
        self.__GA_solver.set_count_of_individuals(
            self.__GA_count_of_individuals_list[self.__GA_count_of_individuals_combobox.current()])
        # Установить вероятность мутации
        self.__GA_solver.set_probability_of_mutation(
            self.__GA_probability_of_mutation_list[self.__GA_probability_of_mutation_combobox.current()])
        # Установить верхнюю допустимую границу решения
        self.__GA_init_data.set_comp_upper_limit(
            self.__GA_competence_upper_limit_list[self.__GA_competence_upper_limit_combobox.current()])
        
        # Передача параметров после естественной обработки языка
        self.__GA_init_data.set_project_competence_list(result.score_list)

        # TODO: Запуск решения генетическим алгоритмом
        # Необходимо удостовериться, что количество компетенций в расчете совпадает как у проекта 
        # так и у сотрудников

        # self.__GA_solver.enable_logger()
        self.__GA_solver.solve()

        # Вывод в консоль текущих парамтеров работы 
        if self.__var_logger_flag.get() == 1:
            self.__log_current_parameters()


    def __log_current_parameters(self):
        """
        Метод отображения параметров работы системы 
        """
        print(
            f"""
\033[2;31;43m ЕСТЕСТВЕННАЯ ОБРАБОТКА ЯЗЫКА \033[0;0m
Путь к ТЗ                                    {self.__input_file_TZ_path.get()}
Метод сравнения                              {self.__COMP_selection_combobox.current() + 1} -- {self.__COMP_selection_combobox.get()}
\033[2;31;43m ГЕНЕТИЧЕСКИЙ АЛГОРИТМ        \033[0;0m    
Выбранный Генетический алгоритм              {self.__GA_selection_combobox.current() + 1} -- {self.__GA_selection_combobox.get()}
Количество попыток решения                   {self.__GA_try_selection_combobox.get()}
Файл вывода работы алгоритма                 {self.__output_file_path.get()}
Количество поколений решения                 {self.__GA_generation_count_combobox.get()}
Количество особей генетического алгоритма    {self.__GA_count_of_individuals_combobox.get()}
Вероятность мутации особей                   {self.__GA_probability_of_mutation_combobox.get()}
Верхняя допустимая граница решения           {self.__GA_competence_upper_limit_combobox.get()}

""")
 