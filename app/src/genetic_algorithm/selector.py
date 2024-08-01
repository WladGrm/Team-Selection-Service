import math
import random
import numpy  as np
import pandas as pd

from tqdm     import tqdm
from time     import gmtime
from time     import strftime
from tabulate import tabulate
from colorama import Fore

def time_of_work(func):
    import time

    def wrapper(self, *args, **kwargs):
        start = time.time()
        func(self, *args, **kwargs)
        end = time.time()
        print('[{}] Время выполнения: {} секунд.'.format(func.__name__, end-start))
    return wrapper

class DNA:
    def __init__(self, count_of_genes=10, probability_of_mutation=0.01, chain=None):

        self.probability_of_mutation = probability_of_mutation
        self.count_of_genes = count_of_genes
        # print("COUNT OF GENES DNA {}".format(self.count_of_genes))

        if chain is not None:
            self.chain = chain
        else:
            # zero_one_array = [0, 1]
            self.chain = np.array([np.random.choice([0, 1], p=[0.99, 0.01]) for i in range(self.count_of_genes)])
            # self.chain = np.array([random.randint(0, 1) for i in range(self.count_of_genes)])

    def __str__(self):
        return "[DNA chain]: \n{}".format(self.chain)

    def __add__(self, other):
        crossover_point = random.randint(1, self.count_of_genes - 1)

        child_1_chain = self.chain.copy()
        child_2_chain = other.chain.copy()

        temp = child_1_chain[crossover_point:].copy()
        child_1_chain[crossover_point:] = child_2_chain[crossover_point:].copy()
        child_2_chain[crossover_point:] = temp.copy()

        child_1 = DNA(chain=child_1_chain.copy(), count_of_genes=self.count_of_genes, probability_of_mutation=self.probability_of_mutation)
        child_2 = DNA(chain=child_2_chain.copy(), count_of_genes=self.count_of_genes, probability_of_mutation=self.probability_of_mutation)
        return [child_1, child_2]

    def __invert_gen(self, gen_number):
        # print("CHAIN LEN {} | GEN NUMBER {}".format(len(self.chain), gen_number))
        self.chain[gen_number] = not self.chain[gen_number]

    def mutation(self):
        # print("COUNT OF GENES {}".format(self.count_of_genes))
        for gen in range(self.count_of_genes - 1):
            probability = random.random()
            if probability < self.probability_of_mutation:
                self.__invert_gen(gen)


# Класс подготовки данных
class Data:
    def __init__(self):

        # Таблица для хранения значений уровней компетенций проекта
        self.__project_competence_table = np.array([])

        # Таблица для хранения значений уровней компетенций сотрудников [В десятичных дробях]
        self.__employee_competence_table = np.array([])

        # Таблица для хранения ставок работников
        self.__employee_rate_table = np.array([])

        # Словарь ID сотрудников
        self.__employee_names = []

        # Словарь названий компетенций
        self.__competence_names = []

        # Верхняя допустимая граница суммарной компетенции
        self.__competence_upper_limit = 3

        # Нижняя допустимая граница суммарной компетенции
        self.__competence_lower_limit = 0

        # Путь до таблицы хранения компетенций сотрудников XLSX
        self.__employee_competence_table_path = './docs/genetic_algorithm_files/employee_competence_table.xlsx'

        # Количество проектов
        self.__project_count = 0

        self.read_employee_competence_from_xlsx()
        # self.read_project_competence_from_xlsx()
        # self.read_employee_rate_from_xlsx()

        # Количество работников
        self.employee_count = len(self.__employee_names)
        # print(self.employee_count)
        # print(self.__employee_names)

        # Настройка вывода данных в консоль
        np.set_printoptions(linewidth=np.inf)
        np.set_printoptions(threshold=np.inf)

    def __str__(self):
        return "[REQUIRED COMPETENCE VALUES]: {}\n [UPPER LIMIT]: {}\n [LOWER LIMIT]: {}".format(
            self.__project_competence_table,
            self.__competence_upper_limit,
            self.__competence_lower_limit)

    # Функция чтения данных о компетенциях сотрудников
    def read_employee_competence_from_xlsx(self):
        # print('Reading employee competences from {}'.format(self.__employee_competence_table_path))
        table = pd.read_excel(self.__employee_competence_table_path, index_col=0)
        self.__employee_competence_table = table.copy().transpose().to_numpy()
        self.__competence_names = table.copy().columns.to_list()
        self.__employee_names = table.copy().index.to_list()

    # Функция вывода имен сотрудников
    def print_employee_names(self):
        print(self.__employee_names)

    # Функция вывода названий компетенций
    def print_competence_name(self):
        print(self.__competence_names)

    # ==============================[GETTERS]===========================================================================

    # Функция, возвращающая таблицу компетенций сотрудников
    def get_employee_competence_table(self):
        return self.__employee_competence_table

    # Функция, возвращающая таблицы необходимых компетенций проектов
    def get_project_competence_table(self):
        return self.__project_competence_table

    # Функция, возвращающая имена сотрудников
    def get_employee_names(self):
        return self.__employee_names

    # Функция, возвращающая названия компетенций
    def get_competence_names(self):
        return self.__competence_names

    # Функция, возвращающая количество рассматриваемых работников
    def get_employee_count(self):
        # print(self.__employee_competence_table.shape[1])
        return self.__employee_competence_table.shape[1]

    # Функция, возвращающая количество рассматриваемых проектов
    def get_project_count(self):
        return self.__project_count

    # Функция, возвращающая количество компетенций
    def get_competence_count(self):
        return self.__project_competence_table.shape[0]

    # Получить значения функций
    def get_functions_values(self, bin_array):
        functions_values = np.sum(bin_array * self.__employee_competence_table.copy(), axis=1)
        return functions_values

    # Получить значение приспособленности,
    # полученное методом функции расстояния (method of distance function)
    # @time_of_work
    def get_fitness_value(self, bin_array):
        function_values = self.get_functions_values(bin_array)
        fitness = sum((np.array(function_values).copy() - self.__project_competence_table.copy()) ** 2)
        if math.sqrt(fitness) == 0:
            return 2
        elif math.sqrt(fitness) < 1:
            return (1 / (1 - math.sqrt(fitness))) + 1
        else:
            return 1 / math.sqrt(fitness)

    # ==============================[SETTERS]===========================================================================

    # Функция установки верхней границы допустимых решений
    def set_competence_upper_limit(self, upper_level):
        self.__competence_upper_limit = upper_level

    # Функция установки нижней границы допустимых решений
    def set_competence_lower_limit(self, lower_level):
        self.__competence_lower_limit = lower_level

    # Установка текущего проекта, для которого производится подбор сотрудников
    def set_current_project_number(self, number):
        self.__current_project_number = number
        self.read_project_competence_from_xlsx()

    def set_project_competence_list(self, list_input : list):
        """
        Метод установки списка компетенций проектов
        """
        self.__project_competence_table = np.array(list_input)
        # print(self.data.__project_competence_table.shape)
        # print(type(self.data.__project_competence_table))
        # print(self.data.__project_competence_table)
    # ==================================================================================================================

    def set_comp_upper_limit(self, upper_limit):
        """
        Метод установки верхней границы допустимых решений
        """
        self.__competence_upper_limit = upper_limit

    # Проверка, подходит ли данное решение
    def is_relevant_solution(self, bin_array):

        functions_values = self.get_functions_values(bin_array)

        for i in range(len(functions_values)):
            if not (self.__project_competence_table[i] - self.__competence_lower_limit <= functions_values[i] <=
                    self.__project_competence_table[i] + self.__competence_upper_limit):
                return False
        return True

    # Удалить сотрудников из рассмотрения для последующих решений
    def delete_employees(self, series_input):

        for i in range(len(series_input.index)):
            if series_input.index[i] in self.__employee_names:
                index = self.__employee_names.index(series_input.index[i])
                self.__employee_competence_table = np.delete(self.__employee_competence_table, index, 1)
                self.__employee_names.pop(index)
                self.employee_count = len(self.__employee_names)
                # print(self.employee_count)
                # print(self.__employee_names)
                # print("INDEX: {}".format(index))
            else:
                break


# Класс генетического алгоритма
class GA:
    def __init__(self, data=Data()):
        # Количество особей в поколении кратное 2
        self.count_of_individuals = 100

        # Количество поколений работы алгоритма
        self.count_of_generations = 400

        # Вероятность мутации каждого гена в составе цепочки ДНК
        self.probability_of_mutation = 0.000001

        # Инициализация класса Data
        self.data = data

        # Количество генов, соответствующее количеству сотрудников
        self.count_of_genes = self.data.employee_count
        # print("GA count_of_genes {}".format(self.count_of_genes))
        # self.count_of_genes = len(data.get_employee_names())

        # Матрица родительских особей
        self.__individual = []

        # Матрица приспособленности особей
        self.fitness_matrix = np.array([])

        # Матрица потомков
        self.__children_matrix = []

        # Массив хранения истории среднего значения приспособленности
        self.__average_history = []

        # Матрица возможных решений
        row_columns_names = []
        row_columns_names.extend(self.data.get_employee_names())
        row_columns_names.extend(self.data.get_competence_names())
        row_columns_names.append('FITNESS')
        self.__solutions_matrix = pd.DataFrame(columns=row_columns_names)

        # Настройка вывода данных в консоль
        np.set_printoptions(linewidth=np.inf)
        np.set_printoptions(threshold=np.inf)

        # Инициализировать начальную популяцию
        # self.__init_population()

    # Инициализировать популяцию
    # заполнить матрицу родительских особей объектами класса DNA
    def __init_population(self):
        self.__individual = []
        for i in range(self.count_of_individuals):
            self.__individual.append(
                DNA(count_of_genes = self.count_of_genes, 
                probability_of_mutation = self.probability_of_mutation))

    # Функция вывода цепочки ДНК всех родительских особей текущего поколения
    def __print_dna_matrix(self):
        for i in range(len(self.__individual)):
            print(self.__individual[i].chain)

    # Функция вывода цепочки ДНК всех дочерних особей текущего поколения
    def __print_children_matrix(self):
        print("CHILDREN MATRIX")
        for i in range(len(self.__children_matrix)):
            print(self.__children_matrix[i].chain)

    # Функция мутации гена
    def __gen_mutation(self):
        for i in range(self.count_of_individuals):
            self.__individual[i].mutation()

    # Функция скрещивания
    def __breeding(self):
        for count in range(self.count_of_individuals // 2):
            i = random.randint(0, self.count_of_individuals - 1)
            j = random.randint(0, self.count_of_individuals - 1)

            while i == j:
                j = random.randint(0, self.count_of_individuals - 1)

            two_children = self.__individual[i] + self.__individual[j]
            self.__individual.append(two_children[0])
            self.__individual.append(two_children[1])

    # Функция расчета приспособленности каждой особи
    # @time_of_work
    def __calculate_fitness(self):
        self.fitness_matrix = []

        for i in range(len(self.__individual)):
            self.fitness_matrix.append(self.data.get_fitness_value(self.__individual[i].chain))

        sum_fitness_matrix = sum(self.fitness_matrix)

        self.fitness_matrix = np.divide(self.fitness_matrix, sum_fitness_matrix)
        """
        for i in range(len(self.__individual)):
            self.fitness_matrix[i] = self.fitness_matrix[i] / sum_fitness_matrix
        """

    # Функция, определяющая методом рулетки скрещивающиеся особи
    def __get_roulette_selected(self):
        random_value = random.random()
        i = 0
        low_limit: float = 0.0
        high_limit = self.fitness_matrix[0]
        while i != len(self.__individual):
            if (random_value > low_limit) and (random_value < high_limit):
                return i
            i += 1
            low_limit = high_limit
            high_limit += self.fitness_matrix[i]

    # Функция отбора особей для следующего поколения
    # @time_of_work
    def __selection(self):
        self.__children_matrix = []

        for i in range(self.count_of_individuals):
            self.__calculate_fitness()
            j = self.__get_roulette_selected()
            self.__children_matrix.append(self.__individual[j])
            self.__individual.pop(j)

    # Функция переопределения потомков в родителей
    def __children_to_parent(self):
        self.__individual = []
        for i in range(self.count_of_individuals):
            self.__individual.append(self.__children_matrix[i])

    # Функция отбора возможных решений
    def __add_solution_option(self):
        sum_fit = 0
        for i in range(len(self.__individual)):
            fitness = self.data.get_fitness_value(self.__individual[i].chain)
            sum_fit += fitness
            if self.data.is_relevant_solution(self.__individual[i].chain):
                row = []
                row.extend(self.__individual[i].chain)
                row.extend(self.data.get_functions_values(self.__individual[i].chain))
                row.append(str(fitness))
                self.__solutions_matrix.loc[len(self.__solutions_matrix.index)] = row

        average_val = sum_fit / len(self.__individual)

        # self.__average_history.append(round(sum_fit * 100, 2))

        self.__average_history.append(round(average_val * 100, 2))
        # self.__average_history.append(len(self.__individual))
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __add_solution(self):
        average_val = 0
        for i in range(len(self.__individual)):
            row = []
            row.extend(self.__individual[i].chain)
            row.extend(self.data.get_functions_values(self.__individual[i].chain))
            fitness = self.data.get_fitness_value(self.__individual[i].chain)
            row.append(str(fitness))
            self.__solutions_matrix.loc[len(self.__solutions_matrix.index)] = row
            average_val += fitness

        average_val = average_val / len(self.__individual)
        self.__average_history.append(average_val)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Вывод данных о средней приспособленности популяции каждого поколения
    def print_hist(self):
        print("[HISTORY]")
        print(self.__average_history)

    # Функция возвращающая данные о средней приспособленности популяции каждого поколения
    def get_hist(self):
        return self.__average_history

    # Печать полученных результатов
    def print_result(self):
        print(self.data)
        if self.__solutions_matrix.empty:
            print("No solution was found")
        else:
            print("\n[SOLUTIONS]\n")
            self.__solutions_matrix = self.__solutions_matrix.drop_duplicates().sort_values('FITNESS',
                                                                            ascending=False).reset_index(drop=True)
            print(tabulate(self.__solutions_matrix, headers='keys', stralign='center', tablefmt='pipe'))
            print("________________________")
            print("* E#1 -- Employee #1")
            print("* C#1 -- Competence #1")

    # Печать информации по текущему поколению
    def __print_gen_info(self):
        for i in range(len(self.__individual)):

            print("[Individual # {:3.0f}]   [Function]:{}   [Fitness]: {:.16f}   [DNA Chain]: {}".format(i,
                                                                                    self.data.get_functions_values(
                                                                                    self.__individual[i].chain),
                                                                                    self.fitness_matrix[i],
                                                                                    self.__individual[i].chain)) 

            # print("{}".format(self.fitness_matrix[i]))

    # Запуск одного поколения генетического алгоритма
    def __run_generation(self):
        # print("COUNT OF GENES {}".format(self.count_of_genes))
        self.__gen_mutation()
        self.__breeding()
        self.__selection()
        self.__children_to_parent()
        self.__add_solution_option()
        # self.__add_solution()

    # Запуск алгоритма решения
    def solve(self):
        self.__init_population()
        """
        # ===================================
        self.__calculate_fitness()
        self.__print_gen_info()
        # ===================================
        """
        for generation in tqdm(range(self.count_of_generations),
                               ncols=100,
                               bar_format="%s{l_bar}{bar}{r_bar}" % Fore.GREEN):
            self.__run_generation()
    
        # self.print_hist()
        # self.__print_gen_info()


    # Функция печати полученных результатов в файл xlsx
    def save_result_xlsx(self):
        if self.__solutions_matrix.empty:
            print("Nothing to write")
        else:
            self.__solutions_matrix = self.__solutions_matrix.drop_duplicates().sort_values('FITNESS',
                                                                                        ascending=False).reset_index(
                                                                                        drop=True)
            self.__solutions_matrix.to_excel("output_{}.xlsx".format(strftime("%Y_%m_%d_%H_%M_%S", gmtime())))

    # Вернуть наилучшее решение из полученных при помощи генетического алгоритма
    def get_solution(self):
        if self.__solutions_matrix.empty:
            return None
        table = self.__solutions_matrix.iloc[0]
        table = pd.to_numeric(table, errors='coerce')
        # print(table[table > 0])
        return table[table > 0]


# Наследованный класс генетического алгоритма Генитор
class GaGenitor(GA):
    def __init__(self, data=Data()):
        super().__init__(data)

    # Переопределение метода селекции, в нем необходимости нет
    def __selection(self):
        pass

    # Вернуть индекс наименее приспособленной особи
    def __get_weakest_id(self):
        self.__calculate_fitness()
        return np.argmin(self.fitness_matrix, axis=0)

    # Переопределение метода скрещивания особей
    def __breeding(self):
        # self.__children_matrix = []
        # +++++++++++++++++++++++
        for count in range(self.count_of_individuals // 4):
            # Потомок занимает место наименее приспособленной особи в популяции
            i = random.randint(0, self.count_of_individuals - 1)
            j = random.randint(0, self.count_of_individuals - 1)

            while i == j:
                j = random.randint(0, self.count_of_individuals - 1)

            two_children = self.__individual[i] + self.__individual[j]
            first_ch = self.data.get_fitness_value(two_children[0])
            second_ch = self.data.get_fitness_value(two_children[0])

            if first_ch > second_ch:
                # В популяцию отправляется первый ребенок
                self.__individual[self.__get_weakest_id()] = first_ch
                pass
            else:
                # В популяцию отправляется второй ребенок
                self.__individual[self.__get_weakest_id()] = second_ch
                pass

            # Потомок занимает место наименее приспособленного родителя
            i = random.randint(0, self.count_of_individuals - 1)
            j = random.randint(0, self.count_of_individuals - 1)

            while i == j:
                j = random.randint(0, self.count_of_individuals - 1)

            two_children = self.__individual[i] + self.__individual[j]
            first_ch = self.data.get_fitness_value(two_children[0])
            second_ch = self.data.get_fitness_value(two_children[0])

            if first_ch > second_ch:
                # Родителя замещает первый ребенок
                if self.data.get_fitness_value(self.__individual[i]) < self.data.get_fitness_value(
                        self.__individual[j]):
                    self.__individual[i] = first_ch
                else:
                    self.__individual[j] = first_ch
                pass
            else:
                # Родителя замещает второй ребенок
                if self.data.get_fitness_value(self.__individual[i]) < self.data.get_fitness_value(
                        self.__individual[j]):
                    self.__individual[i] = second_ch
                else:
                    self.__individual[j] = second_ch
                pass


# Наследованный класс генетического алгоритма Метод прерывистого равновесия
class GaPunctuatedEquilibrium(GA):
    def __init__(self, data=Data()):
        super().__init__(data)

    # Переопределение функции отбора особей для следующего поколения
    # @time_of_work
    def __selection(self):
        self.__children_matrix = []
        self.__calculate_fitness()

        average_fitness = np.sum(self.fitness_matrix.copy()) / self.fitness_matrix.shape[0]

        for i in range(len(self.__individual)):
            if self.data.get_fitness_value(self.__individual[i]) >= average_fitness:
                self.__children_matrix.append(self.__individual[i])


# Наследованный класс генетического алгоритма Генетический алгоритм с нефиксированным размером популяции
class GaUnfixedPopulationSize(GA):
    def __init__(self, data=Data()):
        super().__init__(data)
        # Возраст только родившейся особи
        self.__init_age = 0

        # Матрица возрастов особей
        # self.__ages = [self.__init_age for i in range(self.count_of_individuals)]
        # self.__child_ages = []

        # Максимально возможная продолжительность жизни особи
        self.__MaxLT = 6

        # Минимально возможная продолжительность жизни особи
        self.__MinLT = 1

        # Текущий возраст особи
        self.__current_age = [self.__init_age for i in range(self.count_of_individuals)]
        self.__child_current_age = []

        # Возможный возраст особи
        self.__age = []
        self.__child_age = []

        #
        self.__nu = (1/2) * (self.__MaxLT - self.__MaxLT)

        self.__MinFit = 0
        self.__MaxFit = 0
        self.__AvgFit = 0

    # Функция старения особей
    def __make_individual_older(self):
        for i in range(len(self.__age)):
            self.__age[i] += 1

    # Функция мутации гена
    def __gen_mutation(self):
        for i in range(len(self.__individual)):
            self.__individual[i].mutation()

    # Вернуть минимальное значение приспособленности
    def __get_min_fitness(self):
        # self.__calculate_fitness()
        return self.fitness_matrix.min()

    # Вернуть максимальное значение приспособленности
    def __get_max_fitness(self):
        # self.__calculate_fitness()
        return self.fitness_matrix.max()

    # Вернуть среднее значение приспособленности
    def __get_avg_fitness(self):
        # self.__calculate_fitness()
        return self.fitness_matrix.mean()

    # Очистка значений глобальных параметров
    def __free_param(self):
        self.__fitness_matrix = []
        self.__MinFit = 0
        self.__MaxFit = 0
        self.__AvgFit = 0

    # Расчет продолжительности жизни особи individual_i
    def __get_lifetime(self, individual_i):
        if self.__AvgFit >= self.fitness_matrix[individual_i]:
            lifetime = self.__MinLT + self.__nu * (\
                        (self.fitness_matrix[individual_i].copy() - self.__get_min_fitness()) / \
                        (self.__AvgFit - self.__get_min_fitness()))
        else:
            lifetime = (1/2) * (self.__MinLT + self.__MaxLT) + self.__nu * ((self.fitness_matrix[individual_i].copy()\
                                                                           - self.__AvgFit)/(self.__get_max_fitness()-\
                                                                                             self.__AvgFit))
        return lifetime

    # Рассчитать продолжительности жизней особей
    def __calculate_ages_individuals(self):
        for i in range(len(self.__individual)):
            self.__age[i] = self.__get_lifetime(i)

    def __calculate_ages_children(self):
        for i in range(len(self.__children_matrix)):
            self.__child_age[i] = self.__get_lifetime(i)

    # Функция переопределения потомков в родителей
    def __children_to_parent(self):
        self.__individual = []
        for i in range(len(self.__children_matrix)):
            self.__individual.append(self.__children_matrix[i])
            self.__age.append(self.__child_age[i])
            self.__current_age.append(self.__child_current_age[i])

        self.__children_matrix = []
        self.__child_age = []
        self.__child_current_age = []

    # Убрать особи с вышедшим сроком жизни
    def __kill_older(self):
        for i in range(len(self.__individual)):
            if self.__current_age[i] > self.__age[i]:
                self.__age.pop(i)
                self.__current_age.pop(i)
                self.__individual.pop(i)
                # pass

    # Функция скрещивания
    def __breeding(self):
        self.__children_matrix = []
        for count in range(len(self.__individual) // 2):
            i = random.randint(0, len(self.__individual) - 1)
            j = random.randint(0, len(self.__individual) - 1)

            while i == j:
                j = random.randint(0, len(self.__individual) - 1)

            two_children = self.__individual[i] + self.__individual[j]
            self.__children_matrix.append(two_children[0])
            self.__child_current_age.append(self.__init_age)

            self.__children_matrix.append(two_children[1])
            self.__child_current_age.append(self.__init_age)

    # Инициализировать популяцию
    # заполнить матрицу родительских особей объектами класса DNA
    def __init_population(self):
        self.__individual = []
        for i in range(self.count_of_individuals):
            self.__individual.append(DNA(count_of_genes=self.count_of_genes, probability_of_mutation=self.probability_of_mutation))

        # Оценка возраста начальной популяции
        self.__calculate_fitness()
        self.__MinFit = self.__get_min_fitness()
        self.__MaxFit = self.__get_max_fitness()
        self.__AvgFit = self.__get_avg_fitness()
        self.__calculate_ages_individuals()
        # Очистить
        self.__free_param()

    # Запуск одного поколения генетического алгоритма
    def __run_generation(self):
        self.__make_individual_older()
        self.__gen_mutation()
        self.__calculate_fitness()
        self.__breeding()
        self.__free_param()

        self.__calculate_fitness()
        self.__calculate_ages_children()
        self.__children_to_parent()
        self.__kill_older()
        self.__add_solution_option()
        self.__free_param()


class Solver:
    def __init__(self, data=Data()):

        # Инициализация объекта класса Data()
        self.data = data

        # Количество попыток на каждую итерацию
        self.__try_count = 5

        # Серия для хранения текущего решения
        self.__current_solution_series = pd.Series(dtype='float64')

        # Название файла вывода
        self.__output_file_name = "output1.xlsx"

        # Начальная строка записи в таблице Excel вывода
        self.__row = 1

        # Расстояния между полученными решениями
        self.__space = 3

        # Название используемого в текущий момент алгоритма
        self.__alg_name = "Genetic Algorithm"

        # Текущий алгоритм подбора групп
        self.__current_algorithm = 0

        # Флаг отображения логирования
        self.__logger_flag = 0

        # Объект генетического алгоритма
        self.genetic_algorithm = None

    
    def set_current_algorithm(self, algorithm_index):
        """
        ### Метод установки текущего алгоритма решения:

        - 0 - Genetic Algorithm
        - 1 - Genetic Algorithm Genitor
        - 2 - Genetic Algorithm Punctuated Equilibrium
        - 3 - Genetic Algorithm Unfixed Population Size
        """
        self.__current_algorithm = algorithm_index

    # Метод установки имени выводного файла
    def set_output_file_name(self, name):
        self.__output_file_name = name

    # Метод установки количества попыток подбора
    def set_try_count(self, count):
        self.__try_count = count

    # Метод включения логирования
    def enable_logger(self):
        self.__logger_flag = 1

    def init_alg(self):
        """
        Инициализировать выбранный алгоритм
        """
        if self.__current_algorithm == 0:
            self.genetic_algorithm = GA(data=self.data)
            self.__alg_name = "Genetic Algorithm"
        elif self.__current_algorithm == 1:
            self.genetic_algorithm = GaGenitor(data=self.data)
            self.__alg_name = "Genetic Algorithm Genitor"
        elif self.__current_algorithm == 2:
            self.genetic_algorithm = GaPunctuatedEquilibrium(data=self.data)
            self.__alg_name = "Genetic Algorithm Punctuated Equilibrium"
        elif self.__current_algorithm == 3:
            self.genetic_algorithm = GaUnfixedPopulationSize(data=self.data)
            self.__alg_name = "Genetic Algorithm Unfixed Population Size"
        else:
            self.genetic_algorithm = GA(data=self.data)
            self.__alg_name = "Genetic Algorithm"

    # Функция получения решения для текущего проекта
    def __solution_for_current_project(self):

        for i in range(self.__try_count):
            if self.__logger_flag == 1:
                print("[TRY #{}]==============================================================================".format(i+1))
                print("[USING]--{}--".format(self.__alg_name))
            
            self.genetic_algorithm.solve()

            if self.__logger_flag == 1:
                self.genetic_algorithm.print_result()

            if self.genetic_algorithm.get_solution() is not None:
                self.__current_solution_series = self.genetic_algorithm.get_solution()

                # Удалить сотрудников
                # self.data.delete_employees(self.__current_solution_series)

                if self.__logger_flag == 1:
                    print(self.__current_solution_series)
                return 1
        return 0

    # Записать решение в файл
    def __write_solution_to_excel(self, writer):

        self.__current_solution_series.to_excel(writer, startrow=self.__row, startcol=0)
        self.__row = len(self.__current_solution_series.index) + self.__row + self.__space + 1
        self.__current_solution_series = None
        # print(self.__row)

    @time_of_work
    def solve(self):
        """Метод поиска решений"""

        writer = pd.ExcelWriter(self.__output_file_name, engine='xlsxwriter')
    
        if self.__solution_for_current_project() == 0:
            pass
        else:
            self.__current_solution_series = self.__current_solution_series.rename("PROJECT")
            self.__write_solution_to_excel(writer)

        writer.close()

    def set_count_of_individuals(self, count):
        """
        Метод установки количество особей для расчета в генетическом алгоритме
        """
        if count % 2 == 0:
            self.genetic_algorithm.count_of_individuals = count
        else:
            pass

    def set_probability_of_mutation(self, probability):
        """
        Метод установки вероятности мутации:
        """
        self.genetic_algorithm.probability_of_mutation = probability

    def set_count_of_generations(self, count):
        """
        Метод установки количества поколений ГА
        """
        self.genetic_algorithm.count_of_generations = count

        


