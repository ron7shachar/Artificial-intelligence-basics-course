import csv
import random

import numpy as np
import math

# Assign array


detail_heuristic = {}


def find_path(starting_locations, goal_locations, search_method: 2, detail_output: False):
    surfin_USA = Surfin_USA()
    surfin_USA.find_path(starting_locations, goal_locations, search_method, detail_output)


from test import *


# 5. A genetic algorithm. Population size is 10. In your submitted answers,
# containing the answers to the questions, explain the idea of your genetic
# algorithm (how are you representing solutions, how do you assess solution
# quality, how do you combine solutions and what is your mutation).
class Genetic_algorithm():
    def __init__(self, neighboring_state):
        hill_climbing = Hill_climbing(neighboring_state)
        self.pairing_districts = hill_climbing.pairing_districts
        self.graph = {}
        self.start = None
        self.end = None
        self.population_size = 10
        self.generations = 1000
        self.mutation_rate = 0.05
        self.population = None
        self.fine = 20
        self.max_lenght = 400

    def genetic_algorithm(self, graph, starts, goals, detail_output):
        pathes = []
        self.graph = graph
        pairing = self.pairing_districts(starts, goals)
        if type(pairing) == str: return "No path found"
        for goal, start in pairing.items():
            self.start = start
            self.end = goal
            self.population = self.initialize_population()
            path = self.evolve()
            if self.valid_path(path):
                pathes.append(path)
        return pathes

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            path = self.random_path()
            population.append(path)
        return population

    def random_path(self):
        path = [self.start]
        current = self.start
        i = 0
        while current != self.end and self.max_lenght > i:
            neighbors = list(self.graph[current])
            next_node = random.choice(neighbors)
            path.append(next_node)
            current = next_node
            i += 1
        return path

    def fitness(self, path):
        total_weight = 0
        for i in range(len(path) - 1):
            if path[i + 1] in self.graph[path[i]]:
                total_weight += 1
            else:
                total_weight += self.fine
        return 1 / total_weight

    def selection(self):
        fitness_scores = [(self.fitness(path), path) for path in self.population]
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        selected_paths = [path for _, path in fitness_scores[:self.population_size // 2]]
        return selected_paths

    def crossover(self, parent1, parent2):
        cut = random.randint(1, min(len(parent1), len(parent2)) - 2)
        child = parent1[:cut]
        current = child[-1]
        while current != self.end:
            if current in parent2:
                next_index = parent2.index(current) + 1
                if next_index < len(parent2):
                    next_node = parent2[next_index]
                else:
                    next_node = None
            else:
                next_node = None

            if next_node is None or next_node in child or next_node not in self.graph[current]:
                next_node = random.choice(list(self.graph[current]))

            if next_node in child:
                break
            child.append(next_node)
            current = next_node
        if child[-1] != self.end:
            child.append(self.end)
        return child

    def mutate(self, path):
        if random.random() < self.mutation_rate:
            idx = random.randint(1, len(path) - 2)
            current = path[idx - 1]
            neighbors = list(self.graph[current])
            new_node = random.choice(neighbors)
            if new_node not in path:
                path[idx] = new_node
        return path

    def evolve(self):
        for generation in range(self.generations):
            selected_paths = self.selection()
            next_generation = []
            for _ in range(self.population_size):
                parent1 = random.choice(selected_paths)
                parent2 = random.choice(selected_paths)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_generation.append(child)
            self.population = next_generation
            if generation == 2:
                detail_heuristic[self.end] = [{'fitness' : round(self.fitness(child),3),'child ':child } for child in self.population]
            if generation % 100 == 0:
                best_path = max(self.population, key=self.fitness)
                # print(f'Generation {generation}: Best Path {best_path}, Fitness {self.fitness(best_path)}')

        best_path = max(self.population, key=self.fitness)
        return best_path

    def valid_path(self, path):
        cuntry = self.start
        for neighbor in path:
            if neighbor != self.start:
                if neighbor in self.graph[cuntry]:
                    cuntry = neighbor
                else:
                    return False
        return cuntry == self.end


class Beam_search():
    def __init__(self, neighboring_state):
        hill_climbing = Hill_climbing(neighboring_state)
        self.heuristic = hill_climbing.heuristic
        self.k = 3

    def beam_search(self, graph, starts, goals, detail_output):
        pathes = []
        pairs = self.pairing_districts(starts, goals)
        if type(pairs) == str: return "No path found"
        path = {goal: {goal: None} for goal in pairs}
        states = {goal: [goal] for goal in pairs}
        visited = {goal: [] for goal in pairs}
        while pairs:
            for goal, start in pairs.items():
                next_state = []
                front_neighbors = []
                for k_state in states[goal]:
                    for neighbor in graph[k_state]:
                        if neighbor == start:
                            path[goal][neighbor] = k_state
                            pathes.append(self.reconstruct_path(path[goal], start, goal))
                            pairs.pop(goal)
                            break
                        if neighbor not in front_neighbors and neighbor not in path[goal].keys():
                            next_state.append({"value": self.heuristic(neighbor, [start]), "parent": k_state,"neighbor" :neighbor })
                            front_neighbors.append(neighbor)
                    if goal not in pairs.keys(): break
                if goal not in pairs.keys(): break

                next_state.sort(key=lambda x: x["value"])

                if len(next_state) <= self.k:
                    k_next_state = next_state
                else:
                    k_next_state = next_state[:self.k]

                print([neighbor['neighbor'] for neighbor in  k_next_state ])


                if next_state is None: return "No path found"
                # if detail_output:
                #     detail_heuristic[start][next_state] = min_cost
                for state in k_next_state:
                    path[goal][state["neighbor" ]] = state["parent"]
                visited[goal].append(next_state)
                states[goal] = [neighbor['neighbor'] for neighbor in  k_next_state ]
        return pathes

    def reconstruct_path(self, path, start, goal):
        path_ = [start]
        while goal not in path_:
            path_.append(path[path_[-1]])
        return path_

    def pairing_districts(self, starts, goals):
        starts = starts.copy()
        goals = goals.copy()
        heuristic = {}
        for goal in goals:
            heuristic[goal] = None
            min_value = 1000
            for start in starts:
                value = self.heuristic(goal, [start])
                if value < min_value and start not in heuristic.values():
                    min_value = value
                    heuristic[goal] = start
        if None in heuristic.values():
            return ("No path")
        else:
            return (heuristic)


class Simulated_annealing():
    def __init__(self, neighboring_state):
        hill_climbing = Hill_climbing(neighboring_state)
        self.heuristic = hill_climbing.heuristic
        self.reconstruct_path = hill_climbing.reconstruct_path
        self.pairing_districts = hill_climbing.pairing_districts

    def simulated_annealing(self, graph, starts, goals, detail_output, alpha, temperature, max_time):
        pathes = []
        pairing = self.pairing_districts(starts, goals)
        if type(pairing) == str: return "No path found"
        path = {goal: {} for goal in pairing}
        state = {goal: goal for goal in pairing}
        Tmin = temperature * alpha ** max_time
        while temperature > Tmin:
            for goal, start in pairing.items():
                next_state = state[goal]
                neighbors = graph[state[goal]]
                work_process = ''
                min_cost = self.heuristic(state[goal], [start])
                while neighbors:
                    neighbor = random.choice(neighbors)
                    if neighbor == start:
                        path[goal][neighbor] = state[goal]
                        pathes.append(self.reconstruct_path(path[goal], start, goal))
                        pairing.pop(goal)
                        break
                    heuristic = self.heuristic(neighbor, [start])
                    if heuristic <= min_cost:

                        next_state = neighbor
                        work_process = work_process + ' try :' + str(neighbor) + ' probability : 1'
                        break
                    else:
                        ap = math.exp((min_cost - heuristic) / temperature)
                        work_process += ' try :' + str(neighbor) + ' probability : ' + str(ap)
                        if ap > random.uniform(0.0, 1.0):
                            next_state = neighbor
                            break
                if detail_output:
                    print(work_process)
                    print(f' T : {temperature}  next county : {str(next_state)}')

                if next_state is None: break
                if goal not in pairing: break
                # if next_state is None: return "No path found"
                if next_state not in path[goal]:
                    path[goal][next_state] = state[goal]
                state[goal] = next_state
            temperature *= alpha

        return pathes

    def reconstruct_path(self, path, start, goal):
        path_ = [start]
        while goal not in path_:
            path_.append(path[path_[-1]])
        return path_
class Hill_climbing():
    def __init__(self, neighboring_state):
        self.heuristic = A_stare(neighboring_state).heuristic

    def clear_path(self, path, start, end):
        county = path[start]
        while path:
            county = path.pop(county)
            if county == end:
                return

    def hill_climbing(self, graph, starts, goals):
        starts = np.array(starts)
        goals = np.array(goals)
        for i in range(5):
            np.random.shuffle(starts)
            np.random.shuffle(goals)
            pathes = self.hill_climbing_iteration(graph, starts, goals)
            if type(pathes) is list:
                return pathes
        return "No path found"

    def hill_climbing_iteration(self, graph, starts, goals):
        pathes = []
        pairing = self.pairing_districts(starts, goals)
        if type(pairing) == str: return "No path found"
        path = {goal: {} for goal in pairing}
        state = {goal: goal for goal in pairing}
        visited = {goal: [] for goal in pairing}
        while pairing:
            for goal, start in pairing.items():
                next_state = None
                min_cost = 1000
                for neighbor in graph[state[goal]]:
                    if neighbor == start:
                        path[goal][neighbor] = state[goal]
                        pathes.append(self.reconstruct_path(path[goal], start, goal))
                        pairing.pop(goal)
                        break
                    heuristic = self.heuristic(neighbor, [start])
                    if heuristic < min_cost and neighbor not in visited[goal]:
                        min_cost = heuristic
                        next_state = neighbor
                if goal not in pairing.keys(): break
                if next_state is None: return "No path found"
                path[goal][next_state] = state[goal]
                visited[goal].append(next_state)
                state[goal] = next_state
        return pathes

    def reconstruct_path(self, path, start, goal):
        path_ = [start]
        while True:
            if path_[-1] == goal:
                break
            path_.append(path[path_[-1]])
        return path_

    def pairing_districts(self, starts, goals):
        pairs_m = {}
        pairs = []
        attempt = len(starts)^2
        starts = starts.copy()
        goals = goals.copy()
        for goal in goals:
            for start in starts:
                if start.color == goal.color:
                    if goal not in pairs_m.keys():pairs_m[goal] = {}
                    pairs_m[goal][start] = self.heuristic(goal, [start])
        for i in range(attempt):
            value=0
            starts = list(starts.copy())
            random.shuffle(goals)
            heuristic = {}
            for goal in goals:
                while pairs_m[goal]:
                    start = random.choice(starts)
                    if start in pairs_m[goal].keys():break
                heuristic[goal] = start
                value += pairs_m[goal][start]
                starts.remove(start)
            pairs.append({'value':value ,  'heuristic' : heuristic})

        return (min(pairs, key=lambda p: p['value'])['heuristic'])
class A_stare():
    def __init__(self, neighboring_state):
        self.neighboring_state = neighboring_state

    def a_star(self, graph, starts, goals, detail_output):

        # Defining the variables
        pathes = []
        open_set = {start: [start] for start in starts}
        came_from = {start: {start: None} for start in starts}
        gscore = {start: {start: 0} for start in starts}
        hscore = {start: {start: 0} for start in starts}

        while open_set:
            for start in open_set.keys():
                if not open_set[start]: return "No path found"
                # find the optimal move
                min_node = open_set[start][0]
                min_val = hscore[start][min_node] + gscore[start][
                    min_node]  # find node in openset with lowest hscore value
                for node in open_set[start]:
                    val = hscore[start][node] + gscore[start][node]
                    if val < min_val:
                        min_val = val
                        min_node = node
                current = min_node  # set that node to current
                # Checking whether we have reached the destination
                for goal in goals:
                    if current == goal and goal.color == start.color:
                        # save the path and delete the goal
                        pathes.append(self.reconstruct_path(came_from[start], current))
                        goals.remove(goal)
                        starts.remove(start)
                        open_set.pop(start)
                        hscore.pop(start)
                        gscore.pop(start)
                        break
                if start not in open_set: break
                # update the front in calculate the heuristics
                open_set[start].remove(current)  # remove node from set to be evaluated and
                for neighbor in graph[current]:  # check neighbors of current node
                    if neighbor not in came_from[start].keys():  # ignore neighbor node if its already evaluated
                        open_set[start].append(neighbor)
                        came_from[start][neighbor] = current  # record the best path untill now
                        gscore[start][neighbor] = gscore[start][current] + 1
                        hscore[start][neighbor] = self.heuristic(neighbor, goals)
                        if detail_output and gscore[start][neighbor] == 1:
                            detail_heuristic[start][neighbor] = hscore[start][neighbor]
                gscore[start].pop(current)
                hscore[start].pop(current)
        return pathes

    def heuristic(self, neighbor, goals):
        # find the shortest pass between the goals
        hscore = None
        for goal in goals:
            heuristic = self.us_state_code_bfs(neighbor, goal)
            if hscore is None:
                hscore = heuristic
            elif hscore > heuristic:
                hscore = heuristic
        return hscore

    def us_state_code_bfs(self, neighbor, goal):
        # find the shortest pass to a goal with bfs
        start = neighbor.us_state_code
        goal = goal.us_state_code
        explored = []
        queue = [[start]]
        if start == goal:
            return 0
        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node not in explored:
                neighbours = self.neighboring_state[node]
                for neighbour in neighbours:
                    new_path = list(path)
                    new_path.append(neighbour)
                    queue.append(new_path)
                    if neighbour == goal:
                        return len(new_path)
                explored.append(node)
        return 1000

    def reconstruct_path(self, came_from, current):
        # extract the pass from the tree
        final_path = [current]
        while current in came_from:
            current = came_from[current]
            if current is None:
                return final_path[::-1]
            final_path.append(current)
class County():
    def __init__(self, name, us_state_code):
        self.name = name
        self.us_state_code = us_state_code
        self.connections = []
        self.color = None
        self.group = None

    def add_connection(self, county):
        if county not in self.connections and county != self:
            self.connections.append(county)

    ##### print the county
    def __repr__(self):
        # if self.color == None:
        return f"{self.name}, {self.us_state_code} "
        # if self.color == 'Blue':
        #     return f"{self.name}, {self.us_state_code} (B) ; "
        # color = "(R)"
        # return f"{self.name}, {self.us_state_code} {color} ; "


class Surfin_USA():
    def __init__(self):
        self.counties = []
        self.neighboring_state = {}

        self.graph = {}
        self.upload_data()
        self.goal_locations = []
        self.starting_locations = []

    def find_path(self, starting_locations, goal_locations, search_method, detail_output):
        if self.precheck_input(starting_locations, goal_locations):
            if detail_output:
                for starting_location in self.starting_locations:
                    detail_heuristic[starting_location] = {}
            match search_method:
                case 1:
                    pathes = A_stare(self.neighboring_state).a_star(graph=self.graph, starts=self.starting_locations,
                                                                    goals=self.goal_locations,
                                                                    detail_output=detail_output)
                case 2:
                    detail_output = False
                    pathes = Hill_climbing(self.neighboring_state).hill_climbing(graph=self.graph,
                                                                                 starts=self.starting_locations,
                                                                                 goals=self.goal_locations)
                case 3:

                    pathes = Simulated_annealing(self.neighboring_state).simulated_annealing(graph=self.graph,
                                                                                             starts=self.starting_locations,
                                                                                             goals=self.goal_locations,
                                                                                             detail_output=False,
                                                                                             alpha=0.91,
                                                                                             temperature=1.1,
                                                                                             max_time=1000)

                case 4:
                    pathes = Beam_search(self.neighboring_state).beam_search(graph=self.graph,
                                                                             starts=self.starting_locations,
                                                                             goals=self.goal_locations,
                                                                             detail_output=detail_output)
                case 5:
                    pathes = Genetic_algorithm(self.neighboring_state).genetic_algorithm(graph=self.graph,
                                                                                         starts=self.starting_locations,
                                                                                         goals=self.goal_locations,
                                                                                         detail_output=detail_output)

            if type(pathes) == str:
                print(' No path found')
                return
            elif pathes:
                if len(pathes) == len(starting_locations):
                    self.detail_output_false(pathes, search_method, detail_output)
                    return
            print("No path found")

    def upload_data(self):
        with open('adjacency.csv', 'r') as csv_file:
            adjacency = csv.reader(csv_file)
            for row in adjacency:
                self.add_counties_connection(row)
        for state in self.counties:
            self.graph[state] = state.connections

    ################### sub function
    def add_counties_connection(self, connection):
        county1_string_arr = connection[0].split(', ')
        county2_string_arr = connection[1].split(', ')
        county1 = self.get_counties(county1_string_arr[1], county1_string_arr[0])

        if county1 == None:
            county1 = County(county1_string_arr[0], county1_string_arr[1])
            self.counties.append(county1)

        county2 = self.get_counties(county2_string_arr[1], county2_string_arr[0])
        if county2 == None:
            county2 = County(county2_string_arr[0], county2_string_arr[1])
            self.counties.append(county2)

        county1.add_connection(county2)
        county2.add_connection(county1)

        if county1_string_arr[1] not in self.neighboring_state.keys():
            self.neighboring_state[county1_string_arr[1]] = []
        if county2_string_arr[1] not in self.neighboring_state.keys():
            self.neighboring_state[county2_string_arr[1]] = []
        if county2_string_arr[1] != county1_string_arr[1]:
            if county1_string_arr[1] not in self.neighboring_state[county2_string_arr[1]]:
                self.neighboring_state[county2_string_arr[1]].append(county1_string_arr[1])
            if county2_string_arr[1] not in self.neighboring_state[county1_string_arr[1]]:
                self.neighboring_state[county1_string_arr[1]].append(county2_string_arr[1])

    def get_counties(self, us_state_code, name: None):
        if name == None:
            counties = []
            for county in self.counties:
                if county.us_state_code == us_state_code:
                    counties.append(county)
            return counties
        for county in self.counties:
            if county.name == name and county.us_state_code == us_state_code:
                return county
        return None

    def precheck_input(self, starting_locations, goal_locations):
        self.prepare_the_input(starting_locations, goal_locations)
        if len(self.starting_locations) != len(self.goal_locations):
            print("\033[91mError: uneven starting and goal location\033[0m")
            return False
        precheck_input = True
        if None in self.starting_locations:
            print("\033[91mError: unvalid starting_locations\033[0m")
            precheck_input = False
        if None in self.goal_locations:
            print("\033[91mError: unvalid goal_locations\033[0m")
            precheck_input = False
        if not precheck_input:
            print("\033[91mThe input format is invalid or country not in the list of Countries \033[0m")
            print('Try entering the countries in one of the following formats')
            print(
                "list of strings: ['color{Blue/Red}, country, us state code','color{Blue/Red}, country, us state code'")
            print("string : 'color{Blue/Red}, country, us state code ; color{Blue/Red}, country, us state code ; '")
            print("string : '{color{Blue/Red}, country, us state code ; color{Blue/Red}, country, us state code ; }'")
        return precheck_input

    def prepare_the_input(self, starting_locations, goal_locations):
        if not (isinstance(starting_locations, list) or isinstance(goal_locations, list)):
            if "{" in goal_locations:
                goal_locations = goal_locations[1:-1]
                starting_locations = starting_locations[1:-1]
            goal_locations = goal_locations.split(' ; ')
            starting_locations = starting_locations.split(' ; ')
        for county_string in goal_locations:
            county_list = county_string.split(', ')
            county = self.get_counties(county_list[2], county_list[1])
            county.color = county_list[0]
            self.goal_locations.append(county)
        for county_string in starting_locations:
            county_list = county_string.split(', ')
            county = self.get_counties(county_list[2], county_list[1])
            county.color = county_list[0]
            self.starting_locations.append(county)

    # ############## explore function

    def explore(self):
        # us_state_codes = []
        # state_inter_connection = {}
        # for county in self.counties:
        #     if county.us_state_code not in us_state_codes:
        #         state_inter_connection[county.us_state_code] = []
        #         us_state_codes.append(county.us_state_code)
        #     inter_counties = 0
        #     for connection in county.connections:
        #         if connection.us_state_code == county.us_state_code:
        #             inter_counties+=1
        #     state_inter_connection[county.us_state_code].append(inter_counties)
        #
        # print('us_state_codes ' ,  us_state_codes)
        # print("number of us state : " , len(us_state_codes))
        # for state in state_inter_connection.items():
        #     print(state)
        # for state in self.neighboring_state.items():
        #     print(state)
        pass

    def grouping(self):
        counties = []
        for county in self.counties:
            counties.append(county)
        group = 0
        while counties:
            unvisited = [counties[0]]
            visited = []
            while unvisited:
                county = unvisited.pop(0)
                counties.remove(county)
                county.group = group
                visited.append(county)
                for neighbor in county.connections:
                    if neighbor in counties:
                        neighbor.group = group
                        if neighbor not in unvisited:
                            unvisited.append(neighbor)
            group += 1

    def detail_output_false(self, pathes, search_method, detail_output):
        length = max([len(path) for path in pathes])
        for i in range(length):
            if detail_output and i == 2 and search_method == 1:
                pathstr = "Heuristic: {"
                for path in pathes:

                    if i < len(path):
                        pathstr += str(detail_heuristic[path[0]][path[1]])
                    else:
                        pathstr += str(detail_heuristic[path[0]][path[1]])
                    if path != pathes[-1]:
                        pathstr += ' ; '

                pathstr += "}"
                print(pathstr)
            if detail_output and i == 2 and search_method == 5:
                for goal ,population in detail_heuristic.items():
                    print(f"from : {goal}")
                    for child in population:
                        print("    " , child)
            pathstr = "{"
            for path in pathes:

                if i < len(path):
                    pathstr += str(path[i])
                else:
                    pathstr += str(path[-1])
                pathstr += f"({path[0].color[0]})"
                if path != pathes[-1]:
                    pathstr += ' ; '

            pathstr += "}"
            print(pathstr)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # main()
    for i in [5]:
        find_path(
            ['Blue, Washington County, UT', 'Blue, Chicot County, AR', 'Red, Fairfield County, CT'],
            ["Blue, San Diego County, CA", "Blue, Bienville Parish, LA", "Red, Rensselaer County, NY"],
            i, True)
