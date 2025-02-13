from threading import Thread
import numpy as np
import pandas as pd
import time

class ACO:
    class Ant(Thread):
        def __init__(self,
                     init_location,
                     possible_locations,
                     pheromone_map,
                     distance_callback,
                     alpha,
                     beta,
                     first_pass=False):
            Thread.__init__(self)

            self.init_location = init_location
            self.possible_locations = possible_locations
            self.route = []
            self.distance_traveled = 0.0
            self.location = init_location
            self.pheromone_map = pheromone_map
            self.distance_callback = distance_callback
            self.alpha = alpha
            self.beta = beta
            self.first_pass = first_pass
            self.update_route(init_location)

            self.tour_complete = False

        def run(self):
            while self.possible_locations:
                nxt = self.pick_path()
                self.traverse(self.location, nxt)
            self.possible_locations.append(self.init_location)
            self.traverse(self.location, self.init_location)
            self.tour_complete = True

        def pick_path(self):
            if self.first_pass:
                import random
                rnd = random.choice(self.possible_locations)
                while rnd == self.init_location and len(self.possible_locations) > 1:
                    rnd = random.choice(self.possible_locations)
                return rnd

            attractiveness = dict()
            sum_total = 0.0
            for possible_next_location in self.possible_locations:
                pheromone_amount = float(self.pheromone_map[self.location][possible_next_location])
                distance = float(self.distance_callback(self.location, possible_next_location))
                attractiveness[possible_next_location] = pow(pheromone_amount, self.alpha) * pow(1 / distance,
                                                                                                     self.beta)
                sum_total += attractiveness[possible_next_location]
            if sum_total == 0.0:
                def next_up(x):
                    import math
                    import struct
                    if math.isnan(x) or (math.isinf(x) and x > 0):
                        return x
                    if x == 0.0:
                        x = 0.0
                    n = struct.unpack('<q', struct.pack('<d', x))[0]
                    if n >= 0:
                        n += 1
                    else:
                        n -= 1
                    return struct.unpack('<d', struct.pack('<q', n))[0]

                for key in attractiveness:
                    attractiveness[key] = next_up(attractiveness[key])
                sum_total = next_up(sum_total)
            import random
            toss = random.random()

            cummulative = 0
            for possible_next_location in attractiveness:
                weight = (attractiveness[possible_next_location] / sum_total)
                if toss <= weight + cummulative:
                    return possible_next_location
                cummulative += weight

        def traverse(self, start, end):
            self.update_route(end)
            self.update_distance_traveled(start, end)
            self.location = end

        def update_route(self, new):
            self.route.append(new)
            self.possible_locations = list(self.possible_locations)
            self.possible_locations.remove(new)

        def update_distance_traveled(self, start, end):
            self.distance_traveled += float(self.distance_callback(start, end))

        def get_route(self):
            if self.tour_complete:
                return self.route
            return None

        def get_distance_traveled(self):
            if self.tour_complete:
                return self.distance_traveled
            return None

        def opposite_location(self, loc, n):
            M = (1 + n) // 2
            if n % 2 != 0:
                return M if loc == M else M + (M - loc) if loc < M else M - (loc - M)
            else:
                return M if loc == M or loc == M + 1 else M + (M - loc) if loc < M else M - (loc - M) if loc > M else M + 1

    def __init__(self,
                 nodes_num,
                 distance_matrix,
                 start,
                 ant_count,
                 alpha,
                 beta,
                 pheromone_evaporation_coefficient,
                 pheromone_constant,
                 iterations):
        self.nodes = list(range(nodes_num))
        self.nodes_num = nodes_num
        self.distance_matrix = distance_matrix
        self.pheromone_map = self.init_matrix(nodes_num)
        self.ant_updated_pheromone_map = self.init_matrix(nodes_num)
        self.start = 0 if start is None else start
        self.ant_count = ant_count
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.pheromone_evaporation_coefficient = float(pheromone_evaporation_coefficient)
        self.pheromone_constant = float(pheromone_constant)
        self.iterations = iterations
        self.first_pass = True
        self.ants = self._init_ants(self.start)
        self.shortest_distance = None
        self.shortest_path_seen = None

    def get_distance(self, start, end):
        size = len(self.distance_matrix)
        return self.distance_matrix[start % size][end % size]

    def init_matrix(self, size, value=0.0):
        ret = []
        for row in range(size):
            ret.append([float(value) for _ in range(size)])
        return ret

    def _init_ants(self, start):
        if self.first_pass:
            return [self.Ant(start, self.nodes, self.pheromone_map, self.get_distance,
                             self.alpha, self.beta, first_pass=True) for _ in range(self.ant_count)]
        for ant in self.ants:
            ant.__init__(start, self.nodes, self.pheromone_map, self.get_distance, self.alpha, self.beta)

    def update_pheromone_map(self):
        for start in range(len(self.pheromone_map)):
            for end in range(len(self.pheromone_map)):
                self.pheromone_map[start][end] *= (1 - self.pheromone_evaporation_coefficient)
                self.pheromone_map[start][end] += self.ant_updated_pheromone_map[start][end]

    def populate_ant_updated_pheromone_map(self, ant):
        route = ant.get_route()
        for i in range(len(route) - 1):
            current_pheromone_value = float(self.ant_updated_pheromone_map[route[i]][route[i + 1]])
            new_pheromone_value = self.pheromone_constant / ant.get_distance_traveled()

            self.ant_updated_pheromone_map[route[i]][route[i + 1]] = current_pheromone_value + new_pheromone_value
            self.ant_updated_pheromone_map[route[i + 1]][route[i]] = current_pheromone_value + new_pheromone_value

    def mainloop(self):
        iterations_with_obl = self.iterations // 2  # Number of iterations with OBL
        for it in range(self.iterations):
            for ant in self.ants:
                ant.start()
            for ant in self.ants:
                ant.join()

            for ant in self.ants:
                self.populate_ant_updated_pheromone_map(ant)
                if not self.shortest_distance:
                    self.shortest_distance = ant.get_distance_traveled()

                if not self.shortest_path_seen:
                    self.shortest_path_seen = ant.get_route()

                if ant.get_distance_traveled() < self.shortest_distance:
                    self.shortest_distance = ant.get_distance_traveled()
                    self.shortest_path_seen = ant.get_route()

            if it < iterations_with_obl:
                for ant in self.ants:
                    # Apply Opposition-Based Learning
                    opposite_route = [ant.opposite_location(loc, self.nodes_num) for loc in ant.route]
                    opposite_distance = sum(self.get_distance(opposite_route[i], opposite_route[i + 1]) for i in
                                            range(len(opposite_route) - 1))
                    if opposite_distance < self.shortest_distance:
                        self.shortest_distance = opposite_distance
                        self.shortest_path_seen = opposite_route

            self.update_pheromone_map()

            if self.first_pass:
                self.first_pass = False

            self._init_ants(self.start)

            self.ant_updated_pheromone_map = self.init_matrix(self.nodes_num, value=0)

        ret = []
        for ids in self.shortest_path_seen:
            ret.append(self.nodes[ids])

        return ret


def main():
    ANT_COUNT = 70
    ALPHA = 0.5
    PHER_EVAP_COEFF = 0.1
    BETAS = [5]
    ITERATIONS_LIST = [2]

    distance_matrix = np.loadtxt(open('dataset.txt', 'rb'), delimiter=' ')

    # Define the clusters (indices of nodes belonging to each cluster)
    clusters = [
        [0, 2, 13, 9, 14, 10, 11, 19],  # Cluster 1
        [0, 32, 34, 40, 60, 63, 67, 68, 69], # Cluster 2
        [0, 20, 29, 37, 41, 44, 45, 46, 47, 49], # Cluster 3
        [0, 16, 21, 22, 23, 24, 25, 26, 28, 30], # Cluster 4
        [0, 1, 52,53,54,55,56,57,58,59,65], # Cluster 5
        [0, 31,33,35,36,39,42,43,48,50,51], # CLuster 6
        [0,3,4,5,6,7,8,12,61], # Cluster 7
        [0, 15, 17,18,27,38, 62,64,66,70], #CLuster 8
    ]

    for beta in BETAS:
        for iterations in ITERATIONS_LIST:
            print(f"BETA: {beta}, ITERATIONS: {iterations}")
            total_shortest_distance_no_obl = 0
            total_shortest_distance_with_obl = 0
            total_aco_runtime = 0
            total_obl_runtime = 0

            for i, cluster in enumerate(clusters):
                selected_distance_matrix = distance_matrix[cluster][:, cluster]
                START = cluster.index(0)

                colony_no_obl = ACO(len(cluster),
                                    selected_distance_matrix,
                                    START,
                                    ANT_COUNT,
                                    ALPHA,
                                    beta,
                                    PHER_EVAP_COEFF,
                                    3000.0,
                                    iterations)
                start_time_aco = time.time()  # Record start time for ACO
                answer_no_obl = colony_no_obl.mainloop()
                end_time_aco = time.time()  # Record end time for ACO
                runtime_aco = end_time_aco - start_time_aco
                total_aco_runtime += runtime_aco

                colony_with_obl = ACO(len(cluster),
                                      selected_distance_matrix,
                                      START,
                                      ANT_COUNT,
                                      ALPHA,
                                      beta,
                                      PHER_EVAP_COEFF,
                                      3000.0,
                                      iterations)
                start_time_obl = time.time()  # Record start time for OBL ACO
                answer_with_obl = colony_with_obl.mainloop()
                end_time_obl = time.time()  # Record end time for OBL ACO
                runtime_obl = end_time_obl - start_time_obl
                total_obl_runtime += runtime_obl

                shortest_distance_no_obl = colony_no_obl.shortest_distance
                shortest_distance_with_obl = colony_with_obl.shortest_distance

                final_answer_no_obl = [cluster[index] for index in answer_no_obl]
                final_answer_with_obl = [cluster[index] for index in answer_with_obl]

                print(f"Cluster {i + 1}:")
                print("Without OBL:")
                print("Route:", final_answer_no_obl)
                print("Shortest distance:", shortest_distance_no_obl)
                print(f"Runtime ACO with {iterations} iteration:", runtime_aco, "seconds")
                print()
                print("With OBL:")
                print("Route:", final_answer_with_obl)
                print("Shortest distance:", shortest_distance_with_obl)
                print(f"Runtime OBL with {iterations} iteration:", runtime_obl, "seconds")
                print()

                total_shortest_distance_no_obl += shortest_distance_no_obl
                total_shortest_distance_with_obl += shortest_distance_with_obl

            print("Total Shortest Distance without OBL for all clusters:", total_shortest_distance_no_obl)
            print("Total Shortest Distance with OBL for all clusters:", total_shortest_distance_with_obl)
            print(f"Total Runtime ACO with {iterations} iteration:", total_aco_runtime, "seconds")
            print(f"Total Runtime OBL with {iterations} iteration:", total_obl_runtime, "seconds")
            print("=" * 100)


if __name__ == '__main__':
    main()
