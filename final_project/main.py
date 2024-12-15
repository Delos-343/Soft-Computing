import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def load_data():
    # Load gas station names and distance matrix
    gas_stations = pd.read_csv('dataset/gas_stations.csv')
    distances = pd.read_csv('dataset/x-y_distances.csv', index_col=0)
    return gas_stations, distances

def calculate_total_distance(route, distance_matrix):
    """Calculate total distance of the given route."""
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix.loc[route[i], route[i + 1]]
    # Return to the starting point
    total_distance += distance_matrix.loc[route[-1], route[0]]
    return total_distance

def generate_neighbor(route):
    """Generate a neighboring solution by swapping two random nodes."""
    new_route = route.copy()
    i, j = random.sample(range(len(route)), 2)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

def simulated_annealing(distance_matrix, start_point, initial_temp, cooling_rate, max_iter):
    """Perform Simulated Annealing to find the shortest route."""
    # Initialize variables
    nodes = list(distance_matrix.columns)
    nodes.remove(start_point)
    current_route = [start_point] + random.sample(nodes, len(nodes))
    best_route = current_route
    current_distance = calculate_total_distance(current_route, distance_matrix)
    best_distance = current_distance
    temperature = initial_temp

    # SA main loop
    while temperature > 1:
        for _ in range(max_iter):
            # Generate a neighbor and calculate its distance
            new_route = generate_neighbor(current_route)
            new_distance = calculate_total_distance(new_route, distance_matrix)

            # Acceptance probability
            if new_distance < current_distance or random.random() < np.exp((current_distance - new_distance) / temperature):
                current_route = new_route
                current_distance = new_distance

                # Update best solution
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance

        # Cool down
        temperature *= cooling_rate

    return best_route, best_distance

def plot_route(route, gas_stations):
    """Plot the route on a graph."""
    coordinates = gas_stations.set_index('Station').loc[route]
    plt.figure(figsize=(10, 6))
    plt.plot(coordinates['Longitude'], coordinates['Latitude'], marker='o', markersize=5, linestyle='-', color='blue')
    for i, station in enumerate(route):
        plt.text(coordinates['Longitude'][i], coordinates['Latitude'][i], station, fontsize=9)
    plt.title('Optimized Route')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid()
    plt.show()

# Main execution
def main():
    # Load data
    gas_stations, distance_matrix = load_data()

    # Parameters
    start_point = 'DISPERINDAG Surabaya'
    initial_temp = 1000
    cooling_rate = 0.995
    max_iter = 100

    # Run Simulated Annealing
    best_route, best_distance = simulated_annealing(distance_matrix, start_point, initial_temp, cooling_rate, max_iter)

    # Output results
    print("Best Route:", best_route)
    print("Shortest Distance:", best_distance)

    # Visualize the route
    plot_route(best_route, gas_stations)

if __name__ == "__main__":
    main()
