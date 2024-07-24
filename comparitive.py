import fuzzylite as fl
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import random
from deap import base, creator, tools, algorithms

# Initialize the fuzzy logic engine
engine = fl.Engine(name="OptimizationFIS")

# Define input variables
input1 = fl.InputVariable(name="Input1", minimum=0.0, maximum=10.0)
input1.terms.append(fl.Triangle("Low", 0.0, 2.5, 5.0))
input1.terms.append(fl.Triangle("Medium", 2.5, 5.0, 7.5))
input1.terms.append(fl.Triangle("High", 5.0, 7.5, 10.0))
engine.input_variables.append(input1)

input2 = fl.InputVariable(name="Input2", minimum=0.0, maximum=10.0)
input2.terms.append(fl.Triangle("Low", 0.0, 2.5, 5.0))
input2.terms.append(fl.Triangle("Medium", 2.5, 5.0, 7.5))
input2.terms.append(fl.Triangle("High", 5.0, 7.5, 10.0))
engine.input_variables.append(input2)

# Define output variable
output = fl.OutputVariable(name="Output", minimum=0.0, maximum=100.0)
output.defuzzifier = fl.Centroid(100)
output.aggregation = fl.Maximum()
output.terms.append(fl.Triangle("Low", 0.0, 25.0, 50.0))
output.terms.append(fl.Triangle("Medium", 25.0, 50.0, 75.0))
output.terms.append(fl.Triangle("High", 50.0, 75.0, 100.0))
engine.output_variables.append(output)

# Define rule blocks
rule_block = fl.RuleBlock()
rule_block.conjunction = fl.Minimum()
rule_block.disjunction = fl.Maximum()
rule_block.implication = fl.Minimum()
rule_block.activation = fl.General()
rule_block.rules.append(fl.Rule.create("if Input1 is Low and Input2 is Low then Output is Low", engine))
rule_block.rules.append(fl.Rule.create("if Input1 is Medium or Input2 is Medium then Output is Medium", engine))
rule_block.rules.append(fl.Rule.create("if Input1 is High and Input2 is High then Output is High", engine))
engine.rule_blocks.append(rule_block)

def fuzzy_objective(x):
    input1.value = x[0]
    input2.value = x[1]
    engine.process()
    output_value = output.value
    # Ensure the function returns a tuple with the fitness value
    return (output_value,)

def process_and_print_results(input1_value, input2_value):
    input1.value = input1_value
    input2.value = input2_value
    engine.process()
    print(f"Input1: {input1_value}, Input2: {input2_value}")
    print(f"Output: {output.value}")

# Example: Testing the FIS
x = [5.0, 7.0]
print(f"Fuzzy Output: {fuzzy_objective(x)[0]}")

# Differential Evolution Optimization
bounds = [(0, 10), (0, 10)]

result_de = differential_evolution(lambda x: fuzzy_objective(x)[0], bounds)
print(f"Optimized Inputs (DE): {result_de.x}")
print(f"Optimized Output (DE): {result_de.fun}")

# Genetic Algorithm Optimization
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Use a map function that extracts the first element of the tuple returned by fuzzy_objective
def evaluate_individual(individual):
    return fuzzy_objective(individual)

toolbox.register("evaluate", evaluate_individual)

population = toolbox.population(n=50)
ngen, cxpb, mutpb = 40, 0.7, 0.2

result_ga = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=False)

best_individual = tools.selBest(population, k=1)[0]
print(f"Optimized Inputs (GA): {best_individual}")
print(f"Optimized Output (GA): {fuzzy_objective(best_individual)[0]}")

# Visualization Functions
def plot_fuzzy_sets(variable, title):
    plt.figure(figsize=(10, 6))
    for term in variable.terms:
        x = list(range(int(variable.minimum), int(variable.maximum) + 1))
        y = [term.membership(v) for v in x]
        plt.plot(x, y, label=term.name)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Membership')
    plt.legend()
    plt.show()

plot_fuzzy_sets(input1, "Input1 Fuzzy Sets")
plot_fuzzy_sets(input2, "Input2 Fuzzy Sets")
plot_fuzzy_sets(output, "Output Fuzzy Sets")

# Visualization of Optimization Results
def plot_optimization_results(result_de, result_ga, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Differential Evolution results
    ax.scatter(result_de.x[0], result_de.x[1], color='blue', label=f'DE: {result_de.fun:.2f}')
    
    # Plot Genetic Algorithm results
    best_individual_values = list(best_individual)
    ax.scatter(best_individual_values[0], best_individual_values[1], color='green', label=f'GA: {fuzzy_objective(best_individual)[0]:.2f}')
    
    ax.set_title(title)
    ax.set_xlabel('Input1')
    ax.set_ylabel('Input2')
    ax.legend()
    plt.show()

plot_optimization_results(result_de, result_ga, "Optimization Results")

# Test Cases and Comparison
test_cases = [
    (1, 2),   # Low and Low
    (5, 6),   # Medium and Medium
    (8, 9)    # High and High
]

for temp, light in test_cases:
    process_and_print_results(temp, light)
