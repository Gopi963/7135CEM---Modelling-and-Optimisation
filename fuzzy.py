import fuzzylite as fl
import matplotlib.pyplot as plt

# Initialize the fuzzy logic engine
engine = fl.Engine(name="AssistiveCareFLC")

# Define the input variable for temperature
temperature = fl.InputVariable(name="Temperature", minimum=0.0, maximum=40.0)
temperature.terms.append(fl.Trapezoid("Cold", 0.0, 0.0, 10.0, 15.0))
temperature.terms.append(fl.Triangle("Comfortable", 15.0, 20.0, 25.0))
temperature.terms.append(fl.Trapezoid("Hot", 25.0, 30.0, 40.0, 40.0))
engine.input_variables.append(temperature)

# Define the input variable for light level
light_level = fl.InputVariable(name="Light_Level", minimum=0, maximum=1000)
light_level.terms.append(fl.Trapezoid("Dark", 0, 0, 100, 300))
light_level.terms.append(fl.Triangle("Medium", 200, 500, 800))
light_level.terms.append(fl.Trapezoid("Bright", 600, 900, 1000, 1000))
engine.input_variables.append(light_level)

# Define the output variable for heater
heater = fl.OutputVariable(name="Heater", minimum=0.0, maximum=100.0)
heater.defuzzifier = fl.Centroid(100)
heater.aggregation = fl.Maximum()
heater.terms.append(fl.Triangle("Off", 0.0, 0.0, 25.0))
heater.terms.append(fl.Triangle("Low", 15.0, 50.0, 85.0))
heater.terms.append(fl.Triangle("High", 75.0, 100.0, 100.0))
engine.output_variables.append(heater)

# Define the output variable for lights
lights = fl.OutputVariable(name="Lights", minimum=0.0, maximum=100.0)
lights.defuzzifier = fl.Centroid(100)
lights.aggregation = fl.Maximum()
lights.terms.append(fl.Triangle("Dim", 0.0, 0.0, 25.0))
lights.terms.append(fl.Triangle("Moderate", 15.0, 50.0, 85.0))
lights.terms.append(fl.Triangle("Bright", 75.0, 100.0, 100.0))
engine.output_variables.append(lights)

# Define rule blocks and add rules
rule_block = fl.RuleBlock()
rule_block.conjunction = fl.Minimum()
rule_block.disjunction = fl.Maximum()
rule_block.implication = fl.Minimum()
rule_block.activation = fl.General()
rule_block.rules.append(fl.Rule.create("if Temperature is Cold then Heater is High", engine))
rule_block.rules.append(fl.Rule.create("if Temperature is Comfortable then Heater is Off", engine))
rule_block.rules.append(fl.Rule.create("if Temperature is Hot then Heater is Off", engine))
rule_block.rules.append(fl.Rule.create("if Light_Level is Dark then Lights is Bright", engine))
rule_block.rules.append(fl.Rule.create("if Light_Level is Medium then Lights is Moderate", engine))
rule_block.rules.append(fl.Rule.create("if Light_Level is Bright then Lights is Dim", engine))
engine.rule_blocks.append(rule_block)

# Function to process the FLC and print results
def process_and_print_results(temp_value, light_value):
    temperature.value = temp_value
    light_level.value = light_value
    engine.process()
    print(f"Temperature: {temp_value}, Light Level: {light_value}")
    print(f"Heater Output Level: {heater.value}")
    print(f"Lights Output Level: {lights.value}")

# Example simulation
temperature_value = 18  # Example temperature
light_level_value = 150  # Example light level
process_and_print_results(temperature_value, light_level_value)

# Visualization (optional)
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

plot_fuzzy_sets(temperature, "Temperature Fuzzy Sets")
plot_fuzzy_sets(light_level, "Light Level Fuzzy Sets")
plot_fuzzy_sets(heater, "Heater Output Fuzzy Sets")
plot_fuzzy_sets(lights, "Lights Output Fuzzy Sets")

# Further testing with different inputs
test_cases = [
    (10, 100),  # Cold and Dark
    (20, 600),  # Comfortable and Medium
    (30, 900)   # Hot and Bright
]

for temp, light in test_cases:
    process_and_print_results(temp, light)
