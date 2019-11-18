import GPyOpt
import chaospy
import matplotlib
import math
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
np.set_printoptions(linewidth=200, precision=4)


def equation(x, selection_index):
    target_region = {'x': (0, 1), 'y': (0, 1)}

    def function(selection_index, h=1): #1 is just a dummy value
        if selection_index == 1:
            f = math.sin(h) + math.sin(10 * h / 3)
            region_of_interest = {'x': (2.7, 7.5), 'y': (-2, 1)} 
            
        if selection_index == 2:
            f = - (16 * h ** 2 - 24 * h + 5) * math.e ** -h
            region_of_interest = {'x': (1.9, 3.9), 'y': (-4, -2.4)} 
            
        if selection_index == 3:
            f = - (1.4 - 3 * h) * math.sin(18 * h)
            region_of_interest = {'x': (0, 1.2), 'y': (-1.5, 2.5)} 
            
        if selection_index == 4:
            f = - (h + math.sin(h)) * math.e ** - (h ** 2)
            region_of_interest = {'x': (-10, 10), 'y': (-1, 1)} 
            
        if selection_index == 5:
            f = math.sin(h) + math.sin(10 * h / 3) + math.log(h) - 0.84 * h + 3
            region_of_interest = {'x': (2.7, 7.5), 'y': (-2, 3)} 
            
        if selection_index == 6:
            f = - h * math.sin(h)
            region_of_interest = {'x': (0, 10), 'y': (-8, 6)} 
            
        if selection_index == 7:
            f = math.sin(h) ** 3 + math.cos(h) ** 3
            region_of_interest = {'x': (0, 2 * math.pi), 'y': (-1, 1)} 
            
        if selection_index == 8:
            f = - h ** (2 / 3) - (1 - h ** 2) ** (1 / 3)
            region_of_interest = {'x': (0.001, 0.99), 'y': (-1.6, -1)} 
            
        if selection_index == 9:
            f = - (math.e ** (-h)) * math.sin(2 * math.pi * h)
            region_of_interest = {'x': (0, 4), 'y': (-0.8, 0.6)} 
            
        if selection_index == 10:
            f = (h ** 2 - 5 * h + 6) / (h ** 2 + 1)
            region_of_interest = {'x': (-5, 5), 'y': (-1, 8)} 
        
        return f, region_of_interest

    _, region_of_interest = function(selection_index)    
    x_translate = target_region['x'][0] - region_of_interest['x'][0]
    y_translate = target_region['y'][0] - region_of_interest['y'][0]
    x_squeeze = (target_region['x'][1] - target_region['x'][0]) / (region_of_interest['x'][1] - region_of_interest['x'][0])
    y_squeeze = (target_region['y'][1] - target_region['y'][0]) / (region_of_interest['y'][1] - region_of_interest['y'][0])
    h = x / x_squeeze - x_translate 
    j, _ = function(selection_index, h)
    y = (j + y_translate) * y_squeeze

    return y
    
def plot_evaluated_points(X, Y, X_design, Y_design, x_minimum=0, y_minimum=0):
    title = 'Evaluations for Mixed-variable Balance Case'

    num_discrete = 10

    label_color = 'midnightblue'
    fig_mixed = matplotlib.pyplot.figure(figsize=(10, 5))
    ax_mixed = fig_mixed.add_subplot(1, 1, 1)
    ax_mixed.set_title(title, fontweight = 550, fontsize = 'large')
    resolution = 100
    xyz = np.ones((resolution * num_discrete, 3))

    for index in range(num_discrete):
        start = index * resolution
        end = (index + 1) * resolution
        xyz[start:end, 0] = np.linspace(0, 1, resolution)
        xyz[start:end, 1] *= index + 1
        xyz[start:end, 2] = np.asarray([equation(x[0], x[1]) for x in xyz[start:end, [0, 1]]]).reshape(resolution)
    #    ax_mixed.plot(xs = xyz[start:end, 0], ys = xyz[start:end, 1], zs = xyz[start:end, 2])
    
    X_surface = xyz[:, 0]
    Y_surface = xyz[:, 1]
    X_surface, Y_surface = np.meshgrid(X_surface, Y_surface)
    XY_ravel = np.append(X_surface.ravel()[:, np.newaxis], Y_surface.ravel()[:, np.newaxis], axis=1)
    Z_surface = np.asarray([equation(x[0], x[1]) for x in XY_ravel]).reshape(X_surface.shape)
    #ax_mixed.plot_surface(X_surface, Y_surface, Z_surface,
    #                      cmap=matplotlib.cm.plasma, linewidth=1)
    contour = ax_mixed.contourf(X_surface, Y_surface, Z_surface, cmap=matplotlib.cm.viridis)
    fig_mixed.colorbar(contour, ax=ax_mixed)

    X_acquisition = np.delete(X, list(range(X_design.shape[0])), axis = 0)
    Y_acquisition = np.delete(Y, list(range(Y_design.shape[0])), axis = 0)
    size = np.linspace(100, 10, X_acquisition.shape[0])
    ax_mixed.scatter(x=X_design[:, 0], y=X_design[:, 1], c='firebrick', marker='o', s=100)
    ax_mixed.scatter(x=X_acquisition[:, 0], y=X_acquisition[:, 1], c='orange', marker=7, s=size)
    ax_mixed.scatter(x = X[np.argmin(Y), 0], y = X[np.argmin(Y), 1], c='crimson', marker = 'x', s=200)
    ax_mixed.scatter(x = x_minimum, y = y_minimum, c='black', marker = '*', s=200)

    ax_mixed.set_xlabel('x-value', color = label_color)
    ax_mixed.set_ylabel('Selection index', color = label_color)
    design = matplotlib.lines.Line2D([], [], color = 'firebrick', linestyle='None', marker = 'o', markersize = 10, label = 'design points')
    acquisition = matplotlib.lines.Line2D([], [], color = 'orange', linestyle='None', marker = 7, markersize = 10, label = 'acquisitions')
    located_optimum = matplotlib.lines.Line2D([], [], color = 'crimson', linestyle='None', marker = 'x', markersize = 10, label = 'located optimum')
    actual_optimum = matplotlib.lines.Line2D([], [], color = 'black', linestyle='None', marker = '*', markersize = 10, label = 'actual optimum')
    ax_mixed.legend(handles = [design, acquisition, located_optimum, actual_optimum], loc = 'best', shadow = True)

    fig_mixed.tight_layout(pad=0.35, w_pad=0.5, h_pad=2.5)
    return None

def compare_with_actual(problem, variables):
    continuous_bounds = variables[0]['domain']
    discrete_levels = variables[1]['domain']
    fig = matplotlib.pyplot.figure(figsize=(10, 5 * len(discrete_levels)))
    ax = [None for n in range(2*len(discrete_levels))]
    label_color = 'midnightblue'

    plot = 0
    x1_continuous = np.linspace(continuous_bounds[0], continuous_bounds[1], 1000)
    for x2_discrete in discrete_levels:
        Y_actual = []
        Y_metamodel = []
        for x1 in x1_continuous:
            X = np.asarray([x1, x2_discrete])
            mv = problem.model.predict(X)
            Y_a = equation(x1, x2_discrete)
            Y_m = np.asarray(mv).reshape(2)[0]
            Y_actual.append(Y_a)
            Y_metamodel.append(Y_m)
                
        ax[plot] = fig.add_subplot(len(discrete_levels), 2, plot+1)
        title = f'Discrete value #{x2_discrete} (Actual)'
        ax[plot].set_title(title, fontweight = 550, fontsize = 'large')
        ax[plot].plot(x1_continuous, Y_actual, 'b') 
        ax[plot].set_xlabel('x-position', color = label_color)
        ax[plot].set_ylabel('Distance (to minimize)', color = label_color)
        plot += 1

        ax[plot] = fig.add_subplot(len(discrete_levels), 2, plot+1)
        title = f'Discrete value #{x2_discrete} (Predicted)'
        ax[plot].set_title(title, fontweight = 550, fontsize = 'large')
        ax[plot].plot(x1_continuous, Y_metamodel, 'b') 
        ax[plot].set_xlabel('x-position', color = label_color)
        ax[plot].set_ylabel('Distance (to minimize)', color = label_color)
        plot += 1
    
    fig.tight_layout(pad=0.35, w_pad=0.5, h_pad=3.5)
    return None

def plot_convergence(Y_data):
    X = [x for x in range(1, len(Y_data)+1)]
    Y = [y for y in Y_data]
    convergence_fig = matplotlib.pyplot.figure(figsize=(10, 5))
    ax = convergence_fig.add_subplot(1, 1, 1)
    title = 'Convergence Plot'
    ax.set_title(title, fontweight = 550, fontsize = 'large')
    ax.plot(X, Y, 'b', marker='o') 
    ax.set_xlabel('Batch Iteration')
    ax.set_ylabel('Objective Value')
    return None

def generate_experimental_design(num_design):
    print('Generating experimental design...\n')          
    hammerseley = chaospy.distributions.sampler.sequences.hammersley
    base = hammerseley.create_hammersley_samples(num_design, dim=2, burnin=-1, primes=()) #numpy array
    x = (base[0, :] * 1).tolist()
    selection_index = np.rint(base[1, :] * 9 + 1).astype(int).tolist()
    design = np.asarray([[x[design], selection_index[design]] for design in range(num_design)])
    return design

space_mixed_variables = \
    [{'name': 'x', 'type': 'continuous', 'domain':(0,1)},
     {'name': 'selection_index', 'type': 'discrete', 'domain': (1,2,3,4,5,6,7,8,9,10)}]     

#space_mixed = GPyOpt.core.task.space.Design_space(space_mixed_variables)
#experiment_design_mixed_X = GPyOpt.experiment_design.LatinMixedDesign(space_mixed).get_samples(20)
experiment_design_mixed_X = generate_experimental_design(200)
experiment_design_mixed_Y = []
for x, selection_index in experiment_design_mixed_X:
    Y = equation(x, selection_index)
    experiment_design_mixed_Y.append([Y])
experiment_design_mixed_Y = np.asarray(experiment_design_mixed_Y)
#plot_experiment_design_mixed(experiment_design_mixed_X)
X_values_mixed = experiment_design_mixed_X
Y_values_mixed = experiment_design_mixed_Y
numIterations_mixed = 1

X_initial_values_mixed = X_values_mixed
Y_initial_values_mixed = Y_values_mixed
X_initial_best = X_values_mixed[np.argmin(Y_values_mixed)]
Y_initial_best = Y_values_mixed[np.argmin(Y_values_mixed)]
best_x = []
best_fx = []

for step in range(numIterations_mixed):
    mixed_problem = GPyOpt.methods.BayesianOptimization(
        f = None, 
        domain = space_mixed_variables,
        constraints = None,
        cost_withGradients = None,
        model_type = 'GP',
        X = X_values_mixed,
        Y = Y_values_mixed,
        acquisition_type = 'EI',
        normalize_Y = True,
        exact_feval = False,
        acquisition_optimizer_type = 'lbfgs',
        evaluator_type = 'local_penalization',
        batch_size = 1,
        maximize = False,
        de_duplication = True,
        Gower = True,
        noise_var = 0)
    x_next_mixed = mixed_problem.suggest_next_locations()
    y_next_mixed = []
    for x, selection_index in x_next_mixed:
        Y = equation(x, selection_index)
        y_next_mixed.append([Y])
    y_next_mixed = np.asarray(y_next_mixed)
    X_values_mixed = np.vstack((X_values_mixed, x_next_mixed))
    Y_values_mixed = np.vstack((Y_values_mixed, y_next_mixed))
    print(f'Iteration {step+1}') # This cannot seem to be printed before the acquisition plot and the last print does not appear
    mixed_problem.plot_acquisition()
    print(f'New location/s: {[tuple(point) for point in x_next_mixed]}\n')
    mixed_problem._compute_results()
    #mixed_problem.plot_convergence()
    best_x.append(mixed_problem.x_opt)
    best_fx.append(mixed_problem.fx_opt)
    

best_x = np.asarray(best_x)
best_fx = np.asarray(best_fx)

plot_evaluated_points(X_values_mixed, Y_values_mixed, X_initial_values_mixed, Y_initial_values_mixed)
compare_with_actual(problem = mixed_problem, variables = space_mixed_variables)

print('X_initial_best', X_initial_best)
print('Y_initial_best', Y_initial_best)

print('Located optimum:', mixed_problem.x_opt)
print('Value:', mixed_problem.fx_opt)

plot_convergence(best_fx)

#These can be used to compare with x_opt and fx_opt to check consistency.
#print('Located optimum:', X_values_mixed[np.argmin(Y_values_mixed)])
#print('Value:', Y_values_mixed[np.argmin(Y_values_mixed)])

#print('Actual optimum:', [1, weights[0].index(min(weights[0]))])
#print('Value:', balance(np.asarray([1, weights[0].index(min(weights[0]))]).reshape(1, 2), weights))

mixed_problem.plot_convergence()

# endregion