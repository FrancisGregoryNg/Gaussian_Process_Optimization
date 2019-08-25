# region [Imports]
import GPyOpt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
np.set_printoptions(linewidth=200, precision=4)
# endregion

# region [Function Definitions]

# =============================================================================
# Function Definitions
# =============================================================================

def balance(x_array, weights):
    '''
    Input:
        x_array is a 2D numpy array containing the columns:.
            position_L ---> leftward distance of item_L from the fulcrum,
                            would have to be adjusted given a transformation            
            item_L ---> index, chosen item from the 'left' lists
            item_R ---> index, chosen item from the 'right' lists

    Output:
        y_array is a 2D numpy array containing the column:
            position_R ---> rightward distance of item_R from the fulcrum
    '''
    position_L = x_array[:, [0]]
    item_L = x_array[:, [1]]
    #item_R = x_array[:, [2]]
    item_R = 1
    samples = x_array.shape[0]
    weights_L, weights_R = weights
    weight_L = np.empty((samples, 1))
    #weight_R = np.empty((samples, 1)).
    weight_R = weights_R[item_R]

    for sample in range(samples):
        weight_L[sample, 0] = weights_L[int(item_L[sample, 0])]
    #    weight_R[sample, 0] = weights_R[int(item_R[sample, 0])]
        
    position_L = transform(position_L)
    position_R = weight_L * position_L / weight_R
    y_array = position_R
    return y_array

def balance_simple(x_array, weights):
    '''
    Input:
        x_array is a 2D numpy array containing the column:
            position_L ---> leftward distance of item_L from the fulcrum,
                            would have to be adjusted given a transformation
        
    Output:
        y_array is a 2D numpy array containing the column:
            position_R ---> rightward distance of item_R from the fulcrum
    '''
    position_L = x_array[:, [0]]
    item_L = 3
    item_R = 1
    weights_L, weights_R = weights
    weight_L = weights_L[item_L]
    weight_R = weights_R[item_R]
    position_L = transform(position_L)
    position_R = weight_L * position_L / weight_R
    y_array = position_R
    return y_array

def get_y(x):
    a = 9
    b = 2.8
    c = 2
    d = 7
    e = 3
    f = 2.5
    y = a * np.sin(b * x) ** c * np.sin(d * x + e) * np.cos(f * x)
    return y

def transform(x):
    y = get_y(x)
    r = np.sqrt(x ** 2 + y ** 2)
    return r

def test_transform(x_from, x_to, sample_points, weights):
    x = np.linspace(x_from, x_to, sample_points).reshape(-1, 1) #-1 infers the size of the new dimension from the size of the input array
    y = get_y(x)
    r = transform(x)
    f = balance_simple(x, weights)
    
    label_color = 'midnightblue'
    fig = matplotlib.pyplot.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Artificially complicated balancing problem (to use for test)', fontweight = 550, fontsize = 'large')
    ax.plot(x, y, 'r', label = 'resulting y-position')
    ax.plot(x, r, 'b', label = 'actual radial distance')
    ax.plot(x, f, 'g', label = 'balance value')
    ax.set_xlabel('x-position', color = label_color)
    ax.set_ylabel('Value', color = label_color)
    ax.legend(loc = 'best', shadow = True)
    fig.tight_layout(pad=0.35, w_pad=0.5, h_pad=2.5)

def construct_dicts(labels, weights):
    indices = range(len(weights))
    index_to_labels = {key:value for (key, value) in zip(indices, labels)}
    labels_to_weights = {key:value for (key, value) in zip(labels, weights)}
    return index_to_labels, labels_to_weights

def plot_experiment_design_simple(x_array):
    position_left = x_array[:, 0]
    samples = x_array.shape[0]
    sample_numbers = np.linspace(1, samples, samples)
    
    label_color = 'midnightblue'
    fig_simple = matplotlib.pyplot.figure(figsize=(5, 5))
    axPL = fig_simple.add_subplot(1, 1, 1)
    axPL.set_title('Latin hypercube design (1 variable)', fontweight = 550, fontsize = 'large')
    axPL.scatter(sample_numbers, position_left)
    axPL.set_xlabel('Sample taken', color = label_color)
    axPL.set_ylabel('Left position', color = label_color)
    
    fig_simple.tight_layout(pad=0.35, w_pad=0.5, h_pad=2.5)
    
def plot_experiment_design_mixed(x_array):
    position_left = x_array[:, 0]
    item_left = x_array[:, 1]
    item_right = x_array[:, 2]
    samples = x_array.shape[0]
    sample_numbers = np.linspace(1, samples, samples)
    
    label_color = 'midnightblue'
    fig_mixed = matplotlib.pyplot.figure(figsize=(15, 5))
    axPL = fig_mixed.add_subplot(1, 3, 1)
    axIL = fig_mixed.add_subplot(1, 3, 2)
    axIR = fig_mixed.add_subplot(1, 3, 3)
    axPL.scatter(sample_numbers, position_left)
    axPL.set_xlabel('Sample taken', color = label_color)
    axPL.set_ylabel('Left position', color = label_color)
    axIL.scatter(sample_numbers, item_left)
    axIL.set_xlabel('Sample taken', color = label_color)
    axIL.set_ylabel('Left item index', color = label_color)
    axIR.scatter(sample_numbers, item_right)
    axIR.set_xlabel('Sample taken', color = label_color)
    axIR.set_ylabel('Right item index', color = label_color)
    fig_mixed.tight_layout(pad=0.35, w_pad=2.5, h_pad=2.5)
    fig_mixed.suptitle('Latin hypercube design (1 continuous, 2 discrete variables): 2D scatter plot', fontweight = 550, fontsize = 'large')
    fig_mixed.subplots_adjust(top=0.9)
    
    figure3D = matplotlib.pyplot.figure(figsize=(15, 5))
    axPLIL = figure3D.add_subplot(1, 3, 1, projection = '3d')
    axPLIR = figure3D.add_subplot(1, 3, 2, projection = '3d')
    axILIR = figure3D.add_subplot(1, 3, 3, projection = '3d')
    axPLIL.scatter(sample_numbers, position_left, item_left)
    axPLIL.set_xlabel('Sample taken', color = label_color)
    axPLIL.set_ylabel('Left position', color = label_color)
    axPLIL.set_zlabel('Left item index', color = label_color)
    axPLIR.scatter(sample_numbers, position_left, item_right)
    axPLIR.set_xlabel('Sample taken', color = label_color)
    axPLIR.set_ylabel('Left position', color = label_color)
    axPLIR.set_zlabel('Right item index', color = label_color)
    axILIR.scatter(sample_numbers, item_left, item_right)
    axILIR.set_xlabel('Sample taken', color = label_color)
    axILIR.set_ylabel('Left item index', color = label_color)
    axILIR.set_zlabel('Right item index', color = label_color)
    figure3D.tight_layout(pad=0.35, w_pad=2.5, h_pad=2.5)
    figure3D.suptitle('Latin hypercube design (1 continous, 2 discrete variables): 3D scatter plot', fontweight = 550, fontsize = 'large')
    fig_mixed.subplots_adjust(top=0.9)
    matplotlib.pyplot.draw()

def plot_evaluated_points(X, Y, X_design, Y_design, weights, type='simple'):

    if type == 'simple':
        title = 'Evaluations for Simple Balance Case'

        x_actual = np.linspace(1, 10, 1000).reshape(-1, 1)
        y_actual = balance_simple(x_actual, weights)
        y_actual_max = y_actual[np.argmax(y_actual)]
        X_acquisition = np.setdiff1d(X, X_design, assume_unique = True)
        Y_acquisition = np.setdiff1d(Y, Y_design, assume_unique = True)

        label_color = 'midnightblue'
        fig_simple = matplotlib.pyplot.figure(figsize=(10, 5))
        ax_simple = fig_simple.add_subplot(1, 1, 1)
        ax_simple.set_title(title, fontweight = 550, fontsize = 'large')
        actual, = ax_simple.plot(x_actual, y_actual, 'r', label = 'actual function')

        for (xs, ys) in zip(X_design, Y_design):
            ax_simple.plot(xs, ys, 'g', marker = 'o', markersize = 10)

        for i, (xs, ys) in enumerate(zip(X_acquisition, Y_acquisition)):
            y_random_offset = (y_actual_max - ys) * (0.1 + 0.9 * (np.random.rand()))
            ax_simple.plot(xs, ys, 'b', marker = 7, markersize = 10 - 9 * i/len(X))
            ax_simple.axvline(x = xs, alpha = 0.1 + 0.9 * (i + 1)/(len(X) + 1))
            ax_simple.annotate('{}'.format(i+1), xy = (xs, ys + y_random_offset), alpha = 0.75)  
            
        ax_simple.set_xlabel('x-position', color = label_color)
        ax_simple.set_ylabel('Distance (to minimize)', color = label_color)
        design = matplotlib.lines.Line2D([], [], color = 'g', marker = 'o', markersize = 10, label = 'design points')
        acquisition = matplotlib.lines.Line2D([], [], color = 'b', marker = 7, markersize = 10, label = 'acquisitions')
        ax_simple.legend(handles = [actual, design, acquisition], loc = 'best', shadow = True)
    
        fig_simple.tight_layout(pad=0.35, w_pad=0.5, h_pad=2.5)
    
    elif type == 'mixed':
        title = 'Evaluations for Mixed-variable Balance Case'

        #num_discrete = len(weights[0])
        #x_actual_continuous = np.linspace(1, 10, 1000).reshape(-1, 1)
        #values_discrete = np.arange(num_discrete)
        #quantity_discrete = np.ones((x_actual.shape[0], num_discrete))
        #x_actual_discrete = quantity_discrete * values_discrete
        #x_actual = np.append(x_actual_continuous, x_actual_discrete, axis=1)

        label_color = 'midnightblue'
        fig_mixed = matplotlib.pyplot.figure(figsize=(10, 5))
        ax_mixed = fig_mixed.add_subplot(1, 1, 1)
        ax_mixed.set_title(title, fontweight = 550, fontsize = 'large')
        resolution = 1000
        xyz = np.ones((resolution * len(weights[0]), 3))

        for index in range(len(weights[0])):
            start = index * resolution
            end = (index + 1) * resolution
            xyz[start:end, 0] = np.linspace(1, 10, resolution)
            xyz[start:end, 1] *= index
            xyz[start:end, 2] = balance(xyz[start:end, [0, 1]], weights).reshape(resolution)
        #    ax_mixed.plot(xs = xyz[start:end, 0], ys = xyz[start:end, 1], zs = xyz[start:end, 2])
        
        X_surface = xyz[:, 0]
        Y_surface = xyz[:, 1]
        X_surface, Y_surface = np.meshgrid(X_surface, Y_surface)
        XY_ravel = np.append(X_surface.ravel()[:, np.newaxis], Y_surface.ravel()[:, np.newaxis], axis=1)
        Z_surface = balance(XY_ravel, weights).reshape(X_surface.shape)
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
        ax_mixed.scatter(x = 1, y = weights[0].index(min(weights[0])), c='black', marker = '*', s=200)

        ax_mixed.set_xlabel('x-position', color = label_color)
        ax_mixed.set_ylabel('Item index', color = label_color)
        #ax_mixed.set_zlabel('Distance (to minimize)', color = label_color)
        design = matplotlib.lines.Line2D([], [], color = 'firebrick', linestyle='None', marker = 'o', markersize = 10, label = 'design points')
        acquisition = matplotlib.lines.Line2D([], [], color = 'orange', linestyle='None', marker = 7, markersize = 10, label = 'acquisitions')
        located_optimum = matplotlib.lines.Line2D([], [], color = 'crimson', linestyle='None', marker = 'x', markersize = 10, label = 'located optimum')
        actual_optimum = matplotlib.lines.Line2D([], [], color = 'black', linestyle='None', marker = '*', markersize = 10, label = 'actual optimum')
        ax_mixed.legend(handles = [design, acquisition, located_optimum, actual_optimum], loc = 'best', shadow = True)
    
        fig_mixed.tight_layout(pad=0.35, w_pad=0.5, h_pad=2.5)
        
        #print(X[:, 0].shape)
        #print(X[:, 1].shape)
        #print(Y[:, 0].shape)
        #fig_tricontour = matplotlib.pyplot.figure(figsize=(10, 5))
        #ax_tricontour = fig_tricontour.add_subplot(1, 1, 1)
        #tricountour = ax_tricontour.tricontourf(X[:, 0], X[:, 1], Y[:, 0], cmap=matplotlib.cm.Wistia)
        #tricountour = ax_tricontour.tricontourf(xyz[:, 0], xyz[:, 1], xyz[:, 2], cmap=matplotlib.cm.Wistia)
        #fig_tricontour.colorbar(tricountour, ax=ax_tricontour)
        #ax_tricontour.plot(X_surface, Y_surface, 'ko', ms=3)
        #ax_tricontour.set_title('Tricontour')

        #fig_tricontour.tight_layout(pad=0.35, w_pad=0.5, h_pad=2.5)

    else:
        print('Type is limited to "simple" and "mixed"')

def compare_with_actual(problem, variables,type):
    if type == 'simple':
        pass
    elif type == 'mixed':
        continuous_bounds = variables[0]['domain']
        discrete_levels = variables[1]['domain']
        fig = matplotlib.pyplot.figure(figsize=(10, 5 * len(discrete_levels)))
        ax = [None for n in range(2*len(discrete_levels))]
        label_color = 'midnightblue'

        plot = 0
        x1_continuous = np.linspace(continuous_bounds[0], continuous_bounds[1], 1000)
        for x2_discrete in discrete_levels:
            Y_actual = balance(np.append(x1_continuous.reshape(-1, 1), x2_discrete * np.ones(x1_continuous.reshape(-1, 1).shape), axis = 1), weights)
            Y_metamodel = []
            for x1 in x1_continuous:
                X = np.asarray([x1, x2_discrete])
                mv = problem.model.predict(X)
                Y = np.asarray(mv).reshape(2)[0]
                Y_metamodel.append(Y)
                    
            ax[plot] = fig.add_subplot(len(discrete_levels), 2, plot+1)
            title = f'Discrete value #{x2_discrete}: {weights[0][x2_discrete]} (Actual)'
            ax[plot].set_title(title, fontweight = 550, fontsize = 'large')
            ax[plot].plot(x1_continuous, Y_actual, 'b') 
            ax[plot].set_xlabel('x-position', color = label_color)
            ax[plot].set_ylabel('Distance (to minimize)', color = label_color)
            plot += 1

            ax[plot] = fig.add_subplot(len(discrete_levels), 2, plot+1)
            title = f'Discrete value #{x2_discrete}: {weights[0][x2_discrete]} (Predicted)'
            ax[plot].set_title(title, fontweight = 550, fontsize = 'large')
            ax[plot].plot(x1_continuous, Y_metamodel, 'b') 
            ax[plot].set_xlabel('x-position', color = label_color)
            ax[plot].set_ylabel('Distance (to minimize)', color = label_color)
            plot += 1
        
        fig.tight_layout(pad=0.35, w_pad=0.5, h_pad=3.5)
    else:
        print('"Type" must only be either "simple" or "mixed".')

# endregion

# region [Problem Definition]

# =============================================================================
# Problem Definition
# =============================================================================
'''
    Scenario:
        There are two lists of alternative items with varied weights. These are
        the choices for the item placement on the left and the right side of 
        the fulcrum in the balancing lever. 
        
        The item at the left can be attached to a movable massless platform
        which means that the distance from the fulcrum can vary. To add
        complexity, this half of the lever is a curve described by a certain
        function which would essentially necessitate a transformation.
        
        Meanwhile, the item at the right would then to be placed a certain 
        distance to maintain a balance. The length of the right half of the 
        lever is magical and will somehow neutralize the effect of the left
        half's weight on the balance.
        
        Because 'magic' is costly, the objective would be to minimize the
        required length of the right half. Intuitively, the correct solution 
        would be to use the lightest item at the left side, set to the closest
        possible distance to the fulcrum, and then use the heaviest item at the
        right side.
        
        For the simple case, the items are fixed. Thus, the problem is 
        effectively reduced to only locating the minima of the distance of the 
        right half.
'''

labels_L = ('rock', 'paper', 'scissors', 'cardboard', 'water', 'hamster')
labels_R = ('gold', 'plastic', 'wood', 'bag', 'stone', 'acrylic')

weights_L = (100, 55, 20, 88, 75, 42)
weights_R = (85, 16, 74, 26, 94, 48)
weights = (weights_L, weights_R)

L_index_to_labels, L_labels_to_weights = construct_dicts(labels_L, weights_L)
R_index_to_labels, R_labels_to_weights = construct_dicts(labels_R, weights_R)

# endregion

# region [Test Case - Simple Balance]
'''
# =============================================================================
# GPyOpt: Simple Case (no discrete variables)
# =============================================================================

values = tuple(np.linspace(1, 10, 20))

space_simple_variables = \
    [{'name': 'position_L', 'type': 'continuous', 'domain':(1,10)}]


#space_simple_variables = \
#    [{'name': 'position_L', 'type': 'discrete', 'domain':values}]

space_simple = GPyOpt.core.task.space.Design_space(space_simple_variables)
experiment_design_simple_X = GPyOpt.experiment_design.LatinDesign(space_simple).get_samples(10)
experiment_design_simple_Y = balance_simple(experiment_design_simple_X, weights)
#plot_experiment_design_simple(experiment_design_simple_X)
X_values_simple = experiment_design_simple_X
Y_values_simple = experiment_design_simple_Y

numIterations_simple = 15
X_initial_values_simple = X_values_simple
Y_initial_values_simple = Y_values_simple
X_initial_best = X_values_simple[np.argmin(Y_values_simple)]
Y_initial_best = Y_values_simple[np.argmin(Y_values_simple)]
best_x = []
best_fx = []
for step in range(numIterations_simple):
    simple_problem = GPyOpt.methods.BayesianOptimization(
        f = None, 
        domain = space_simple_variables,
        constraints = None,
        cost_withGradients = None,
        model_type = 'GP',
        X = X_values_simple,
        Y = Y_values_simple,
        acquisition_type = 'EI',
        normalize_Y = True,
        exact_feval = True,
        acquisition_optimizer_type = 'lbfgs',
        evaluator_type = 'local_penalization',
        batch_size = 5,
        maximize = False,
        de_duplication = True,
        mixed_variables = False)
    x_next_simple = simple_problem.suggest_next_locations(ignored_X = X_values_simple)
    y_next_simple = balance_simple(x_next_simple, weights)
    X_values_simple = np.vstack((X_values_simple, x_next_simple))
    Y_values_simple = np.vstack((Y_values_simple, y_next_simple))
    print('Iteration {}'.format(step+1)) # This cannot seem to be printed before the acquisition plot and the last print does not appear
    simple_problem.plot_acquisition()
    simple_problem._compute_results()
    #simple_problem.plot_convergence()
    best_x.append(simple_problem.x_opt)
    best_fx.append(simple_problem.fx_opt)

best_x = np.asarray(best_x)
best_fx = np.asarray(best_fx)

plot_evaluated_points(X_values_simple, Y_values_simple, X_initial_values_simple, Y_initial_values_simple, weights, type='simple')

print('Located optimum:', simple_problem.x_opt)
print('Value:', simple_problem.fx_opt)

print('X_initial_best', X_initial_best)
print('Y_initial_best', Y_initial_best)

simple_problem.plot_convergence()
'''
# endregion

# region [Test Case - Balance with Mixed Variables]

# =============================================================================
# GPyOpt: Mixed Case (with discrete variables)
# =============================================================================

space_mixed_variables = \
    [{'name': 'position_L', 'type': 'continuous', 'domain':(1,10)},
     {'name': 'item_L', 'type': 'discrete', 'domain': tuple(range(len(labels_L)))}]     

space_mixed = GPyOpt.core.task.space.Design_space(space_mixed_variables)
experiment_design_mixed_X = GPyOpt.experiment_design.LatinMixedDesign(space_mixed).get_samples(20)
experiment_design_mixed_Y = balance(experiment_design_mixed_X, weights)
#plot_experiment_design_mixed(experiment_design_mixed_X)
X_values_mixed = experiment_design_mixed_X
Y_values_mixed = experiment_design_mixed_Y
numIterations_mixed = 20

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
        batch_size = 5,
        maximize = False,
        de_duplication = True,
        Gower = False,
        noise_var = 0)
    x_next_mixed = mixed_problem.suggest_next_locations(ignored_X = X_values_mixed)
    y_next_mixed = balance(x_next_mixed, weights)
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

plot_evaluated_points(X_values_mixed, Y_values_mixed, X_initial_values_mixed, Y_initial_values_mixed, weights, type='mixed')
compare_with_actual(problem = mixed_problem, variables = space_mixed_variables, type = 'mixed')

print('X_initial_best', X_initial_best)
print('Y_initial_best', Y_initial_best)

print('Located optimum:', mixed_problem.x_opt)
print('Value:', mixed_problem.fx_opt)

#These can be used to compare with x_opt and fx_opt to check consistency.
#print('Located optimum:', X_values_mixed[np.argmin(Y_values_mixed)])
#print('Value:', Y_values_mixed[np.argmin(Y_values_mixed)])

print('Actual optimum:', [1, weights[0].index(min(weights[0]))])
print('Value:', balance(np.asarray([1, weights[0].index(min(weights[0]))]).reshape(1, 2), weights))

mixed_problem.plot_convergence()

# endregion

# region [other stuff]


# test_transform(0, 10, 1000, weights)

# add comparision of fitting using sequential sampling for Gower Kriging compared to only LHD 


#max_iter = 200
#
#myProblem = GPyOpt.methods.BayesianOptimization(**simple)
#myProblem.run_optimization(max_iter)
#myProblem.plot_acquisition()
#
#print(myProblem.x_opt)
#print(myProblem.fx_opt)

# =============================================================================
# qwer = np.asarray([[6.657, 2, 1]])
# asdf = balance(qwer, weights)
# print(qwer)
# print(asdf)
# =============================================================================

# endregion