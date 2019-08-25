# region [Imports]
import xlrd
import xlwt
import xlutils.copy
import GPyOpt
import matplotlib
import numpy as np
np.set_printoptions(linewidth=200, precision=4)
# endregion

# region [Definitions]
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

class SLT_Optimization():
    def __init__(self, plane_index=0, beam_length_range=(None, None), beam_clearance_range=(None, None)):
        '''
        X = input variables 
                0 - 'wing_connect_L'        - connection point from fuselage
                1 - 'wing_connect_W'        - connection point from trailing edge of wing
                2 - 'beam_length'           - length of beam
                3 - 'beam_clearance'        - distance from beam endpoints
                4 - 'propeller'             - propeller selection
                5 - 'motor'                 - motor selection
                6 - 'battery'               - battery seletion
        D = drag coefficients
                  - data from simulations   - intermediate between X and Y
        Y = output variables
                  - endurance estimate      - calculated value
        '''
        # -----Experimental Design---------------------------
        self.X_design = None
        self.D_design = None
        self.Y_design = None

        # -----Best from Experimental Design-----------------
        self.X_best_design = None
        self.D_best_design = None
        self.Y_best_design = None

        # -----Current Iteration-----------------------------
        self.X = None
        self.D = None
        self.Y = None

        # -----Best up to the Current Iteration--------------
        self.X_best = None
        self.D_best = None
        self.Y_best = None

        # -----History of Data-------------------------------
        self.X_history = None
        self.D_history = None
        self.Y_history = None

        # -----Historical Progression of Best Data-----------
        self.X_best_history = None
        self.D_best_history = None
        self.Y_best_history = None

        # -----Others----------------------------------------
        self.iteration = 0
        self.requested_points = 0
        self.plane = plane_index    # choice of plane is not an optimization variable (default is 0)
        self.plane_data, self.beam_data, self.battery_data, self.motor_data, self.propeller_data = self.load_component_info()
        self.variables = self.define_variables(plane_index, beam_length_range, beam_clearance_range)
        self.space = GPyOpt.core.task.space.Design_space(self.variables)
        return None
    
    def load_component_info(self):
        def _load_component(sheet_name):
            component = []
            with xlrd.open_workbook('components.xlsx', on_demand=True) as book:
                sheet = book.sheet_by_name(sheet_name)
                labels = sheet.row_values(0)
                for rowx in range(1, sheet.nrows):
                    data = sheet.row_values(rowx)
                    component.append(dict(zip(labels, data)))
            return component
        def _add_value_of_motor_for_propeller(sheet_name):
            with xlrd.open_workbook('components.xlsx', on_demand=True) as book:
                sheet = book.sheet_by_name(sheet_name)
                motors = sheet.col_values(0) #maintain alignment with rowx, get() will just ignore the nonexisting key
                propellers = sheet.row_values(0, start_colx=1)
                for rowx in range(1, sheet.nrows):
                    values = sheet.row_values(rowx, start_colx=1)
                    value_dict = ({propeller: value for propeller, value in zip (propellers, values) 
                                if (value is not None and value != '' and value != 0)})
                    motor_to_update = next((item for item in motor_data if item.get('name') == motors[rowx]), None)
                    if motor_to_update is not None:
                        motor_to_update.update({sheet_name: value_dict}) 
            return None
        plane_data = _load_component('plane')
        beam_data = _load_component('beam')
        battery_data = _load_component('battery')
        motor_data = _load_component('motor')
        propeller_data = _load_component('propeller')
        _add_value_of_motor_for_propeller('thrust')
        _add_value_of_motor_for_propeller('efficiency')
        
        return plane_data, beam_data, battery_data, motor_data, propeller_data

    def define_variables(self, beam_length_range=(None, None), beam_clearance_range=(None, None)):
        plane = self.plane_data[self.plane]
        if beam_length_range == (None, None):
            beam_length_min = 3*plane['wing_W_max']
            beam_length_max = 10*plane['wing_W_max']
            beam_length_range = (beam_length_min, beam_length_max)
        if beam_clearance_range == (None, None):
            beam_clearance_min = 5
            beam_clearance_max = 50
            beam_clearance_range = (beam_clearance_min, beam_clearance_max)
        propellers = tuple(range(len(self.propeller_data)))
        motors = tuple(range(len(self.propeller_data)))
        batteries = tuple(range(len(self.battery_data)))
        variables = \
            [
            {'name': 'wing_connect_L', 'type': 'continuous', 'domain':(plane['wing_L_min'], plane['wing_L_max'])},  # connection point from fuselage
            {'name': 'wing_connect_W', 'type': 'continuous', 'domain':(plane['wing_W_min'], plane['wing_W_max'])},  # connection point from trailing edge of wing
            {'name': 'beam_length', 'type': 'continuous', 'domain':beam_length_range},  # length of beam
            {'name': 'beam_clearance', 'type': 'continuous', 'domain':beam_clearance_range},    # distance from beam endpoints
            {'name': 'propeller', 'type': 'discrete', 'domain': propellers},    # propeller selection
            {'name': 'motor', 'type': 'discrete', 'domain': motors},    # motor selection
            {'name': 'battery', 'type': 'discrete', 'domain':batteries},    # battery seletion
            ]             
        return variables

    def request_actual_results(self, X=None):
        if X is None:
            X = self.X
        file = 'evaluations.xls'
        batch_size = X.shape[0]
        variables = X.shape[1]
        # ----- Write style -----
        style = xlwt.XFStyle()
        style.font = xlwt.Font()
        style.font.name = 'Levenim MT'
        style.font.height = 200 # multiply font size by 20
        style.alignment = xlwt.Alignment()
        style.alignment.horz = xlwt.Alignment.HORZ_CENTER
        style.alignment.vert = xlwt.Alignment.VERT_CENTER
        # ----------------------
        with xlrd.open_workbook(file, formatting_info = True, on_demand=True) as book:
            book_write = xlutils.copy.copy(book)
        sheet = book_write.get_sheet(0)     # write configurations to the sheet named "current"
        for configuration in range(batch_size):
            for variable in range(variables):
                sheet.write(configuration+1, variable, float(X[configuration, variable]), style)
        book_write.save(file)
        input(f'Please open {file}, fill up the output column, then save and close the file.\nPress any key when done.')
        with xlrd.open_workbook(file, formatting_info = True, on_demand=True) as book:
            sheet = book.sheet_by_name('current')
            evaluation = sheet.col_values(variables, start_rowx=1, end_rowx=batch_size+1)
            valid = all(evaluation)
            book_write = xlutils.copy.copy(book)
        if valid:
            sheet = book_write.get_sheet(1)     # copy configurations and evaluations to the sheet named "all"
            for configuration in range(batch_size):
                current_row = self.requested_points+1
                for variable in range(variables):
                    sheet.write(current_row, variable, float(X[configuration, variable]), style)
                sheet.write(current_row, variables, evaluation[configuration], style)
                self.requested_points += 1
            sheet = book_write.get_sheet(0)     # clear the sheet named "current"
            for configuration in range(batch_size):
                for variable in range(variables+1):
                    sheet.write(configuration+1, variable, '', style)            
            book_write.save(file)            
            D = np.asarray(evaluation)
        else:
            print(f'\nInvalid entries. Try again...')
            D = self.request_actual_results(X)
        return D

    def calculate_endurance_estimate(self, X=None, D=None):
        if X is None and D is None:
            X = self.X
            D = self.D
        Y = []
        for configuration in range(X.shape[0]):
            plane = self.plane_data[self.plane]
            propeller = self.propeller_data[X[configuration, 4]]
            motor = self.motor_data[X[configuration, 5]]
            battery = self.battery_data[X[configuration, 6]]
            R = battery['battery_hour_rating']      # battery hour rating is typically equal to 1hr
            n = 1.3     # Peukert exponent is typically equal to 1.3 for LiPo batteries
            eff = motor['efficiency'][propeller['name']]  # motor-propeller efficiency is stored in the motor data and can be accessed using the propeller's name
            V = battery['cell'] * 3.7     # 3.7 volts in each LiPo cell
            C = battery['mah'] / 1000       # convert from mAh to Ah
            p = 0       # get the density of air in kg/m3
            U = 0       # get the flight velocity in m/s
            S = 0       # get the reference area in m2
            C_D0 = 0 * D[configuration] # find a way to get the zero-lift drag-coefficient
            W = (plane['weight'] + propeller['weight'] + motor['weight'] + battery['weight']) * 0.0098      # convert from grams to Newtons
            k = 0       # find a way to get the lift-dependent drag factor
            E = R**(1-n) * ((eff * V * C)/(0.5 * p * U**3 * S * C_D0 + (2 * W**2 * k) / (p * U * S))) ** n
            Y.append(E)
        return Y

    def check_constraints(self):
        # positional_interference
        # electrical compatibility
        # weight restriction
        # center of mass

        return

    def get_best_values(self, X=None, D=None, Y=None):
        if X is None and D is None and Y is None:
            X = self.X_history
            D = self.D_history
            Y = self.Y_history
        index_best = np.argmin(Y)
        X_best = X[index_best]
        D_best = D[index_best]
        Y_best = Y[index_best]
        return X_best, D_best, Y_best

    def run_optimization(self, num_design = 20, num_iteration = 50):
        self.X_design = GPyOpt.experiment_design.LatinMixedDesign(self.space).get_samples(num_design)
        self.D_design = self.request_actual_results(self.X_design)
        self.Y_design = self.calculate_endurance_estimate(self.X_design, self.D_design)
        self.X_best_design, self.D_best_design, self.Y_best_design = self.get_best_values(self.X_design, self.D_design, self.Y_design)
        # get X_best_design and Y_best_design (preferrably define a function to get the best value so it can be recycled for taking the overall best)
        '''
        X_initial_best = X_values_mixed[np.argmin(Y_values_mixed)]
        Y_initial_best = Y_values_mixed[np.argmin(Y_values_mixed)]
        '''
        # 
        '''
        for step in range(numIterations_mixed):
            mixed_problem = GPyOpt.methods.BayesianOptimization(
                f = None, 
                domain = variables,
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
                Gower = True,
                noise_var = 0)
            x_next_mixed = mixed_problem.suggest_next_locations(ignored_X = X_values_mixed)
            y_next_mixed = balance(x_next_mixed, weights)
            X_values_mixed = np.vstack((X_values_mixed, x_next_mixed))
            Y_values_mixed = np.vstack((Y_values_mixed, y_next_mixed))
            print(f'Iteration {step+1}')
            mixed_problem.plot_convergence()
            print(f'New location/s: {[tuple(point) for point in x_next_mixed]}\n')
            mixed_problem._compute_results()
            #mixed_problem.plot_convergence()
            best_x.append(mixed_problem.x_opt)
            best_fx.append(mixed_problem.fx_opt)     
        '''   
        return None

# endregion        

thesis = SLT_Optimization()
thesis.run_optimization(num_design = 20, num_iteration = 50)