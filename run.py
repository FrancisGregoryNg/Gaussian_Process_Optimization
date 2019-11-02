# region [Imports]
import GPyOpt
import xlwings as xw
import pathlib
import matplotlib
import numpy as np
np.set_printoptions(linewidth=200, precision=4)
# endregion

# region [Definitions]
class SLT_Optimization():
    Datafolder = pathlib.Path(__file__).parents[2].joinpath('2---Data')
    def __init__(self, target_velocity, plane_index=0, beam_index = 0, plane_battery_cell = [3], beam_length_range=(None, None)):
        '''
        X = input variables 
                0 - 'motor'                     - motor selection
                1 - 'propeller'                 - propeller selection
                2 - 'battery'                   - battery selection
                3 - 'distanceFromCenterline'    - connection point from the center of the fuselage
                4 - 'beam_length'               - length of beam
                5 - 'pitch'                     - angle of the drone with the apparent wind
                
        D = drag force
                  - data from simulations       - intermediate between X and Y
        
        L = lift force
                  - data from simulations       - used for checking actual flight capability
                  
        Y = output variables
                  - endurance estimate          - calculated value
        '''
        # -----Experimental Design---------------------------
        self.X_design = None
        self.D_design = None
        self.L_design = None
        self.Y_design = None

        # -----Best from Experimental Design-----------------
        self.X_best_design = None
        self.D_best_design = None
        self.L_best_design = None
        self.Y_best_design = None

        # -----Current Iteration-----------------------------
        self.X = None
        self.D = None
        self.L = None
        self.Y = None

        # -----Best up to the Current Iteration--------------
        self.X_best = None
        self.D_best = None
        self.L_best = None
        self.Y_best = None

        # -----History of Data-------------------------------
        self.X_history = None
        self.D_history = None
        self.L_history = None
        self.Y_history = None

        # -----Historical Progression of Best Data-----------
        self.X_best_history = None
        self.D_best_history = None
        self.L_best_history = None
        self.Y_best_history = None

        # -----Others----------------------------------------
        self.iteration = 0
        self.requested_points = 0
        self.target_velocty = target_velocity   # km/hr input value
        self.plane = plane_index    # choice of plane is not an optimization variable (default is 0)
        self.beam = beam_index  # choice of beam is not an optimization variable (default is 0)
        self.plane_battery_cell = plane_battery_cell
        self.plane_data, self.beam_data, self.quad_battery_data, self.plane_battery_data, self.motor_data, self.propeller_data = self.load_component_info()
        self.variables, self.X_invalid = self.define_variables(beam_length_range)
        self.invalid_constraints_log = []
        self.space = GPyOpt.core.task.space.Design_space(self.variables)
        return None
    
    def load_component_info(self):
        def _load_component(sheet_name):
            if sheet_name in ['quad_battery', 'plane_battery']:
                type = sheet_name
                sheet_name = 'battery'
            else:
                type = None
            component = []
            componentFile = str(self.Datafolder) + '/MBO/components.xlsx'
            componentData = xw.Book(str(pathlib.PureWindowsPath(componentFile)))
            sheet = componentData.sheets[sheet_name]
            active_columns = sheet.range('A1').end('right').column
            active_rows = sheet.range('A1').end('down').row
            labels = sheet.range((1, 1), (1, active_columns)).value
            for row in range(2, active_rows+1): # 1-index, skip label row
                data = sheet.range((row, 1), (row, active_columns)).value
                entry = dict(zip(labels, data))
                if type == 'plane_battery' and entry.get('cell') not in self.plane_battery_cell:
                    continue
                component.append(entry)
            return component
        quad_battery_data = _load_component('quad_battery')
        plane_battery_data = _load_component('plane_battery')
        plane_data = _load_component('plane')
        beam_data = _load_component('beam')
        motor_data = _load_component('motor')
        propeller_data = _load_component('propeller')

        return plane_data, beam_data, quad_battery_data, plane_battery_data, motor_data, propeller_data

    def define_variables(self, beam_length_range=(None, None)):
        plane = self.plane_data[self.plane]
        if beam_length_range == (None, None):
            beam_length_range = (50, 100)
        centerline_distance_range = (plane['connection_min'], plane['connection_max'])
        pitch_range = (0, 30)
        propellers = tuple(range(len(self.propeller_data)))
        motors = tuple(range(len(self.motor_data)))
        quad_batteries = tuple(range(len(self.quad_battery_data)))
        plane_batteries = tuple(range(len(self.plane_battery_data)))
        variables = \
            [
            {'name': 'motor', 'type': 'discrete', 'domain': motors},    # motor selection
            {'name': 'propeller', 'type': 'discrete', 'domain': propellers},    # propeller selection
            {'name': 'quad_battery', 'type': 'discrete', 'domain': quad_batteries},    # battery seletion for quadcopter motors
            {'name': 'plane_battery', 'type': 'discrete', 'domain': plane_batteries},    # battery seletion for plane
            {'name': 'distanceFromCenterline', 'type': 'continuous', 'domain': centerline_distance_range},  # connection point from center of fuselage
            {'name': 'beam_length', 'type': 'continuous', 'domain': beam_length_range},  # length of beam
            {'name': 'pitch', 'type': 'continuous', 'domain': pitch_range},    # pitch of the SLT hybrid
            ]             
        X_invalid = np.asarray([0, 0, 0, 0, 0, 0, 0]) # initialize array of invalid values to be ignored in optimization
        return variables, X_invalid

    def estimate_pitch(self, X=None):
        pitchFile = str(self.Datafolder) + '/CFD/Pitch.xls'
        if X is None:
            X = self.X
        plane = self.plane_data[self.plane] 
        beam = self.beam_data[self.beam]
        pitch = []
        for configuration in range(X.shape[0]):
            motor = self.motor_data[int(X[configuration, 0])]
            propeller = self.propeller_data[int(X[configuration, 1])]
            quad_battery = self.quad_battery_data[int(X[configuration, 2])]
            plane_battery = self.plane_battery_data[int(X[configuration, 3])]
            beam_length = X[configuration, 5]
            W = (plane['weight'] 
                + beam_length * beam['weight_per_L']
                + 4 * propeller['weight']
                + 4 * motor['weight']
                + quad_battery['weight']
                + plane_battery['weight']) * 0.0098      # convert from grams to Newtons
            pitchData = xw.Book(str(pathlib.PureWindowsPath(pitchFile)))
            weight_max = pitchData.sheets['Interface'].range('B2').value
            if W < weight_max:
                pitchData.sheets['Interface'].range('B1').value = W
                estimate = pitchData.sheets['Interface'].range('B3').value
            else:
                estimate = None
            pitch.append(estimate)
        return pitch

    def process_actual_results(self, X=None):
        evaluationsFile = str(self.Datafolder) + '/MBO/evaluations.xls'
        if X is None:
            X = self.X
        batch_size = X.shape[0]
        pitch = self.estimate_pitch(X)
        evaluationsData = xw.Book(str(pathlib.PureWindowsPath(evaluationsFile)))
        for configuration in range(batch_size):
            row = configuration+2 # skip label row and convert from 0-index to 1-index
            evaluationsData.sheets['current'].range(f'A{row}:D{row}').value = [int(X[configuration, value]+1) for value in range(4)] #convert to 1-index for user
            evaluationsData.sheets['current'].range(f'E{row}:F{row}').value = [float(X[configuration, value]) for value in [4, 5]] 
            evaluationsData.sheets['current'].range(f'G{row}').value = float(pitch[configuration])
        evaluationsData.save()
        input(f'Please fill up the output column.\nPress any key when done.') #can include option to exclude configurations due to faulty CFD data
        D = evaluationsData.sheets['current'].range(f'H2:H{batch_size+1}').value
        L = evaluationsData.sheets['current'].range(f'I2:I{batch_size+1}').value
        if all([value is not None for value in D + L]): #check if drag and lift values are all filled
            Y = self.calculate_endurance_estimate(X, D)
            evaluationsData.sheets['current'].range(f'J2:J{batch_size+1}').value = Y
            currentData = evaluationsData.sheets['current'].range(f'A2:J{batch_size+1}').value
            evaluationsData.sheets['all'].range(f'A{self.requested_points+2}:J{self.requested_points+batch_size+2}').value = currentData #skip label row
            self.requested_points += batch_size
            evaluationsData.sheets['current'].range(f'A2:J{batch_size+1}').value = [['' for _ in range(10)] for _ in range(batch_size)] #10 I/O columns
            evaluationsData.save()   
            D = np.asarray(D)
            L = np.asarray(L)
        else:
            print(f'\nInvalid entries. Try again...')
            D, L, Y = self.process_actual_results(X)
        return D, L, Y

    def calculate_endurance_estimate(self, X=None, D=None):
        if X is None and D is None:
            X = self.X
            D = self.D
        planeFile = str(self.Datafolder) + '/MBO/RCbenchmark/plane.xlsx' 
        planeData = xw.Book(str(pathlib.PureWindowsPath(planeFile)))
        Y = []
        for configuration in range(X.shape[0]):
            plane_battery = self.plane_battery_data[int(X[configuration, 3])]
            planeData.sheets['Interface'].range('B1').value = D[configuration]
            eff = planeData.sheets['Interface'].range('B5').value # motor efficiency varies based on thrust to overcome drag
            if plane_battery['battery_hour_rating'] == 'X':
                R = 1      # battery hour rating is typically equal to 1hr
            else:
                R = plane_battery['battery_hour_rating']
            n = 1.3     # Peukert exponent is typically equal to 1.3 for LiPo batteries
            V = plane_battery['cell'] * 3.7     # 3.7 volts in each LiPo cell
            C = plane_battery['mah'] / 1000       # convert from mAh to Ah
            U = self.target_velocty * (1000/3600)      # get the flight velocity in m/s from km/hr input value
            E = R**(1-n) * ((eff * V * C)/(D[configuration] * U)) ** n
            Y.append(E)
        return Y

    def pre_check_constraints(self, X=None):
        if X is None:
            X = self.X
        plane = self.plane_data[self.plane] 
        beam = self.beam_data[self.beam]
        all_valid = False
        invalidConfigurations_indices = []
        problems_log = [None]

        def add_log(configuration, problem):
            if problems_log[configuration] is None:
                problems_log[configuration] = problem
            else:
                problems_log[configuration] + problem

            if configuration not in invalidConfigurations_indices:
                invalidConfigurations_indices.append(configuration)

        for configuration in range(X.shape[0]):
            motor_index = int(X[configuration, 0])
            propeller_index = int(X[configuration, 1])
            quad_battery_index = int(X[configuration, 2])
            plane_battery_index = int(X[configuration, 3])
            distanceFromCEnterline = X[configuration, 4]
            beam_length = X[configuration, 5]

            motor = self.motor_data[motor_index]
            propeller = self.propeller_data[propeller_index]
            quad_battery = self.quad_battery_data[quad_battery_index]
            plane_battery = self.plane_battery_data[plane_battery_index]

            # physical clashing
            if distanceFromCEnterline + (propeller['diameter'] / 2) < plane['connection_min']:
                add_log(configuration, 'Propellers clash with fuselage. ')
            if beam_length < propeller['diameter']:
                add_log(configuration, 'Propellers clash together.')

            # electrical compatibility
            battery_cellCount = quad_battery['cell']
            if int(battery_cellCount) <= int(motor['cell_min']): 
                add_log(configuration, 'Cell count below minimum. ')
            elif int(battery_cellCount) >= int(motor['cell_max']): 
                add_log(configuration, 'Cell count above maximum. ')
            else:
                testFile = (str(self.Datafolder) + '/MBO/RCbenchmark/' 
                        + 'M' + '{:02}'.format(int(motor_index+1)) + '_' 
                        + 'P' + '{:02}'.format(int(propeller_index+1)) + '_'
                        + 'B' + battery_cellCount + '.xlsx')
                testData = xw.Book(str(pathlib.PureWindowsPath(testFile)))
                thrust_max = testData.sheets['Interface'].range('B2').value
                # weight restriction
                W = (plane['weight'] 
                    + beam_length * beam['weight_per_L']
                    + 4 * propeller['weight']
                    + 4 * motor['weight']
                    + quad_battery['weight']
                    + plane_battery['weight']) * 0.0098      # convert from grams to Newtons
                pitchFile = str(self.Datafolder) + '/CFD/Pitch.xls'
                pitchData = xw.Book(str(pathlib.PureWindowsPath(pitchFile)))
                weight_max = pitchData.sheets['Interface'].range('B2').value
                if W < weight_max:
                    add_log(configuration, 'Insufficient estimated plane lift. ')
                else:
                    planeFile = (str(self.Datafolder) + '/MBO/RCbenchmark/plane.xlsx')
                    planeData = xw.Book(str(pathlib.PureWindowsPath(planeFile)))
                    plane_thrust_max = planeData.sheets['Interface'].range('B2').value
                    drag = pitchData.sheets['Interface'].range('B4').value
                    if plane_thrust_max < drag:
                        add_log(configuration, 'Insuffucient estimated plane thrust. ')
                if W > 4 * thrust_max:
                    add_log(configuration, 'Insufficient quadcopter thrust. ')
                else:
                    testData.sheets['Interface'].range('B1').value = W/4
                    current_draw = testData.sheets['Interface'].range('B3').value
                    if current_draw < (quad_battery['mah'] / 1000) * quad_battery['discharge']:
                        add_log(configuration, 'Too much current draw. ')        

        invalid_configurations = np.asarray(X[invalidConfigurations_indices])
        if (invalid_configurations.ndim and invalid_configurations.size) == 0: #check if no configurations are invalid
            all_valid = True

        return all_valid, invalid_configurations, problems_log

    def post_check_constraints(self, X=None, D=None, L=None):
        if X is None:
            X = self.X
            D = self.D
            L = self.L
        plane = self.plane_data[self.plane] 
        beam = self.beam_data[self.beam]
        all_valid = False
        invalidConfigurations_indices = []
        problems_log = [None]

        def add_log(configuration, problem):
            if problems_log[configuration] is None:
                problems_log[configuration] = problem
            else:
                problems_log[configuration] + problem
            if configuration not in invalidConfigurations_indices:
                invalidConfigurations_indices.append(configuration)

        for configuration in range(X.shape[0]):
            motor_index = int(X[configuration, 0])
            propeller_index = int(X[configuration, 1])
            quad_battery_index = int(X[configuration, 2])
            plane_battery_index = int(X[configuration, 3])
            beam_length = X[configuration, 5]

            motor = self.motor_data[motor_index]
            propeller = self.propeller_data[propeller_index]
            quad_battery = self.quad_battery_data[quad_battery_index]
            plane_battery = self.plane_battery_data[plane_battery_index]

            # drag and thrust comparison
            planeFile = (str(self.Datafolder) + '/MBO/RCbenchmark/plane.xlsx')
            planeData = xw.Book(str(pathlib.PureWindowsPath(planeFile)))
            plane_thrust_max = planeData.sheets['Interface'].range('B2').value
            if plane_thrust_max < D[configuration]:
                add_log(configuration, 'Insuffucient simulated thrust. ')

            # weight restriction
            W = (plane['weight'] 
                + beam_length * beam['weight_per_L']
                + 4 * propeller['weight']
                + 4 * motor['weight']
                + quad_battery['weight']
                + plane_battery['weight']) * 0.0098      # convert from grams to Newtons
            if W < L[configuration]:
                add_log(configuration, 'Insufficient simulated plane lift. ')   
            else:
                planeData.sheets['Interface'].range('B1').value = D[configuration]       
                current_draw = planeData.sheets['Interface'].range('B3').value
                if current_draw < (quad_battery['mah'] / 1000) * quad_battery['discharge']:
                    add_log(configuration, 'Too much current draw. ')   

        invalid_configurations = np.asarray(X[invalidConfigurations_indices])
        if (invalid_configurations.ndim and invalid_configurations.size) == 0: #check if no configurations are invalid
            all_valid = True

        return all_valid, invalid_configurations, problems_log

    def get_best_values(self, X=None, D=None, L=None, Y=None):
        if X is None and D is None and Y is None:
            X = self.X_history
            D = self.D_history
            L = self.L_history
            Y = self.Y_history
        index_best = np.argmin(Y)
        X_best = X[index_best]
        D_best = D[index_best]
        L_best = L[index_best]
        Y_best = Y[index_best]
        return X_best, D_best, L_best, Y_best

    def run_optimization(self, num_design = 20, num_iteration = 50):
        # Experimental Design
        self.X_design = GPyOpt.experiment_design.LatinMixedDesign(self.space).get_samples(num_design)
        self.D_design, self.L_design, self.Y_design = self.process_actual_results(self.X_design)
        self.X_history, self.D_history, self.L_history, self.Y_history = self.X_design, self.D_design, self.L_design, self.Y_design    # Initialize history of data

        self.X_best_design, self.D_best_design, self.L_bdest_design, self.Y_best_design = self.get_best_values(self.X_design, self.D_design, self.L_design, self.Y_design)
        self.X_best, self.D_best, self.Y_best = self.X_best_design, self.D_best_design, self.Y_best_design  # Initialize current best value
        self.X_best_history, self.D_best_history, self.Y_best_history = self.X_best_design, self.D_best_design, self.Y_best_design # Initialize history of best values

        # Iterations
        for step in range(num_iteration):
            mixed_problem = GPyOpt.methods.BayesianOptimization(
                f = None, 
                domain = self.variables,
                constraints = None,
                cost_withGradients = None,
                model_type = 'GP',
                X = self.X_history,
                Y = self.Y_history,
                acquisition_type = 'EI',
                normalize_Y = True,
                exact_feval = False,
                acquisition_optimizer_type = 'lbfgs',
                evaluator_type = 'local_penalization',
                batch_size = 10,
                maximize = False,
                de_duplication = True,
                Gower = False,
                noise_var = 0)

            # Precheck constraints prior to suggesting for evaluation
            # If unsatisfactory, and add to invalid configurations
            all_valid = False
            while not all_valid:
                self.X = np.asarray(mixed_problem.suggest_next_locations(ignored_X = np.vstack((self.X_history, self.X_invalid))))
                all_valid, invalid_configurations, problems_log = self.pre_check_constraints(self.X)
                self.X_invalid = np.vstack((self.X_invalid, invalid_configurations))
                self.invalid_constraints_log = np.vstack((self.invalid_constraints_log, problems_log))

            # Recheck constraints after obtaining lift and drag in simulations
            # If unsatisfactory, do not add to history, and add to invalid configurations
            self.D, self.L, self.Y = self.process_actual_results(self.X)
            all_valid, invalid_configurations, problems_log = self.post_check_constraints(self.X, self.D, self.L)
            self.X_invalid = np.vstack((self.X_invalid, invalid_configurations))
            self.invalid_constraints_log = np.vstack((self.invalid_constraints_log, problems_log))

            # Update history of valid configurations
            self.X_history = np.vstack((self.X_history, self.X))
            self.D_history = np.vstack((self.D_history, self.D))
            self.L_history = np.vstack((self.L_history, self.L))
            self.Y_history = np.vstack((self.Y_history, self.Y))

            # Update history of best configurations per iteration
            self.X_best, self.D_best, self.L_best, self.Y_best = self.get_best_values(self.X_history, self.D_history, self.L_history, self.Y_history)
            self.X_best_history = np.vstack((self.X_best_history, self.X_best))
            self.D_best_history = np.vstack((self.D_best_history, self.D_best))
            self.L_best_history = np.vstack((self.L_best_history, self.L_best))
            self.Y_best_history = np.vstack((self.Y_best_history, self.Y_best))

            print(f'Iteration {step+1}')
            mixed_problem.plot_convergence()
            print(f'New location/s: {[tuple(point) for point in self.X]}\n')
        
        return None

# endregion        

thesis = SLT_Optimization(45)
thesis.run_optimization(num_design = 20, num_iteration = 50)