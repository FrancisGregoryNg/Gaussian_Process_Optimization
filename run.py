# region [Imports]
import GPyOpt
import xlrd
import xlwt
import xlutils.copy
import pathlib
import matplotlib
import numpy as np
np.set_printoptions(linewidth=200, precision=4)
# endregion

# region [Definitions]
class SLT_Optimization():
    Datafolder = pathlib.Path(__file__).parents[2].joinpath('2---Data')
    def __init__(self, target_velocity, plane_index=0, beam_index = 0, beam_length_range=(None, None), beam_clearance_range=(None, None)):
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
                  
        Y = output variables
                  - endurance estimate          - calculated value
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
        self.target_velocty = target_velocity   # km/hr input value
        self.plane = plane_index    # choice of plane is not an optimization variable (default is 0)
        self.beam = beam_index  # choice of beam is not an optimization variable (default is 0)
        self.plane_data, self.beam_data, self.battery_data, self.motor_data, self.propeller_data = self.load_component_info()
        self.variables = self.define_variables(beam_length_range, beam_clearance_range)
        self.space = GPyOpt.core.task.space.Design_space(self.variables)
        return None
    
    def load_component_info(self):
        def _load_component(sheet_name):
            component = []
            with xlrd.open_workbook(str(self.Datafolder) + '/MBO/components.xlsx', on_demand=True) as book:
                sheet = book.sheet_by_name(sheet_name)
                labels = sheet.row_values(0)
                for rowx in range(1, sheet.nrows):
                    data = sheet.row_values(rowx)
                    component.append(dict(zip(labels, data)))
            return component
        def _add_value_of_motor_for_propeller(sheet_name):
            with xlrd.open_workbook(str(self.Datafolder) + '/MBO/components.xlsx', on_demand=True) as book:
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

    def estimate_pitch(self, X=None):
        file = str(self.Datafolder) + '/CFD/Pitch.xlsx'
        if X is None:
            X = self.X
        plane = self.plane_data[self.plane] 
        beam = self.beam_data[self.beam]
        motor = self.motor_data[X[configuration, 0]]
        propeller = self.propeller_data[X[configuration, 1]]
        battery = self.battery_data[X[configuration, 2]]
        # ----- Write style -----
        style = xlwt.XFStyle()
        style.font = xlwt.Font()
        style.font.name = 'Levenim MT'
        style.font.height = 200 # multiply font size by 20
        style.alignment = xlwt.Alignment()
        style.alignment.horz = xlwt.Alignment.HORZ_CENTER
        style.alignment.vert = xlwt.Alignment.VERT_CENTER
        # ----------------------        
        pitch = []
        for configuration in range(X.shape[0]):
            beam_length = X[configuration, 4]
            W = (plane['weight'] 
                + beam_length * beam['weight_per_L']
                + 4 * propeller['weight']
                + 4 * motor['weight']
                + battery['weight']) * 0.0098      # convert from grams to Newtons
            with xlrd.open_workbook(file, formatting_info = True, on_demand=True) as book:
                book_write = xlutils.copy.copy(book)
            sheet = book_write.get_sheet(0)     # write configurations to the sheet named "Summary"
            sheet.write(0, 1, float(W), style)  # write in field for "weight"
            book_write.save(file)        
            with xlrd.open_workbook(file, on_demand=True) as book:
                sheet = book.sheet_by_name('Summary')
                estimate = sheet.row_value(1)[1]
            pitch.append(estimate)
        return pitch

    def request_actual_results(self, X=None):
        file = str(self.Datafolder) + '/MBO/evaluations.xls'
        if X is None:
            X = self.X
        batch_size = X.shape[0]
        variables = X.shape[1]
        pitch = self.estimate_pitch(X)
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
            sheet.write(configuration+1, variables, float(pitch(configuration), style))
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
            motor = self.motor_data[X[configuration, 0]]
            propeller = self.propeller_data[X[configuration, 1]]
            battery = self.battery_data[X[configuration, 2]]
            R = battery['battery_hour_rating']      # battery hour rating is typically equal to 1hr
            n = 1.3     # Peukert exponent is typically equal to 1.3 for LiPo batteries
            eff = motor['efficiency'][propeller['name']]  # motor-propeller efficiency is stored in the motor data and can be accessed using the propeller's name
            V = battery['cell'] * 3.7     # 3.7 volts in each LiPo cell
            C = battery['mah'] / 1000       # convert from mAh to Ah
            U = self.target_velocty * (1000/3600)      # get the flight velocity in m/s from km/hr input value
            E = R**(1-n) * ((eff * V * C)/(D * U)) ** n
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
        # Experimental Design
        self.X_design = GPyOpt.experiment_design.LatinMixedDesign(self.space).get_samples(num_design)
        self.D_design = self.request_actual_results(self.X_design)
        self.Y_design = self.calculate_endurance_estimate(self.X_design, self.D_design)
        self.X_history, self.D_history, self.Y_history = self.X_design, self.D_design, self.Y_design    # Initialize history of data

        self.X_best_design, self.D_best_design, self.Y_best_design = self.get_best_values(self.X_design, self.D_design, self.Y_design)
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

            self.X = np.asarray(mixed_problem.suggest_next_locations(ignored_X = self.X_history))
            self.D = self.request_actual_results(self.X)
            self.Y = self.calculate_endurance_estimate(self.X, self.D)
            self.X_history = np.vstack(self.X_history, self.X)
            self.D_history = np.vstack(self.D_history, self.D)
            self.Y_history = np.vstack(self.Y_history, self.Y)

            self.X_best, self.D_best, self.Y_best = self.get_best_values(self.X_history, self.D_history, self.Y_history)
            self.X_best_history = np.vstack(self.X_best_history, self.X_best)
            self.D_best_history = np.vstack(self.D_best_history, self.D_best)
            self.Y_best_history = np.vstack(self.Y_best_history, self.Y_best)

            print(f'Iteration {step+1}')
            mixed_problem.plot_convergence()
            print(f'New location/s: {[tuple(point) for point in self.X]}\n')
        
        return None

# endregion        

thesis = SLT_Optimization(45)
thesis.run_optimization(num_design = 20, num_iteration = 50)
