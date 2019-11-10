# region [Imports]
import GPyOpt
import xlwings as xw
import pathlib
import chaospy
import math
import numpy as np
np.set_printoptions(linewidth=200, precision=4)
# endregion

# region [Definitions]
class SLT_Optimization():
    Datafolder = pathlib.Path(__file__).parents[2].joinpath('2---Data')
    def __init__(self, target_velocity, payload_weight=500, motor_angle = 5, plane_index=0, beam_index = 0, plane_battery_cell = [3], beam_length_range=(None, None)):
        '''
        X = input variables 
                0 - 'motor'                     - motor selection
                1 - 'propeller'                 - propeller selection
                2 - 'battery'                   - battery selection
                3 - 'distanceFromCenterline'    - connection point from the center of the fuselage
                4 - 'beam_length'               - length of beam
                
        D = drag force
                  - data from simulations       - intermediate between X and Y
        
        L = lift force
                  - data from simulations       - used for checking actual flight capability
                  
        Y = output variables
                  - endurance estimate          - calculated value
        '''
        print(f'''
        {'-'*50}
                        Initialization
        {'-'*50}
        ''')
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
        self.payload_weight = payload_weight    # weight of payload in grams
        self.motor_angle = motor_angle  # angle of plane motor with respect to a flat wing (0° pitch)
        self.plane = plane_index    # choice of plane is not an optimization variable (default is 0)
        self.beam = beam_index  # choice of beam is not an optimization variable (default is 0)
        self.plane_battery_cell = plane_battery_cell
        self.plane_data, self.beam_data, self.quad_battery_data, self.plane_battery_data, self.motor_data, self.propeller_data = self.load_component_info()
        self.variables, self.X_invalid = self.define_variables(beam_length_range)
        self.invalid_constraints_log = []
        self.space = GPyOpt.core.task.space.Design_space(self.variables)
        return None
    
    def load_component_info(self):
        componentFile = str(self.Datafolder) + '/MBO/components.xlsx'
        componentData = xw.Book(str(pathlib.PureWindowsPath(componentFile)))
        print('Loading component information...\n')
        
        def _load_component(sheet_name):
            if sheet_name in ['quad_battery', 'plane_battery']:
                type = sheet_name
                sheet_name = 'battery'
            else:
                type = None
            component = []
            sheet = componentData.sheets[sheet_name]
            active_columns = sheet.range('A1').end('right').column
            active_rows = sheet.range('A1').end('down').row
            labels = sheet.range((1, 1), (1, active_columns)).value
            for row in range(2, active_rows+1): # 1-index, skip label row
                data = sheet.range((row, 1), (row, active_columns)).value
                entry = dict(zip(labels, data))
                if type == 'plane_battery' and entry.get('cell') not in self.plane_battery_cell:
                    continue
                if sheet_name == 'motor' or sheet_name == 'propeller':
                    if entry['name'][0] == 'X':
                        continue
                component.append(entry)
            return component

        quad_battery_data = _load_component('quad_battery')
        plane_battery_data = _load_component('plane_battery')
        plane_data = _load_component('plane')
        beam_data = _load_component('beam')
        motor_data = _load_component('motor')
        propeller_data = _load_component('propeller')

        for motor_index, motor in enumerate(motor_data, start=0):   # internally 0-index (1-index only in filenames and Excel)
            propeller_compatibility = componentData.sheets['test'].range((motor_index+2, 2), (motor_index+2, len(propeller_data)+1)).value
            propeller = [propeller_compatibility.index(propeller) for propeller in propeller_compatibility if propeller != 'X']
            motor['propeller'] = propeller
            battery = [quad_battery_data.index(battery) for battery in quad_battery_data if round(motor['cell_min']) <= round(battery['cell']) <= round(motor['cell_max'])]
            motor['battery'] = battery

        data = [plane_data, beam_data, quad_battery_data, plane_battery_data, motor_data, propeller_data]
        labels = ['Plane', 'Beam', 'Quadcopter Batteries', 'Plane Batteries', 'Motor', 'Propeller']

        for index in range(len(data)):
            print(f'    [{labels[index]}]')
            [print(item) for item in data[index]]
            print()

        return plane_data, beam_data, quad_battery_data, plane_battery_data, motor_data, propeller_data

    def define_variables(self, beam_length_range=(None, None)):
        plane = self.plane_data[self.plane]
        if beam_length_range == (None, None):
            beam_length_range = (50, 100)
        centerline_distance_range = (plane['connection_min'], plane['connection_max'])
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
            ]             
        X_invalid = [[0, 0, 0, 0, 0, 0, 0]] # initialize array of invalid values to be ignored in optimization
        print('    [Variables]')
        [print(variable) for variable in variables]
        return variables, X_invalid

    def estimate_pitch(self, X=None):
        pitchFile = str(self.Datafolder) + '/CFD/Angle_of_Attack.xlsx'
        if X is None:
            X = self.X
        plane = self.plane_data[self.plane] 
        beam = self.beam_data[self.beam]
        pitch = []
        for configuration in range(len(X)):
            motor = self.motor_data[round(X[configuration][0])]
            propeller = self.propeller_data[round(X[configuration][1])]
            quad_battery = self.quad_battery_data[round(X[configuration][2])]
            plane_battery = self.plane_battery_data[round(X[configuration][3])]
            beam_length = X[configuration][5]
            W = (plane['weight'] 
                + beam_length * beam['weight_per_L']
                + 4 * propeller['weight']
                + 4 * motor['weight']
                + quad_battery['weight']
                + plane_battery['weight']) * 0.0098      # convert from grams to Newtons
            pitchData = xw.Book(str(pathlib.PureWindowsPath(pitchFile)))
            weight_max = float(pitchData.sheets['Interface'].range('B2').value)
            if W < weight_max:
                pitchData.sheets['Interface'].range('B1').value = W
                estimate = float(pitchData.sheets['Interface'].range('B3').value)
            else:
                estimate = None
            pitch.append(estimate)
        return pitch

    def process_actual_results(self, X=None, penalized_values=None):
        evaluationsFile = str(self.Datafolder) + '/MBO/evaluations.xls'
        print('Setting up evaluations...\n')

        if X is None:
            X = self.X
            penalized_values = [0 for _ in self.X]
        batch_size = len(X)
        pitch = self.estimate_pitch(X)
        evaluationsData = xw.Book(str(pathlib.PureWindowsPath(evaluationsFile)))
        for configuration in range(batch_size):
            row = configuration+2 # skip label row and convert from 0-index to 1-index
            evaluationsData.sheets['current'].range(f'A{row}:D{row}').value = [round(X[configuration][value]+1) for value in range(4)] #convert to 1-index for user
            evaluationsData.sheets['current'].range(f'E{row}:F{row}').value = [float(X[configuration][value]) for value in [4, 5]] 
            evaluationsData.sheets['current'].range(f'G{row}').value = float(pitch[configuration])
            if penalized_values[configuration] != 0:
                evaluationsData.sheets['current'].range(f'H{row}').value = 'None'
                evaluationsData.sheets['current'].range(f'I{row}').value = 'None'
        evaluationsData.save()
        print('\n    [ USER INSTRUCTION ]:')
        input(f'Please fill up the output columns (orange) in "{str(pathlib.PureWindowsPath(evaluationsFile))}".\nKey in "Enter" when done.')
        print('Processing results...\n')
        D = evaluationsData.sheets['current'].range(f'H2:H{batch_size+1}').value
        L = evaluationsData.sheets['current'].range(f'I2:I{batch_size+1}').value
        if all([value is not None for value in D + L]): #check if drag and lift values are all filled
            Y = self.calculate_endurance_estimate(X, D, penalized_values)
            evaluationsData.sheets['current'].range(f'J2:J{batch_size+1}').value = [[endurance] for endurance in Y]
            currentData = evaluationsData.sheets['current'].range(f'A2:J{batch_size+1}').value
            evaluationsData.sheets['all'].range(f'A{self.requested_points+2}:J{self.requested_points+batch_size+2}').value = currentData #skip label row
            self.requested_points += batch_size
            evaluationsData.sheets['current'].range(f'A2:J{batch_size+1}').value = [['' for _ in range(10)] for _ in range(batch_size)] #10 I/O columns
            evaluationsData.save()   
        else:
            print(f'\nInvalid entries. Try again...')
            D, L, Y = self.process_actual_results(X, penalized_values)

        print('┌────────┬───────┬───────────┬──────────────┬───────────────┬─────────────────────┬─────────────┬──────────┬──────────┬───────────┐')
        print('│ Design │ Motor │ Propeller │ Quad battery │ Plane battery │ Centerline distance │ Beam length │   Drag   │   Lift   │ Endurance │')
        print('├────────┼───────┼───────────┼──────────────┼───────────────┼─────────────────────┼─────────────┼──────────┼──────────┼───────────┤')
        [print(f'│   {index:2d}   │ {point[0]+1:3d}   │ {point[1]+1:4d}      │    {point[2]+1:5d}     │    {point[3]+1:5d}      ' +
                f'│      {point[4]:8.2f}       │  {point[5]:8.2f}   │ {D[index-1]:7.2f}  │ {L[index-1]:7.2f}  │ {Y[index-1]:7.2f}   │') 
            if D[index-1] != 'None' else
            print(f'│   {index:2d}   │ {point[0]+1:3d}   │ {point[1]+1:4d}      │    {point[2]+1:5d}     │    {point[3]+1:5d}      ' +
                f'│      {point[4]:8.2f}       │  {point[5]:8.2f}   │    None  │    None  │ {Y[index-1]:7.2f}   │')
            for index, point in enumerate(X, start=1)]
        print('└────────┴───────┴───────────┴──────────────┴───────────────┴─────────────────────┴─────────────┴──────────┴──────────┴───────────┘\n')

        return D, L, Y

    def calculate_endurance_estimate(self, X=None, D=None, penalized_values=None):
        if X is None and D is None:
            X = self.X
            D = self.D
            penalized_values = [0 for _ in self.X]
        planeFile = str(self.Datafolder) + '/MBO/RCbenchmark/plane.xlsx' 
        planeData = xw.Book(str(pathlib.PureWindowsPath(planeFile)))
        Y = []
        for configuration in range(len(X)):
            if D[configuration] == 'None':
                Y.append(penalized_values[configuration])
            else:
                plane_battery = self.plane_battery_data[round(X[configuration][3])]
                planeData.sheets['Interface'].range('B1').value = D[configuration]
                eff = float(planeData.sheets['Interface'].range('B5').value) # motor efficiency varies based on thrust to overcome drag
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
        print('Conducting preliminary constraint checks...\n')
        if X is None:
            X = self.X
        plane = self.plane_data[self.plane] 
        beam = self.beam_data[self.beam]
        all_valid = False
        invalidConfigurations_indices = []
        problems_log = [None for configuration in range(len(X))]
        penalized_values = [0 for configuration in range(len(X))]
        pitch_estimates = self.estimate_pitch(X)

        def add_log(configuration, problem):
            if problems_log[configuration] == None:
                problems_log[configuration] = problem
            else:
                problems_log[configuration] += problem
            if configuration not in invalidConfigurations_indices:
                invalidConfigurations_indices.append(configuration)

        def add_penalty(configuration, penalty):
            if penalized_values[configuration] == 0:
                penalized_values[configuration] = penalty
            else:
                penalized_values[configuration] += penalty
            if configuration not in invalidConfigurations_indices:
                invalidConfigurations_indices.append(configuration)

        for configuration in range(len(X)):
            motor_index = round(X[configuration][0])
            propeller_index = round(X[configuration][1])
            quad_battery_index = round(X[configuration][2])
            plane_battery_index = round(X[configuration][3])
            distanceFromCenterline = X[configuration][4]
            beam_length = X[configuration][5]
            pitch = float(pitch_estimates[configuration])
            motor = self.motor_data[motor_index]
            propeller = self.propeller_data[propeller_index]
            quad_battery = self.quad_battery_data[quad_battery_index]
            plane_battery = self.plane_battery_data[plane_battery_index]
            print(f'#{configuration+1:2d}: [{X[configuration][0]+1:3d} {X[configuration][1]+1:3d}' +
                f'{X[configuration][2]+1:4d} {X[configuration][3]+1:4d} {X[configuration][4]:7.2f}' +
                f'{X[configuration][5]:7.2f}]')
            # motor-propeller compatibility
            if propeller_index not in motor['propeller']:
                add_log(configuration, 'Incompatible propeller. ')
                add_penalty(configuration, -1)
            else:
                # physical clashing
                if distanceFromCenterline - (propeller['diameter'] / 2) < plane['connection_min']:
                    minimumFuselageDistance = plane['connection_min'] + (propeller['diameter'] / 2) 
                    distanceCorrection = (distanceFromCenterline - minimumFuselageDistance) / minimumFuselageDistance
                    add_log(configuration, f'Propellers clash with fuselage ({distanceCorrection*100:.2f}% difference from minimum fuselage distance). ')
                    add_penalty(configuration, distanceCorrection)
                if beam_length < propeller['diameter']:
                    lengthCorrection = (beam_length - propeller['diameter']) / propeller['diameter']
                    add_log(configuration, f'Propellers clash together ({lengthCorrection*100:.2f}% difference from minimum beam length). ')
                    add_penalty(configuration, lengthCorrection)
                # electrical compatibility
                battery_cellCount = quad_battery['cell']
                if battery_cellCount < motor['cell_min']:
                    voltageCorrection = (battery_cellCount - motor['cell_min']) / motor['cell_min']
                    add_log(configuration, f'Cell count below minimum ({voltageCorrection*100:.2f}% difference from minimum voltage). ')
                    add_penalty(configuration, voltageCorrection)
                elif battery_cellCount > motor['cell_max']: 
                    voltageCorrection = (motor['cell_max'] - battery_cellCount) / motor['cell_max']
                    add_log(configuration, f'Cell count above maximum ({-voltageCorrection*100:.2f}% difference from maximum voltage). ')
                    add_penalty(configuration, voltageCorrection)
                else:
                    testFile = (str(self.Datafolder) + '/MBO/RCbenchmark/' 
                            + 'M' + '{:02}'.format(round(motor_index+1)) + '_' 
                            + 'P' + '{:02}'.format(round(propeller_index+1)) + '_'
                            + 'B' + '{:02}'.format(round(battery_cellCount)) + '.xlsx')
                    testData = xw.Book(str(pathlib.PureWindowsPath(testFile)))
                    thrust_max = float(testData.sheets['Interface'].range('B2').value)
                    # weight restriction
                    W = (plane['weight'] 
                        + beam_length * beam['weight_per_L']
                        + 4 * propeller['weight']
                        + 4 * motor['weight']
                        + quad_battery['weight']
                        + plane_battery['weight']
                        + self.payload_weight) * 0.0098      # convert from grams to Newtons
                    pitchFile = str(self.Datafolder) + '/CFD/Angle_of_Attack.xlsx'
                    pitchData = xw.Book(str(pathlib.PureWindowsPath(pitchFile)))
                    weight_max = float(pitchData.sheets['Interface'].range('B2').value)
                    planeFile = (str(self.Datafolder) + '/MBO/RCbenchmark/plane.xlsx')
                    planeData = xw.Book(str(pathlib.PureWindowsPath(planeFile)))
                    plane_thrust_max = float(planeData.sheets['Interface'].range('B2').value)
                    vertical_thrust_component =  math.sin(math.radians(pitch - self.motor_angle))
                    if W > weight_max + plane_thrust_max * vertical_thrust_component:
                        planeLiftCorrection = (weight_max - W) / W
                        add_log(configuration, f'Insufficient estimated plane lift ({planeLiftCorrection*100:.2f}% above total weight). ')
                        add_penalty(configuration, planeLiftCorrection)
                    else:
                        drag = float(pitchData.sheets['Interface'].range('B4').value)
                        horizontal_thrust_component =  math.cos(math.radians(pitch - self.motor_angle))
                        if plane_thrust_max * horizontal_thrust_component < drag:
                            planeThrustCorrection = (plane_thrust_max - drag) / drag
                            add_log(configuration, f'Insuffucient plane thrust ({planeThrustCorrection*100:.2f}% difference from estimated drag). ')
                            add_penalty(configuration, planeThrustCorrection)
                    if W > 4 * thrust_max:
                        quadcopterThrustCorrection = (4 * thrust_max - W) / W
                        add_log(configuration, f'Insufficient quadcopter thrust ({quadcopterThrustCorrection*100:.2f}% difference from total weight). ')
                        add_penalty(configuration, quadcopterThrustCorrection)
                    else:
                        testData.sheets['Interface'].range('B1').value = W/4
                        current_draw = float(testData.sheets['Interface'].range('B3').value)
                        if current_draw > (quad_battery['mah'] / 1000) * quad_battery['discharge']:
                            currentLimit = (quad_battery['mah'] / 1000) * quad_battery['discharge']
                            currentCorrection = (currentLimit - current_draw) / currentLimit
                            add_log(configuration, f'Too much quadcopter current draw ({-currentCorrection*100:.2f}% difference from battery limit). ')
                            add_penalty(configuration, currentCorrection)  
            if problems_log[configuration] is not None: 
                print(f'  └──> {problems_log[configuration]}\n')
        invalid_configurations = [X[index] for index in invalidConfigurations_indices]
        if not invalid_configurations: #check if no configurations are invalid (empty list)
            print('All configurations pass the pre-simulation constraints check.\n\n')
            all_valid = True
        else:
            print(f'\n{len(invalid_configurations)} potentially invalid configurations. ' +
                  'Penalized objective values are calculated for these based on percentage violation of constraints (skip CFD, penalty on top of zero endurance).\n\n')

        return all_valid, invalid_configurations, problems_log, penalized_values

    def post_check_constraints(self, X=None, D=None, L=None, Y=None):
        print('Conducting post-simulation constraint checks... (not redundant with preliminary checks)\n')
        if X is None:
            X = self.X
            D = self.D
            L = self.L
            Y = self.Y
        plane = self.plane_data[self.plane] 
        beam = self.beam_data[self.beam]
        all_valid = False
        invalidConfigurations_indices = []
        problems_log = [None for configuration in range(len(X))]
        penalized_values = [0 for configuration in range(len(X))]
        pitch_estimates = self.estimate_pitch(X)

        def add_log(configuration, problem):
            if problems_log[configuration] == None:
                problems_log[configuration] = problem
            else:
                problems_log[configuration] += problem
            if configuration not in invalidConfigurations_indices:
                invalidConfigurations_indices.append(configuration)

        def add_penalty(configuration, penalty):
            if penalized_values[configuration] == 0:
                penalized_values[configuration] = penalty
            else:
                penalized_values[configuration] += penalty
            if configuration not in invalidConfigurations_indices:
                invalidConfigurations_indices.append(configuration)
                invalidConfigurations_indices.append(configuration)

        for configuration in range(len(X)):
            if D[configuration] != 'None':
                motor_index = round(X[configuration][0])
                propeller_index = round(X[configuration][1])
                quad_battery_index = round(X[configuration][2])
                plane_battery_index = round(X[configuration][3])
                beam_length = X[configuration][5]
                pitch = float(pitch_estimates[configuration])
                motor = self.motor_data[motor_index]
                propeller = self.propeller_data[propeller_index]
                quad_battery = self.quad_battery_data[quad_battery_index]
                plane_battery = self.plane_battery_data[plane_battery_index]
                print(f'#{configuration+1:2d}: [{X[configuration][0]+1:5d} {X[configuration][1]+1:5d}' +
                    f'{X[configuration][2]+1:5d} {X[configuration][3]+1:5d} {X[configuration][4]:8.2f}' +
                    f'{X[configuration][5]:8.2f}]')
                # drag and thrust comparison
                planeFile = (str(self.Datafolder) + '/MBO/RCbenchmark/plane.xlsx')
                planeData = xw.Book(str(pathlib.PureWindowsPath(planeFile)))
                plane_thrust_max = float(planeData.sheets['Interface'].range('B2').value)
                horizontal_thrust_component =  math.cos(math.radians(pitch - self.motor_angle))
                if plane_thrust_max * horizontal_thrust_component < D[configuration]:
                    thrustCorrection = (plane_thrust_max - D[configuration]) / D[configuration]
                    add_log(configuration, f'Insuffucient plane thrust ({thrustCorrection*100:.2f}% difference from simulated drag). ')
                    add_penalty(configuration, thrustCorrection)
                # weight restriction
                W = (plane['weight'] 
                    + beam_length * beam['weight_per_L']
                    + 4 * propeller['weight']
                    + 4 * motor['weight']
                    + quad_battery['weight']
                    + plane_battery['weight']
                    + self.payload_weight) * 0.0098      # convert from grams to Newtons
                vertical_thrust_component =  math.sin(math.radians(pitch - self.motor_angle))
                if W < L[configuration] + plane_thrust_max * vertical_thrust_component:
                    liftCorrection = (W - L[configuration]) / L[configuration]
                    add_log(configuration, f'Insufficient simulated plane lift ({liftCorrection*100:.2f}% difference from total weight). ')
                    add_penalty(configuration, liftCorrection)
                else:
                    planeData.sheets['Interface'].range('B1').value = D[configuration]       
                    current_draw = float(planeData.sheets['Interface'].range('B3').value)
                    if current_draw > (plane_battery['mah'] / 1000) * plane_battery['discharge']:
                        currentLimit = (plane_battery['mah'] / 1000) * plane_battery['discharge']
                        currentCorrection = (currentLimit - current_draw) / currentLimit
                        add_log(configuration, f'Too much plane current draw ({-currentCorrection*100:.2f}% difference from battery limit). ')
                        add_penalty(configuration, currentCorrection)
            if problems_log[configuration] is not None: 
                print(f'  └──> {problems_log[configuration]}\n')
        invalid_configurations = [X[index] for index in invalidConfigurations_indices]
        if not invalid_configurations: #check if no configurations are invalid (empty list)
            print('\nAll evaluated configurations pass the post-simulation constraints check.\n\n')
            all_valid = True
        else:
            print(f'\n{len(invalid_configurations)} evaluated configurations failed post-simulation constraint check; results are penalized.\n\n')
        penalized_Y = Y + penalized_values

        return all_valid, invalid_configurations, problems_log, penalized_Y

    def get_best_values(self, X=None, D=None, L=None, Y=None):
        if X is None and D is None and Y is None:
            X = self.X_history
            D = self.D_history
            L = self.L_history
            Y = self.Y_history
        index_best = np.argmin(np.asarray(Y))
        X_best = X[index_best]
        D_best = D[index_best]
        L_best = L[index_best]
        Y_best = Y[index_best]
        return X_best, D_best, L_best, Y_best

    def update(self, array_old, array_add):
        [array_old.append(item) for item in array_add]
        return array_old

    def generate_experimental_design(self, num_design):
        print('Generating experimental design...\n')          
        hammerseley = chaospy.distributions.sampler.sequences.hammersley
        base = hammerseley.create_hammersley_samples(num_design, dim=len(self.variables), burnin=-1, primes=()) #numpy array
        motor_selections = np.around(base[0, :] * (len(self.motor_data)-1), decimals=0).astype(int).tolist()
        propeller_selections_subindex = np.around(base[1, :] * np.asarray([len(self.motor_data[motor]['propeller'])-1 for motor in motor_selections]), decimals=0).astype(int).tolist()
        propeller_selections = [self.motor_data[motor]['propeller'][index] for (motor, index) in list(zip(motor_selections, propeller_selections_subindex))]

        quad_battery_selections_subindex = np.around(base[2, :] * np.asarray([len(self.motor_data[motor]['battery'])-1 for motor in motor_selections]), decimals=0).astype(int).tolist()
        quad_battery_selections = [self.motor_data[motor]['battery'][index] for (motor, index) in list(zip(motor_selections, quad_battery_selections_subindex))]
        plane_battery_selections = np.around(base[3, :] * len(self.plane_battery_data)-1, decimals=0).astype(int).tolist()
        distanceFromCenterline_values = (base[4, :] * self.variables[4]['domain'][1]).tolist()
        beam_length_values = (base[5, :] * self.variables[5]['domain'][1]).tolist()

        design = [[motor_selections[design], propeller_selections[design], quad_battery_selections[design], plane_battery_selections[design],
                distanceFromCenterline_values[design], beam_length_values[design]] for design in range(num_design)]
        
        def _check_if_duplicate(design, index, entry=None):
                duplicate = False
                if entry is None:
                    entry = design[index]
                    for previous_index in range(index):
                        #print(entry[0:3], design[previous_index][0:3])
                        if entry[0:3] == design[previous_index][0:3]:
                            duplicate = True
                            break
                else:
                    for all_index in range(len(design)):
                        print(entry[0:3], design[all_index][0:3])
                        if entry[0:3] == design[all_index][0:3]:
                            duplicate = True
                            break
                return duplicate

        duplicates = [index for index in range(len(design)) if _check_if_duplicate(design, index)]

        for index in duplicates:
            entry = design[index]
            attempt=0
            while _check_if_duplicate(design, index, entry) and attempt<100:
                attempt+=1
                print(f'duplicate#{index}, attempt#{attempt}')
                motor_selection = np.random.choice(range(len(self.motor_data)-1))
                propeller_selection = np.random.choice(self.motor_data[motor_selection]['propeller'])
                battery_selection = np.random.choice(self.motor_data[motor_selection]['battery'])
                entry = design[index][:]
                entry[0:3] = [motor_selection, propeller_selection, battery_selection]             
            design[index][0:3] = entry[0:3]
        print('┌────────┬───────┬───────────┬──────────────┬───────────────┬─────────────────────┬─────────────┐')
        print('│ Design │ Motor │ Propeller │ Quad battery │ Plane battery │ Centerline distance │ Beam length │')
        print('├────────┼───────┼───────────┼──────────────┼───────────────┼─────────────────────┼─────────────┤')
        [print(f'│   {index:2d}   │ {point[0]+1:3d}   │ {point[1]+1:4d}      │    {point[2]+1:5d}     │    {point[3]+1:5d}      ' +
               f'│      {point[4]:8.2f}       │  {point[5]:8.2f}   │') for index, point in enumerate(design, start=1)]
        
        print('└────────┴───────┴───────────┴──────────────┴───────────────┴─────────────────────┴─────────────┘\n')

        return design

    def run_optimization(self, num_design = 20, num_iteration = 50):
        print(f'''
        {'-'*50}
                        Experimental Design
        {'-'*50}
        ''')
        # Precheck constraints prior to suggesting for evaluation
        # For unsatisfactory configurations, skip evaluation and compute penalized objective value
        self.X_design = self.generate_experimental_design(num_design)
        all_valid, invalid_configurations, problems_log, penalized_values = self.pre_check_constraints(self.X_design)
        if not all_valid:
            self.X_invalid = self.update(self.X_invalid, invalid_configurations)
            self.invalid_constraints_log = self.update(self.invalid_constraints_log, problems_log)
        self.D_design, self.L_design, self.Y_design = self.process_actual_results(self.X_design, penalized_values)

        # Recheck constraints after obtaining lift and drag in simulations
        # For unsatifactory configurations, significantly penalize the objective value (not as severe as pre-check since base is nonzero from simulation results)
        all_valid, invalid_configurations, problems_log, self.Y_design = self.post_check_constraints(self.X_design, self.D_design, self.L_design, self.Y_design)
        if not all_valid:
            self.X_invalid = self.update(self.X_invalid, invalid_configurations)
            self.invalid_constraints_log = self.update(self.invalid_constraints_log, problems_log)      

        self.X_history, self.D_history, self.L_history, self.Y_history = self.X_design, self.D_design, self.L_design, self.Y_design    # Initialize history of data
        self.X_best_design, self.D_best_design, self.L_bdest_design, self.Y_best_design = self.get_best_values(self.X_design, self.D_design, self.L_design, self.Y_design)
        self.X_best, self.D_best, self.Y_best = self.X_best_design, self.D_best_design, self.Y_best_design  # Initialize current best value
        self.X_best_history, self.D_best_history, self.Y_best_history = self.X_best_design, self.D_best_design, self.Y_best_design # Initialize history of best values

        print(f'''
        {'-'*50}
                        Optimization Cycle
        {'-'*50}
        ''')
        for step in range(num_iteration):
            mixed_problem = GPyOpt.methods.BayesianOptimization(
                f = None, 
                domain = self.variables,
                constraints = None,
                cost_withGradients = None,
                model_type = 'GP',
                X = np.asarray(self.X_history),
                Y = np.asarray(self.Y_history),
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
            # For unsatisfactory points, skip evaluation and compute penalized objective value
            self.X = np.asarray(mixed_problem.suggest_next_locations())
            all_valid, invalid_configurations, problems_log, penalized_values = self.pre_check_constraints(self.X)
            self.X_invalid = np.vstack((self.X_invalid, invalid_configurations))
            self.invalid_constraints_log = np.vstack((self.invalid_constraints_log, problems_log))
            print(f'{len(invalid_configurations)} potentially invalid configurations. Penalized objective values are calculated for these (skip CFD).')
            self.D, self.L, self.Y = self.process_actual_results(self.X)

            # Recheck constraints after obtaining lift and drag in simulations
            # For unsatifactory configurations, significantly penalize the objective value (not as severe as pre-check since base is nonzero from simulation results)
            all_valid, invalid_configurations, problems_log, self.Y = self.post_check_constraints(self.X, self.D, self.L, self.Y)
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