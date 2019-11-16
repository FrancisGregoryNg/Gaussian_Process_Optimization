import chaospy
import GPyOpt
import math
import matplotlib
import numpy as np
import pathlib
import time
import xlwings as xw
np.set_printoptions(linewidth=200, precision=4)

class SLT_Optimization():
    Datafolder = pathlib.Path(__file__).parents[2].joinpath('2---Data')
    def __init__(self, target_velocity, payload_weight=500, motor_angle = 5, plane_index=0, beam_index = 0, plane_battery_cell = [3], beam_length_max=80):
        '''
        S = simulated values
                0 - 'lift'                      - intermediate between X and Y
                1 - 'drag'                      - used for checking actual flight capability

        E = estimated values
                0 - 'pitch'                     - estimated angle of attack based on plane data to accommodate weight
                1 - 'weight'                    - calculated weight of configuration                

        X = input variables 
                0 - 'motor'                     - motor selection
                1 - 'propeller'                 - propeller selection
                2 - 'battery'                   - battery selection
                3 - 'distanceFromCenterline'    - connection point from the center of the fuselage
                4 - 'beam_length'               - length of beam
                  
        Y = output variables
                0 - endurance_estimate          - calculated value for the endurance
                1 - objective_value             - endurance estimate and/or penalty for constraints violations
        '''
        print(f'''
        {'-'*50}
                        Initialization
        {'-'*50}
        ''')
        time.sleep(1.0)
        # -----Experimental Design---------------------------
        self.S_design = None
        self.E_design = None
        self.X_design = None
        self.Y_design = None
        self.constraints_log_design = []

        # -----Best from Experimental Design-----------------
        self.S_best_design = None
        self.E_best_design = None
        self.X_best_design = None
        self.Y_best_design = None

        # -----Current Iteration-----------------------------
        self.S = None
        self.E = None
        self.X = None
        self.Y = None
        self.constraints_log = []

        # -----Best up to the Current Iteration--------------
        self.S_best = None
        self.E_best = None
        self.X_best = None
        self.Y_best = None

        # -----History of Data-------------------------------
        self.S_history = None
        self.E_history = None
        self.X_history = None
        self.Y_history = None
        self.constraints_log_history = []

        # -----Historical Progression of Best Data-----------
        self.S_best_history = None
        self.E_best_history = None
        self.X_best_history = None
        self.Y_best_history = None

        # -----Others----------------------------------------
        self.processed_points = 0
        self.target_velocity = target_velocity   # km/hr input value
        self.payload_weight = payload_weight    # weight of payload in grams
        self.motor_angle = motor_angle  # angle of plane motor with respect to a flat wing (0° pitch)
        self.plane = plane_index    # choice of plane is not an optimization variable (default is 0)
        self.beam = beam_index  # choice of beam is not an optimization variable (default is 0)
        self.plane_battery_cell = plane_battery_cell
        self.plane_data, self.beam_data, self.quad_battery_data, self.plane_battery_data, self.motor_data, self.propeller_data = self.load_component_info()
        self.variables, self.X_invalid = self.define_variables(beam_length_max)
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
            battery = [quad_battery_data.index(battery) for battery in quad_battery_data if int(round(motor['cell_min'])) <= int(round(battery['cell'])) <= int(round(motor['cell_max']))]
            motor['battery'] = battery
        data = [plane_data, beam_data, quad_battery_data, plane_battery_data, motor_data, propeller_data]
        labels = ['Plane', 'Beam', 'Quadcopter Batteries', 'Plane Batteries', 'Motor', 'Propeller']
        for index in range(len(data)):
            print(f'    [{labels[index]}]')
            for item in data[index]:
                print(item)
                time.sleep(0.1)
            print()
        componentData.close()
        return plane_data, beam_data, quad_battery_data, plane_battery_data, motor_data, propeller_data

    def define_variables(self, beam_length_max=None):
        plane = self.plane_data[self.plane]
        if beam_length_max is None:
            beam_length_max = 100
        beam_length_min = min([propeller['diameter'] * 2.54 for propeller in self.propeller_data])
        beam_length_range = (beam_length_min, beam_length_max)
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
        X_invalid = []
        print('    [Variables]')
        for variable in variables:
            print(variable)
            time.sleep(0.1)
        return variables, X_invalid

    def get_max_weight(self):
        pitchFile = str(self.Datafolder) + '/CFD/Angle_of_Attack.xlsx'
        pitchData = xw.Book(str(pathlib.PureWindowsPath(pitchFile)))
        max_weight = float(pitchData.sheets['Interface'].range('B2').value)
        return max_weight

    def estimate_values(self, X=None):
        print('Preparing pitch estimates: ', end='')
        pitchFile = str(self.Datafolder) + '/CFD/Angle_of_Attack.xlsx'
        pitchData = xw.Book(str(pathlib.PureWindowsPath(pitchFile)))
        max_weight = float(pitchData.sheets['Interface'].range('B2').value)
        pitchData.sheets['Interface'].range('B1').value = max_weight
        max_pitch = float(pitchData.sheets['Interface'].range('B3').value)
        if X is None:
            X = self.X
        plane = self.plane_data[self.plane] 
        beam = self.beam_data[self.beam]
        pitch = []
        weight = []
        for configuration in range(len(X)):
            if configuration in [int(round((step+1)*len(X)/10)) for step in range(10)]:
                print('█', end='')
            motor = self.motor_data[int(round(X[configuration][0]))]
            propeller = self.propeller_data[int(round(X[configuration][1]))]
            quad_battery = self.quad_battery_data[int(round(X[configuration][2]))]
            plane_battery = self.plane_battery_data[int(round(X[configuration][3]))]
            beam_length = X[configuration][5]
            W = (plane['weight'] 
                + beam_length * beam['weight_per_L']
                + 4 * propeller['weight']
                + 4 * motor['weight']
                + quad_battery['weight']
                + plane_battery['weight']
                + self.payload_weight) * 0.0098      # convert from grams to Newtons
            if W < max_weight:
                pitchData.sheets['Interface'].range('B1').value = W
                estimate = float(pitchData.sheets['Interface'].range('B3').value)
            else:
                estimate = max_pitch
            pitch.append(estimate)
            weight.append(W)
        print()
        pitchData.close()
        estimates = [[p, w] for p,w in zip(pitch, weight)]     
        return estimates

    def pre_check_constraints(self, X=None):
        if X is None:
            X = self.X
        plane = self.plane_data[self.plane]
        invalidConfigurations_indices = []
        constraints_log = [['Passed','Passed'] for configuration in range(len(X))]
        penalized_values = [0 for configuration in range(len(X))]
        estimates = self.estimate_values(X)    
        pitch_estimates, weight = [v[0] for v in estimates], [v[1] for v in estimates]
        max_weight = self.get_max_weight()
        print('\nConducting preliminary constraint checks...\n')

        def add_log(configuration, problem):
            if constraints_log[configuration][0] == 'Passed':
                constraints_log[configuration][0] = problem
            else:
                constraints_log[configuration][0] += problem
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
            motor_index = int(round(X[configuration][0]))
            propeller_index = int(round(X[configuration][1]))
            quad_battery_index = int(round(X[configuration][2]))
            plane_battery_index = int(round(X[configuration][3]))
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
                if distanceFromCenterline - (propeller['diameter'] * 2.54 / 2) < plane['connection_min']:
                    minimumFuselageDistance = plane['connection_min'] + (propeller['diameter'] * 2.54 / 2) 
                    distanceCorrection = (distanceFromCenterline - minimumFuselageDistance) / minimumFuselageDistance
                    add_log(configuration, f'Propellers clash with fuselage ({distanceCorrection*100:.2f}% difference from minimum fuselage distance). ')
                    add_penalty(configuration, distanceCorrection)
                if beam_length < (propeller['diameter'] * 2.54):
                    lengthCorrection = (beam_length - (propeller['diameter'] * 2.54)) / (propeller['diameter'] * 2.54)
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
                            + 'M' + '{:02}'.format(int(round(motor_index+1))) + '_' 
                            + 'P' + '{:02}'.format(int(round(propeller_index+1))) + '_'
                            + 'B' + '{:02}'.format(int(round(battery_cellCount))) + '.xlsx')
                    testData = xw.Book(str(pathlib.PureWindowsPath(testFile)))
                    thrust_max = float(testData.sheets['Interface'].range('B2').value)
                    # weight restriction
                    planeFile = (str(self.Datafolder) + '/MBO/RCbenchmark/plane.xlsx')
                    planeData = xw.Book(str(pathlib.PureWindowsPath(planeFile)))
                    plane_thrust_max = float(planeData.sheets['Interface'].range('B2').value)
                    vertical_thrust_component =  math.sin(math.radians(pitch - self.motor_angle))
                    if weight[configuration] > max_weight + plane_thrust_max * vertical_thrust_component:
                        planeLiftCorrection = (max_weight - weight[configuration]) / weight[configuration]
                        add_log(configuration, f'Insufficient estimated plane lift ({planeLiftCorrection*100:.2f}% above total weight). ')
                        add_penalty(configuration, planeLiftCorrection)
                    else:
                        pitchFile = str(self.Datafolder) + '/CFD/Angle_of_Attack.xlsx'
                        pitchData = xw.Book(str(pathlib.PureWindowsPath(pitchFile)))
                        pitchData.sheets['Interface'].range('B1').value = weight[configuration]
                        drag = float(pitchData.sheets['Interface'].range('B4').value)
                        horizontal_thrust_component =  math.cos(math.radians(pitch - self.motor_angle))
                        if plane_thrust_max * horizontal_thrust_component < drag:
                            planeThrustCorrection = (plane_thrust_max - drag) / drag
                            add_log(configuration, f'Insuffucient plane thrust ({planeThrustCorrection*100:.2f}% difference from estimated drag). ')
                            add_penalty(configuration, planeThrustCorrection)
                        else:
                            planeData.sheets['Interface'].range('B1').value = drag      
                            current_draw = float(planeData.sheets['Interface'].range('B3').value)
                            if current_draw > (plane_battery['mah'] / 1000) * plane_battery['discharge']:
                                currentLimit = (plane_battery['mah'] / 1000) * plane_battery['discharge']
                                currentCorrection = (currentLimit - current_draw) / currentLimit
                                add_log(configuration, f'Too much estimated plane current draw ({-currentCorrection*100:.2f}% difference from battery limit). ')
                                add_penalty(configuration, currentCorrection)
                    if weight[configuration] > 4 * thrust_max:
                        quadcopterThrustCorrection = (4 * thrust_max - weight[configuration]) / weight[configuration]
                        add_log(configuration, f'Insufficient quadcopter thrust ({quadcopterThrustCorrection*100:.2f}% difference from total weight). ')
                        add_penalty(configuration, quadcopterThrustCorrection)
                    else:
                        testData.sheets['Interface'].range('B1').value = weight[configuration]/4
                        current_draw = float(testData.sheets['Interface'].range('B3').value)
                        if current_draw > (quad_battery['mah'] / 1000) * quad_battery['discharge']:
                            currentLimit = (quad_battery['mah'] / 1000) * quad_battery['discharge']
                            currentCorrection = (currentLimit - current_draw) / currentLimit
                            add_log(configuration, f'Too much quadcopter current draw ({-currentCorrection*100:.2f}% difference from battery limit). ')
                            add_penalty(configuration, currentCorrection) 
                    testData.close()
                    pitchData.close()
                    planeData.close()
            if constraints_log[configuration][0] != 'Passed': 
                print(f'  └──> {constraints_log[configuration][0]}\n')
        invalid_configurations = [X[index] for index in invalidConfigurations_indices]
        if not invalid_configurations: #check if no configurations are invalid (empty list)
            print('All configurations pass the pre-simulation constraint checks.\n\n')
        else:
            print(f'\n{len(invalid_configurations)} potentially invalid configurations. ' +
                  'Penalized objective values are calculated for these based on percentage violation of constraints (skip CFD, penalty on top of zero endurance).\n\n')
        return constraints_log, penalized_values, estimates

    def calculate_endurance(self, S=None, X=None, penalized_values=None):
        if X is None and S is None:
            X = self.X
            S = self.S
            penalized_values = [0 for _ in self.X]
        D = [v[0] for v in S]
        planeFile = str(self.Datafolder) + '/MBO/RCbenchmark/plane.xlsx' 
        planeData = xw.Book(str(pathlib.PureWindowsPath(planeFile)))
        Y = [['None', 'None'] for _ in range(len(X))]
        for configuration in range(len(X)):
            if D[configuration] == 'None':
                Y[configuration][1] = penalized_values[configuration]
            else:
                plane_battery = self.plane_battery_data[int(round(X[configuration][3]))]
                planeData.sheets['Interface'].range('B1').value = D[configuration]
                eff = float(planeData.sheets['Interface'].range('B5').value) # motor efficiency varies based on thrust to overcome drag
                if plane_battery['battery_hour_rating'] == 'X':
                    R = 1      # battery hour rating is typically equal to 1hr
                else:
                    R = plane_battery['battery_hour_rating']
                n = 1.3     # Peukert exponent is typically equal to 1.3 for LiPo batteries
                V = plane_battery['cell'] * 3.7     # 3.7 volts in each LiPo cell
                C = plane_battery['mah'] / 1000       # convert from mAh to Ah
                U = self.target_velocity * (1000/3600)      # get the flight velocity in m/s from km/hr input value
                E = R**(1-n) * ((eff * V * C)/(D[configuration] * U)) ** n
                Y[configuration] = [E, E]
        planeData.close()
        return Y

    def process_actual_results(self, E=None, X=None, penalized_values=None, S=None):
        evaluationsFile = str(self.Datafolder) + '/MBO/evaluations.xls'
        evaluationsData = xw.Book(str(pathlib.PureWindowsPath(evaluationsFile)))
        print('Setting up evaluations...\n')
        if X is None:
            X = self.X
            E = self.E
            penalized_values = [0 for _ in self.X]
        batch_size = len(X)
        write_values = [[None for _ in range(10)] for configuration in range(batch_size)]

        for configuration in range(batch_size):
            if penalized_values[configuration] != 0:
                write_values[configuration][0:2] = ['None', 'None']
            elif S is not None:
                if S[configuration] != [0, 0]:
                    write_values[configuration][0:2] = S[configuration]
            write_values[configuration][2:4] = E[configuration]
            write_values[configuration][4:8] = [int(round(X[configuration][value]+1)) for value in range(4)] #convert to 1-index for user
            write_values[configuration][8:10] = [float(X[configuration][value]) for value in [4, 5]] 
        evaluationsData.sheets['current'].range(f'A{2}:J{batch_size+2}').value = write_values
        evaluationsData.save()
        print('\n    [ USER INSTRUCTION ]:')
        input(f'Please fill up the output columns (orange) in "{str(pathlib.PureWindowsPath(evaluationsFile))}".\nKey in "Enter" when done.')
        print('Processing results...\n')
        D = evaluationsData.sheets['current'].range(f'A2:A{batch_size+1}').value
        L = evaluationsData.sheets['current'].range(f'B2:B{batch_size+1}').value
        S = [[drag, lift] for drag, lift in zip(D,L)]
        if all([value is not None for value in D + L]): #check if drag and lift values are all filled
            Y = self.calculate_endurance(S, X, penalized_values)
            evaluationsData.sheets['current'].range(f'K2:L{batch_size+1}').value = [endurance_objective for endurance_objective in Y]
            evaluationsData.save()   
        else:
            print(f'\nInvalid entries. Try again...')
            S, Y = self.process_actual_results(X, penalized_values)
        evaluationsData.save()
        return S, Y

    def post_check_constraints(self, S=None, E=None, X=None, Y=None, constraints_log=None):
        print('Conducting post-simulation constraint checks... (not redundant with preliminary checks)\n')
        if X is None:
            S = self.S
            E = self.E
            X = self.X
            Y = self.Y
            constraints_log = self.constraints_log
        D, L = [v[0] for v in S], [v[1] for v in S]
        invalidConfigurations_indices = []
        penalized_values = [0 for configuration in range(len(X))]
        pitch_estimates, weight = [v[0] for v in E], [v[1] for v in E]

        def add_log(configuration, problem):
            if constraints_log[configuration][1] == 'Passed':
                constraints_log[configuration][1] = problem
            else:
                constraints_log[configuration][1] += problem
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
                plane_battery_index = int(round(X[configuration][3]))
                pitch = float(pitch_estimates[configuration]) 
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
                vertical_thrust_component =  math.sin(math.radians(pitch - self.motor_angle))
                if weight[configuration] < L[configuration] + plane_thrust_max * vertical_thrust_component:
                    liftCorrection = (weight[configuration] - L[configuration]) / L[configuration]
                    add_log(configuration, f'Insufficient simulated plane lift ({liftCorrection*100:.2f}% difference from total weight). ')
                    add_penalty(configuration, liftCorrection)
                else:
                    drag = D[configuration]
                    horizontal_thrust_component =  math.cos(math.radians(pitch - self.motor_angle))
                    if plane_thrust_max * horizontal_thrust_component < drag:
                        planeThrustCorrection = (plane_thrust_max - drag) / drag
                        add_log(configuration, f'Insuffucient plane thrust ({planeThrustCorrection*100:.2f}% difference from simulated drag). ')
                        add_penalty(configuration, planeThrustCorrection)
                    else:
                        planeData.sheets['Interface'].range('B1').value = D[configuration]       
                        current_draw = float(planeData.sheets['Interface'].range('B3').value)
                        if current_draw > (plane_battery['mah'] / 1000) * plane_battery['discharge']:
                            currentLimit = (plane_battery['mah'] / 1000) * plane_battery['discharge']
                            currentCorrection = (currentLimit - current_draw) / currentLimit
                            add_log(configuration, f'Too much plane current draw ({-currentCorrection*100:.2f}% difference from battery limit). ')
                            add_penalty(configuration, currentCorrection)
                planeData.close()
            if constraints_log[configuration][1] != 'Passed': 
                print(f'  └──> {constraints_log[configuration][1]}\n')
        invalid_configurations = [X[index] for index in invalidConfigurations_indices]
        if not invalid_configurations: #check if no configurations are invalid (empty list)
            print('\nAll evaluated configurations pass the post-simulation constraint checks.\n\n')
        else:
            print(f'\n{len(invalid_configurations)} evaluated configurations failed post-simulation constraint checks; results are penalized.\n\n')
        endurance = [value[0] for value in Y]
        objective_values = [value[1] for value in Y]
        penalized_objective = [objective + penalty for objective, penalty in zip(objective_values, penalized_values)]
        Y = [[endurance, objective] for endurance, objective in zip(endurance, penalized_objective)]
        evaluationsFile = str(self.Datafolder) + '/MBO/evaluations.xls'
        evaluationsData = xw.Book(str(pathlib.PureWindowsPath(evaluationsFile)))
        evaluationsData.sheets['current'].range(f'K2:L{len(X)+1}').value = Y
        #batch_size = len(X)
        #for configuration in range(batch_size):                
        #    evaluationsData.sheets['current'].range(f'K2:L{batch_size+1}').value = [endurance_objective for endurance_objective in Y]
        evaluationsData.save()   
        return constraints_log, Y

    def adjust_pitch(self, S=None, E=None):
        pitchFile = str(self.Datafolder) + '/CFD/Angle_of_Attack.xlsx'
        pitchData = xw.Book(str(pathlib.PureWindowsPath(pitchFile)))
        print(f'Determining pitch adjustments (simulated lift must be within 5% of the weight)...\n')
        if S is None:
            S = self.S
            E = self.E
        old_S = S.copy()   
        old_E = E.copy()       
        above_below = ['above', 'below']
        factor_shrink_rate = 3.0        # affects how rapidly the adjustment factor appraoches 1 (can be any positive number)
        min_adjustment_factor = 0.1     # minimum value for the adjustment factor (limit adjustment when difference is high)
        max_adjustment_factor = 1.0   
        max_pitch_adjustment = 5.0      # limit the change in pitch
        min_pitch = -5.0                # constrain pitch to a certain interval
        max_pitch = 40.0
        adjusted_configurations = []
        for configuration in range(len(S)):
            if S[configuration][1] != 'None':
                lift = float(S[configuration][1])
                weight = float(E[configuration][1])
                percent_difference = float(100 * (lift - weight) / weight)
                if abs(percent_difference) > 5:
                    print(f'#{configuration+1:2d}: !!! Previously simulated lift is {abs(percent_difference):5.2f}% {above_below[int(percent_difference<0)]} the calculated weight.')
                    S[configuration] = [0, 0]
                    old_pitch = E[configuration][0]
                    pitchData.sheets['Interface'].range('B1').value = lift
                    expected_pitch_for_lift = float(pitchData.sheets['Interface'].range('B3').value)
                    pitch_adjustment = expected_pitch_for_lift - old_pitch
                    shrink_curve = 1 - ((abs(percent_difference) ** factor_shrink_rate) - (5  ** factor_shrink_rate)) / (abs(percent_difference) ** factor_shrink_rate)
                    adjustment_factor = min_adjustment_factor + (max_adjustment_factor - min_adjustment_factor) * shrink_curve  
                    proposed_pitch = float(old_pitch - pitch_adjustment * adjustment_factor)
                    if abs(proposed_pitch - old_pitch) > max_pitch_adjustment:
                        proposed_pitch = float(old_pitch - 3 * pitch_adjustment / pitch_adjustment)
                    if proposed_pitch < min_pitch:
                        proposed_pitch = min_pitch
                    if proposed_pitch > max_pitch:
                        proposed_pitch = max_pitch
                    E[configuration][0] = proposed_pitch
                    adjusted_configurations.append(configuration)
                    print(f'  └──> {old_pitch:5.2f} ──> {E[configuration][0]:5.2f}\n')
                else:
                    print(f'#{configuration+1:2d}: OK. Previously simulated lift is {abs(percent_difference):5.2f}% {above_below[int(percent_difference<0)]} the calculated weight.')   
        evaluationsFile = str(self.Datafolder) + '/MBO/evaluations.xls'
        evaluationsData = xw.Book(str(pathlib.PureWindowsPath(evaluationsFile)))               
        evaluationsData.sheets['current'].range(f'C2:D{len(S)+2}').value = E
        evaluationsData.save()   
        print(f'The pitch of {len(adjusted_configurations)} configurations have been adjusted.')
        return S, E, old_S, old_E

    def reconcile_pitch(self, S, E, old_S, old_E):
        if S is None:
            S = self.S
            E = self.E
        for configuration in range(len(S)):
            pitch = float(E[configuration][0])
            old_pitch = float(old_E[configuration][0])
            if pitch != old_pitch:
                lift = float(S[configuration][1])
                weight = float(E[configuration][1])
                percent_difference = float(100 * (lift - weight) / weight)
                if abs(percent_difference) > 5:
                    print(f'#{configuration+1:2d}: !!! New simulated lift is {abs(percent_difference):5.2f}% {above_below[int(percent_difference<0)]} the calculated weight. Interpolating...')
                    drag = float(S[configuration][0])
                    old_drag = float(old_S[configuration][0])
                    old_lift = float(old_S[configuration][1])
                    S[configuration][1] = weight 
                    S[configuration][0] = old_drag + (drag - old_drag) * (weight - old_lift) / (lift - old_lift)
                    E[configuration][0] = old_pitch + (pitch - old_pitch) * (weight - old_lift) / (lift - old_lift)
                    print(f'  ├──> Drag: {old_drag:5.2f} & {drag:5.2f}  ──> {S[configuration][0]:5.2f}\n')
                    print(f'  ├──> Lift: {old_lift:5.2f} & {lift:5.2f}  ──> {S[configuration][1]:5.2f}\n')
                    print(f'  └──> Pitch: {old_pitch:5.2f} & {pitch:5.2f}  ──> {E[configuration][0]:5.2f}\n')
                else:
                    print(f'#{configuration+1:2d}: OK. New simulated lift is {abs(percent_difference):5.2f}% {above_below[int(percent_difference<0)]} the calculated weight.')   

        return S, E

    def get_best_values(self, S=None, E=None, X=None, Y=None):
        if S is None and E is None and X is None and Y is None:
            S = self.S_history
            E = self.E_history
            X = self.X_history
            Y = self.Y_history
        index_best = np.argmax(np.asarray(Y))
        S_best = S[index_best]
        E_best = E[index_best]
        X_best = X[index_best]
        Y_best = Y[index_best]
        return S_best, E_best, X_best, Y_best

    def update(self, array_old, array_add):
        [array_old.append(item) for item in array_add]
        return array_old

    def print_internal_variables(self):
        variables = [
            ('self.S_design', self.S_design),
            ('self.E_design', self.E_design),
            ('self.X_design', self.X_design),
            ('self.Y_design', self.Y_design),
            ('self.constraints_log_design', self.constraints_log_design),
            ('self.S_best_design', self.S_best_design),
            ('self.E_best_design', self.E_best_design),
            ('self.X_best_design', self.X_best_design),
            ('self.Y_best_design', self.Y_best_design),
            ('self.S', self.S),
            ('self.E', self.E),
            ('self.X', self.X),
            ('self.Y', self.Y),
            ('self.constraints_log', self.constraints_log),
            ('self.S_best', self.S_best),
            ('self.E_best', self.E_best),
            ('self.X_best', self.X_best),
            ('self.Y_best', self.Y_best),
            ('self.S_history', self.S_history),
            ('self.E_history', self.E_history),
            ('self.X_history', self.X_history),
            ('self.Y_history', self.Y_history),
            ('self.constraints_log_history', self.constraints_log_history),
            ('self.S_best_history', self.S_best_history),
            ('self.E_best_history', self.E_best_history),
            ('self.X_best_history', self.X_best_history),
            ('self.Y_best_history', self.Y_best_history),
        ]

        for label, value in variables:
            print(f'    {label}')
            if ('best' not in label or 'history' in label) and value is not None:
                for entry in value:
                    print(entry)
                    time.sleep(0.02)
            else:
                print(value)
                time.sleep(0.02)
            print()
        return None

    def tabulate_data(self, S=None, E=None, X=None, Y=None):
        if X is None:
            S = self.S
            E = self.E
            X = self.X
            Y = self.Y
        print('┌────────┬──────────┬──────────┬──────────┬──────────┬───────┬───────────┬──────────────┬───────────────┬─────────────────────┬─────────────┬───────────┬───────────┐')
        print('│ Design │   Drag   │   Lift   │   Pitch  │  Weight  │ Motor │ Propeller │ Quad battery │ Plane battery │ Centerline distance │ Beam length │ Endurance │ Objective │')
        print('├────────┼──────────┼──────────┼──────────┼──────────┼───────┼───────────┼──────────────┼───────────────┼─────────────────────┼─────────────┼───────────┼───────────┤')
        for index, point in enumerate(X, start=1):
            if S[index-1][0] != 'None':
                print(f'│  {index:3d}   '
                    + f'│ {S[index-1][0]:7.2f}  '
                    + f'│ {S[index-1][1]:7.2f}  '
                    + f'│ {E[index-1][0]:7.2f}  '
                    + f'│ {E[index-1][1]:7.2f}  '
                    + f'│ {point[0]+1:3d}   '
                    + f'│ {point[1]+1:4d}      '
                    + f'│    {point[2]+1:5d}     '
                    + f'│    {point[3]+1:5d}      '
                    + f'│      {point[4]:8.2f}       '
                    + f'│  {point[5]:8.2f}   '
                    + f'│ {Y[index-1][0]:7.2f}   '
                    + f'│ {Y[index-1][1]:7.2f}   │') 
            else:
                print(f'│  {index:3d}   '
                    + f'│    None  '
                    + f'│    None  '
                    + f'│ {E[index-1][0]:7.2f}  '
                    + f'│ {E[index-1][1]:7.2f}  '
                    + f'│ {point[0]+1:3d}   '
                    + f'│ {point[1]+1:4d}      '
                    + f'│    {point[2]+1:5d}     '
                    + f'│    {point[3]+1:5d}      '
                    + f'│      {point[4]:8.2f}       '
                    + f'│  {point[5]:8.2f}   '
                    + f'│    None   '
                    + f'│ {Y[index-1][1]:7.2f}   │')
            time.sleep(0.1)
        print('└────────┴──────────┴──────────┴──────────┴──────────┴───────┴───────────┴──────────────┴───────────────┴─────────────────────┴─────────────┴───────────┴───────────┘\n')
        return None

    def write_data_to_spreadsheet(self, initial_batch=False):

        def update_sheet(sheet_name):
            if sheet_name == 'design':
                S = self.S_design
                E = self.E_design
                X = self.X_design
                Y = self.Y_design
                log = self.self.constraints_log_design
            elif sheet_name == 'best_design':
                S = self.S_best_design
                E = self.E_best_design
                X = self.X_best_design
                Y = self.Y_best_design
                log = None
            elif sheet_name == 'current':
                S = self.S
                E = self.E
                X = self.X
                Y = self.Y
                log = self.self.constraints_log
            elif sheet_name == 'current_best':
                S = self.S_best
                E = self.E_best
                X = self.X_best
                Y = self.Y_best
                log = None
            elif sheet_name == 'all':
                S = self.S_history
                E = self.E_history
                X = self.X_history
                Y = self.Y_history
                log = self.self.constraints_log_history
            elif sheet_name == 'all_best':
                S = self.S_best_history
                E = self.E_best_history
                X = self.X_best_history
                Y = self.Y_best_history
                log = None
            evaluationsFile = str(self.Datafolder) + '/MBO/evaluations.xls'
            evaluationsData = xw.Book(str(pathlib.PureWindowsPath(evaluationsFile)))
            active_rows = evaluationsData.sheets[sheet_name].range('A1').end('down').row
            evaluationsData.sheets[sheet_name].range(f'A2:K{active_rows}').clear_contents()
            if 'best' not in sheet_name or 'all_best' in sheet_name:
                for entry in range(len(X)):
                    row = entry + 2
                    evaluationsData.sheets[sheet_name].range(f'A{row}:B{row}').value = S[entry]
                    evaluationsData.sheets[sheet_name].range(f'C{row}:D{row}').value = E[entry]
                    evaluationsData.sheets[sheet_name].range(f'E{row}:H{row}').value = [int(round(X[entry][value]+1)) for value in range(4)] #convert to 1-index for user
                    evaluationsData.sheets[sheet_name].range(f'I{row}:J{row}').value = [float(X[entry][value]) for value in [4, 5]]
                    evaluationsData.sheets[sheet_name].range(f'K{row}:L{row}').value = Y[entry]
                    if log is not None:
                        evaluationsData.sheets[sheet_name].range(f'M{row}:N{row}').value = log[entry]
                evaluationsData.save()   
            elif X is not None:
                evaluationsData.sheets[sheet_name].range('A2:B2').value = S
                evaluationsData.sheets[sheet_name].range('C2:D2').value = E
                evaluationsData.sheets[sheet_name].range('E2:H2').value = [int(round(X[value]+1)) for value in range(4)] #convert to 1-index for user
                evaluationsData.sheets[sheet_name].range('I2:J2').value = [float(X[value]) for value in [4, 5]]
                evaluationsData.sheets[sheet_name].range('K2:L2').value = Y
                evaluationsData.save()   
            else:
                pass

        if initial_batch:
            update_sheet('design')
            update_sheet('best_design')
            update_sheet('current_best')
        else:
            update_sheet('current')
            update_sheet('current_best')
        update_sheet('all')
        update_sheet('all_best')
        return None

    def load_previous_data_from_spreadsheet(self):

        def load_sheet(sheet_name):
            evaluationsFile = str(self.Datafolder) + '/MBO/evaluations.xls'
            evaluationsData = xw.Book(str(pathlib.PureWindowsPath(evaluationsFile)))  
            active_rows = evaluationsData.sheets[sheet_name].range('A1').end('down').row
            S = [evaluationsData.sheets[sheet_name].range(f'A{row}:B{row}').value for row in range(2, active_rows+1)]
            E = [evaluationsData.sheets[sheet_name].range(f'C{row}:D{row}').value for row in range(2, active_rows+1)]
            X = [[int(value) for value in evaluationsData.sheets[sheet_name].range(f'E{row}:H{row}').value]
                 + evaluationsData.sheets[sheet_name].range(f'I{row}:J{row}').value for row in range(2, active_rows+1)]
            Y = [evaluationsData.sheets[sheet_name].range(f'K{row}:L{row}').value for row in range(2, active_rows+1)]
            log = [evaluationsData.sheets[sheet_name].range(f'M{row}:N{row}').value for row in range(2, active_rows+1)]
            if sheet_name == 'design':
                self.S_design = S
                self.E_design = E
                self.X_design = X
                self.Y_design = Y
                self.self.constraints_log_design = log
            elif sheet_name == 'best_design':
                self.S_best_design = S
                self.E_best_design = E
                self.X_best_design = X
                self.Y_best_design = Y
            elif sheet_name == 'current':
                self.S = S
                self.E = E
                self.X = X
                self.Y = Y
                self.self.constraints_log = log
            elif sheet_name == 'current_best':
                self.S_best = S
                self.E_best = E
                self.X_best = X
                self.Y_best = Y
            elif sheet_name == 'all':
                self.S_history = S
                self.E_history = E
                self.X_history = X
                self.Y_history = Y
                self.self.constraints_log_history = log
            elif sheet_name == 'all_best':
                self.S_best_history = S
                self.E_best_history = E
                self.X_best_history = X
                self.Y_best_history = Y

        load_sheet('design')
        load_sheet('best_design')
        load_sheet('current_best')
        load_sheet('current')
        load_sheet('current_best')
        load_sheet('all')
        load_sheet('all_best')
        return None

    def plot_convergence(self):
        X = [x for x in range(1, len(self.Y_best)+1)]
        Y = [y*60 for y in self.Y_best]
        fig = matplotlib.pyplot.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        title = 'Convergence Plot'
        ax.set_title(title, fontweight = 550, fontsize = 'large')
        ax.plot(X, Y, 'b', marker='o', s=50) 
        ax.set_xlabel('Batch Iteration')
        ax.set_ylabel('Endurance Estimate (minutes)')
        return None

    def generate_experimental_design(self, num_design):
        print('Generating experimental design...\n')          
        hammerseley = chaospy.distributions.sampler.sequences.hammersley
        base = hammerseley.create_hammersley_samples(num_design, dim=len(self.variables), burnin=-1, primes=()) #numpy array
        motor_selections = np.rint(base[0, :] * (len(self.motor_data)-1)).astype(int).tolist()
        propeller_selections_subindex = np.rint(base[1, :] * np.asarray([len(self.motor_data[motor]['propeller'])-1 for motor in motor_selections])).astype(int).tolist()
        propeller_selections = [self.motor_data[motor]['propeller'][index] for (motor, index) in list(zip(motor_selections, propeller_selections_subindex))]
        quad_battery_selections_subindex = np.rint(base[2, :] * np.asarray([len(self.motor_data[motor]['battery'])-1 for motor in motor_selections])).astype(int).tolist()
        quad_battery_selections = [self.motor_data[motor]['battery'][index] for (motor, index) in list(zip(motor_selections, quad_battery_selections_subindex))]
        plane_battery_selections = np.rint(base[3, :] * (len(self.plane_battery_data)-1)).astype(int).tolist()
        distanceFromCenterline_values = (base[4, :] * self.variables[4]['domain'][1]).tolist()
        beam_length_values = (base[5, :] * self.variables[5]['domain'][1]).tolist()
        design = [[motor_selections[design], propeller_selections[design], quad_battery_selections[design], plane_battery_selections[design],
                distanceFromCenterline_values[design], beam_length_values[design]] for design in range(num_design)]
        
        def _check_if_duplicate(design, index, entry=None):
                duplicate = False
                if entry is None:
                    entry = design[index]
                    for previous_index in range(index):
                        if entry[0:4] == design[previous_index][0:4]:
                            duplicate = True
                            break
                else:
                    for all_index in range(len(design)):
                        if entry[0:4] == design[all_index][0:4]:
                            duplicate = True
                            break
                return duplicate

        duplicates = [index for index in range(len(design)) if _check_if_duplicate(design, index)]
        for index in duplicates:
            entry = design[index]
            attempt=0
            while _check_if_duplicate(design, index, entry) and attempt<100:
                attempt+=1
                motor_selection = np.random.choice(range(len(self.motor_data)-1))
                propeller_selection = np.random.choice(self.motor_data[motor_selection]['propeller'])
                quad_battery_selection = np.random.choice(self.motor_data[motor_selection]['battery'])
                plane_battery_selection = np.random.choice(range(len(self.plane_battery_data)))
                entry = design[index][:]
                entry[0:4] = [motor_selection, propeller_selection, quad_battery_selection, plane_battery_selection]            
            design[index][0:4] = entry[0:4]
        print('┌────────┬───────┬───────────┬──────────────┬───────────────┬─────────────────────┬─────────────┐')
        print('│ Design │ Motor │ Propeller │ Quad battery │ Plane battery │ Centerline distance │ Beam length │')
        print('├────────┼───────┼───────────┼──────────────┼───────────────┼─────────────────────┼─────────────┤')
        for index, point in enumerate(design, start=1):
            print(f'│  {index:3d}   '
                + f'│ {point[0]+1:3d}   '
                + f'│ {point[1]+1:4d}      '
                + f'│    {point[2]+1:5d}     '
                + f'│    {point[3]+1:5d}      '
                + f'│      {point[4]:8.2f}       '
                + f'│  {point[5]:8.2f}   │')
            time.sleep(0.1)
        print('└────────┴───────┴───────────┴──────────────┴───────────────┴─────────────────────┴─────────────┘\n')
        for index in duplicates:
            print(f'Design#{index:2d} is randomized to avoid duplicate categorical variable settings.') 
        return design

    def initial_batch(self, num_design = 10, adjust_pitch=False):
        print(f'\n\n\nNote:\n The target velocity is {float(self.target_velocity):.2f}kph. Meanwhile, pitch estimates are made using data at 45kph.\n\n')
        print(f'''
        {'-'*50}
                        Experimental Design
        {'-'*50}
        ''')
        time.sleep(1.0)
        # Precheck constraints prior to suggesting for evaluation
        # For unsatisfactory configurations, skip evaluation and compute penalized objective value
        self.X_design = self.generate_experimental_design(num_design)
        self.constraints_log_design, penalized_values, self.E_design = self.pre_check_constraints(self.X_design)
        self.S_design, self.Y_design = self.process_actual_results(self.E_design, self.X_design, penalized_values)

        # Recheck constraints after obtaining lift and drag in simulations
        # For unsatifactory configurations, significantly penalize the objective value (not as severe as pre-check since base is nonzero from simulation results)
        self.constraints_log_design, self.Y_design = self.post_check_constraints(self.S_design, self.E_design, self.X_design, self.Y_design, self.constraints_log_design)
        self.tabulate_data(self.S_design, self.E_design, self.X_design, self.Y_design)
        if adjust_pitch:
            self.S_design, self.E_design, old_S, old_E = self.adjust_pitch(self.S_design, self.E_design)
            self.S_design, self.Y_design = self.process_actual_results(self.E_design, self.X_design, penalized_values, self.S_design)
            self.S_design, self.E_design = self.reconcile_pitch(self.S_design, self.E_design, old_S, old_E)
            self.constraints_log_design, self.Y_design = self.post_check_constraints(self.S_design, self.E_design, self.X_design, self.Y_design, self.constraints_log_design)
        self.tabulate_data(self.S_design, self.E_design, self.X_design, self.Y_design)

        # Initialize history with values from the experimental design
        self.S_history, self.E_history, self.X_history, self.Y_history, self.constraints_log_history = self.S_design, self.E_design, self.X_design, self.Y_design, self.constraints_log_design      # Initialize history of data
        self.S_best_design, self.E_best_design, self.X_best_design, self.Y_best_design = self.get_best_values(self.S_design, self.E_design, self.X_design, self.Y_design)                           # Get best design configuration
        self.S_best, self.E_best, self.X_best, self.Y_best = self.S_best_design, self.E_best_design, self.X_best_design, self.Y_best_design                                                         # Initialize current best value
        self.S_best_history, self.E_best_history, self.X_best_history, self.Y_best_history = [self.S_best_design], [self.E_best_design], [self.X_best_design], [self.Y_best_design]                 # Initialize history of best values

        #self.write_data_to_spreadsheet(initial_batch=True)

    def additional_batch(self, evaluations_per_batch = 5, adjust_pitch=False):
        self.load_previous_data_from_spreadsheet()
        print(f'\n\n\nNote:\n The target velocity is {float(self.target_velocity):.2f}kph. Meanwhile, pitch estimates are made using data at 45kph.\n\n')
        print(f'''
        {'-'*50}
                        Optimization Cycle
        {'-'*50}
        ''')
        time.sleep(1.0)
        metamodel = GPyOpt.methods.BayesianOptimization(
            f = None, 
            domain = self.variables,
            constraints = None,
            cost_withGradients = None,
            model_type = 'GP',
            X = np.asarray(self.X_history),
            Y = np.asarray(self.Y_history).reshape(-1, 1),  # reshape into 2d array (column)
            acquisition_type = 'EI',
            normalize_Y = True,
            exact_feval = False,
            acquisition_optimizer_type = 'lbfgs',
            evaluator_type = 'local_penalization',
            batch_size = evaluations_per_batch,
            maximize = False,
            de_duplication = True,
            Gower = False,
            noise_var = 0)
        # Precheck constraints prior to suggesting for evaluation
        # For unsatisfactory points, skip evaluation and compute penalized objective value
        self.X = [[int(round(x[0])), int(round(x[1])), int(round(x[2])), int(round(x[3])), x[4], x[5]] for x in metamodel.suggest_next_locations()]
        self.constraints_log, penalized_values, self.E = self.pre_check_constraints(self.X)
        self.S, self.Y = self.process_actual_results(self.E, self.X, penalized_values)

        # Recheck constraints after obtaining lift and drag in simulations
        # For unsatifactory configurations, significantly penalize the objective value (not as severe as pre-check since base is nonzero from simulation results)
        self.constraints_log, self.Y = self.post_check_constraints(self.S, self.E, self.X, self.Y, self.constraints_log)
        self.tabulate_data(self.S_design, self.E_design, self.X_design, self.Y_design)

        # Update history
        self.S_history = self.update(self.S_history, self.S)
        self.E_history = self.update(self.E_history, self.E)
        self.X_history = self.update(self.X_history, self.X)
        self.Y_history = self.update(self.Y_history, self.Y)
        self.constraints_log_history = self.update(self.self.constraints_log_history, self.constraints_log)

        # Update history of best configurations per iteration
        self.S_best, self.E_best, self.X_best, self.Y_best = self.get_best_values(self.S_history, self.E_history, self.X_history, self.Y_history)
        self.S_best_history = self.update(self.S_best_history, self.S_best)
        self.E_best_history = self.update(self.E_best_history, self.E_best)
        self.X_best_history = self.update(self.X_best_history, self.X_best)
        self.Y_best_history = self.update(self.Y_best_history, self.Y_best)

        metamodel.plot_acquisition()
        
        return None    

thesis = SLT_Optimization(target_velocity = 44, payload_weight = 500, motor_angle = 5, beam_length_max = 80)
thesis.initial_batch(num_design = 100, adjust_pitch=True)
#thesis.additional_batch(evaluations_per_batch = 20)
#thesis.load_previous_data_from_spreadsheet()
#thesis.S_design, thesis.E_design = thesis.adjust_pitch(thesis.S_design, thesis.E_design)
#thesis.write_data_to_spreadsheet()