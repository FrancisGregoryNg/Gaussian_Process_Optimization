import xlwings as xw
import pathlib

Datafolder = pathlib.Path(__file__).parents[2].joinpath('2---Data')
motor_index = '{:02}'.format(1)
propeller_index = '{:02}'.format(1)
battery_cellCount = '{:02}'.format(3)
plane = {'weight': 1000}
beam = {'weight_per_L': 1}
motor = {'weight': 30}
propeller = {'weight': 10}
battery = {'weight': 100, 'mah': 3000, 'discharge': 20}
beam_length = 50

test = (str(Datafolder) + '/MBO/RCbenchmark/' 
        + 'M' + motor_index + '_' 
        + 'P' + propeller_index + '_'
        + 'B' + battery_cellCount + '.xlsx')
testData = xw.Book(str(pathlib.PureWindowsPath(test)))

thrust_max = testData.sheets['Interface']['B2'].value
# weight restriction
W = (plane['weight'] 
    + beam_length * beam['weight_per_L']
    + 4 * propeller['weight']
    + 4 * motor['weight']
    + battery['weight']) * 0.0098      # convert from grams to Newtons
print(W)
print(thrust_max)
if W > 4 * thrust_max:
    print('Insufficient quadcopter thrust. ')
else:
    print('Weight is OK. ')
    testData.sheets['Interface']['B1'].value = W/4
    current_draw = testData.sheets['Interface']['B3'].value
    if current_draw < (battery['mah'] / 1000) * battery['discharge']:
        print('Too much current draw. ')     
    else:
        print('Battery is OK. ')
    print(current_draw)