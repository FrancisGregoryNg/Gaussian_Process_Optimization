import xlrd

def load_component_info():
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
        
        '''
        with xlrd.open_workbook('components.xlsx', on_demand=True) as book:
            sheet = book.sheet_by_name('thrust')
            motors = sheet.col_values(0) #maintain alignment with rowx, get() will just ignore the nonexisting key
            propellers = sheet.row_values(0, start_colx=1)
            for rowx in range(1, sheet.nrows):
                thrust = sheet.row_values(rowx, start_colx=1)
                thrust_dict = ({propeller: thrust_value for propeller, thrust_value in zip (propellers, thrust) 
                               if (thrust_value is not None and thrust_value != '' and thrust_value != 0)})
                motor_to_update = next((item for item in motor_data if item.get('name') == motors[rowx]), None)
                if motor_to_update is not None:
                    motor_to_update.update({'thrust': thrust_dict}) 
        '''
        return plane_data, beam_data, battery_data, motor_data, propeller_data

plane, beam, battery, motor, propeller = load_component_info()
for item in [plane, beam, battery, motor, propeller]:
    print(item)

print(battery['weight'])