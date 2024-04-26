from prog_models.models import BatteryCircuit
import warnings
import pickle
import os
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")
script_directory = os.path.dirname(os.path.abspath("__file__"))

machine = BatteryCircuit()
states = machine.default_parameters['x0']

order_model = pickle.load(open(os.path.join(script_directory, 'breakdownClassifier.pkl'), 'rb'))
#lot_model = pickle.load(open(os.path.join(script_directory, 'lot_prognostics_' + type(learner).__name__), 'rb'))
    

def approxHealth(health, throughputRate, processingTime):
    failure = order_model.predict([[health, throughputRate, processingTime]])[0]
    return (failure)

#def getLotSize(throughputRate, degradation):
#    # Look for load_time that achieves a degradation equal to the remaining health of the machine
#    processingTime = lot_model.predict([[throughputRate, degradation]])[0]
#    return (processingTime)

# Use prog_models package for simulation of health
def runMachine(load, load_time): 
        global states
        global machine
        # Define load of battery
        def future_loading(t, x=None):
            return {'i': load}

        # Set current state of machine
        machine.parameters['x0'] = states

        # Simulate 'time' steps
        options = {
            'save_freq': load_time,  # Frequency at which results are saved
            'dt': 2  # Timestep
        }
        (_, _, states_list, _, event_states) = machine.simulate_to(load_time, future_loading, **options)
        health = event_states[-1]['EOD']
        states = states_list[-1]
        return(health)

def resetMachine():
    global states
    states = machine.default_parameters['x0']



def storeScore(run, maintenanceStrategy, numOrders, scheduledMaintenanceInterval, setupTime,
                                 maintenanceTime, repairTime, shockProb, shockImpactMulti, noiseSigma, MTTF,
                                 numRepair, numMain, makespanSeconds):
    file_path = os.path.join(script_directory, 'scores.xlsx')
    # Create a DataFrame with the new row
    new_row = {
        "Date": datetime.now(),
        "Run": run,
        "Strategy": maintenanceStrategy,
        "NumOrders": numOrders,
        "ScheduledMaintenanceInterval": scheduledMaintenanceInterval,
        "SetupTime": setupTime,
        "MaintenanceTime": maintenanceTime,
        "RepairTime": repairTime,
        "ShockProb": shockProb,
        "ShockImpactMulti": shockImpactMulti,
        "noiseSigma": noiseSigma,
        "MTTF": MTTF,
        "NumRepair": numRepair,
        "NumMain": numMain,
        "MakespanSeconds": makespanSeconds
    }
    new_df = pd.DataFrame([new_row])

    # Read existing Excel file into a DataFrame or create a new DataFrame if the file doesn't exist
    try:
        existing_df = pd.read_excel(file_path)
    except FileNotFoundError:
        existing_df = pd.DataFrame(columns=["Datetime", "Run", "Strategy", "NumOrders", "ScheduledMaintenanceInterval",
                                            "SetupTime", "MaintenanceTime", "RepairTime", "ShockProb",
                                            "ShockImpactMulti", "noiseSigma", "MTTF", "NumRepair", "NumMain",
                                            "MakespanSeconds"])

    # Concatenate the existing DataFrame with the new row and write back to the Excel file
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    updated_df.to_excel(file_path, index=False)


    
    
        