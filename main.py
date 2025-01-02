import pandas as pd
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


import xgboost as xgb


from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

import lightgbm as lgb



def XGB_Reg(X_train,Y_train,X_test,Y_test):
    # Create a DMatrix
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test, label=Y_test)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 4,
        'eta': 0.1,
        'eval_metric': 'rmse'
    }

    # Train the model
    model = xgb.train(params, dtrain, num_boost_round=100)
    y_train_pred = model.predict(dtrain)
    y_test_pred = model.predict(dtest)
    model.save_model('model1.json')
    print(f'RSME_Training: {np.sqrt(np.mean((Y_train - y_train_pred) ** 2))}')
    print(f'RSME_Training: {np.sqrt(np.mean((Y_test - y_test_pred) ** 2))}')
    print(f'R2_Training: {r2_score(Y_train, y_train_pred)}')
    print(f'R2_Training: { r2_score(Y_test, y_test_pred)}')
    return  y_train_pred, y_test_pred



# Function to read files in a directory and convert them to DataFrames
def read_excel_files_to_dataframes(directory,listOfWells):
    combined_dataframe = pd.DataFrame()
    for file_name in listOfWells:
            file_path = os.path.join(directory, file_name)
            df = pd.read_excel(file_path+'.xlsx')
           
            df['Strat_Simplified_Viro'] = df['Strat_Simplified_Viro'].fillna(method='ffill')
            df['WellName'] = file_name
            combined_dataframe = pd.concat([combined_dataframe, df], ignore_index=True)
            
    return combined_dataframe


# Directory containing the uploaded files
listOfExistingWells=["REZOVACCKE_KRCCEVINE-1","VIROVITICA-3ALFA","SUHOPOLJE-1"]
listOfUnknownWells=["REZOVACCKE_KRCCEVINE-1 ISTOK","VIROVITICA-2"]

# Read the files
dfExistingWells = read_excel_files_to_dataframes(r'./Files/',listOfExistingWells)
dfExistingWells.to_csv('CombinedWells.txt', sep='\t', index=False)
dfUnknownWells= read_excel_files_to_dataframes(r'./Files/unknown/',listOfUnknownWells)
print ("Reading Data: Finished------------------")

# Prediction
model_name='XGB_Reg'

pred_para= 'perm'
# scatter log
plotting= 'log'

dfExistingWells = dfExistingWells.dropna(subset=[pred_para]).reset_index(drop=True)
inputData= pd.DataFrame()
inputData['x']= dfExistingWells['Easting (m)']
inputData['y']= dfExistingWells['Northing (m)']
inputData['TVDss']= dfExistingWells['TVDSS (m)']
inputData['Layer']= dfExistingWells['Strat_Simplified_Viro']



input_features = inputData


prediction_parameters = dfExistingWells[pred_para]

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(input_features, prediction_parameters, test_size=0.2, random_state=42)

# Normalize the input data set before training
scaler_X = MinMaxScaler()



if model_name=='XGB_Reg':
        y_train_pred, y_test_pred=XGB_Reg(X_train,y_train,X_test,y_test)
        loaded_model = xgb.XGBRegressor()
        loaded_model.load_model('model1.json')

# Use the loaded model for predictions
y_pred = loaded_model.predict(X_test)

if (plotting== 'scatter'):
    # Visualize the quality of training and testing
    # Plot true vs predicted values for training and testing data
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot true vs predicted values for training and testing data
    train_plot = ax.scatter(y_train, y_train_pred, color='blue', label='True Train Data vs Predicted Train Data')
    test_plot = ax.scatter(y_test, y_test_pred, color='red', label='True Test Data vs Predicted Test Data')

    # Add title and labels
    ax.set_title('True vs Predicted Values')
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.legend()

    # Add toggle functionality
    def toggle_visibility(event):
        if event.key == 't':  # Toggle Train Plot visibility
            train_plot.set_visible(not train_plot.get_visible())
            plt.draw()
        if event.key == 'p':  # Toggle Test Plot visibility
            test_plot.set_visible(not test_plot.get_visible())
            plt.draw()

    # Connect the toggle functionality to the figure
    fig.canvas.mpl_connect('key_press_event', toggle_visibility)

    # Display the plot
    plt.tight_layout()
    plt.show()
    
    
elif (plotting=='log'):
    newWellData=pd.DataFrame()
    newWellData['x']= dfUnknownWells['Easting (m)']
    newWellData['y']= dfUnknownWells['Northing (m)']
    newWellData['TVDss']= dfUnknownWells['TVDSS (m)']
    newWellData['Layer']= dfUnknownWells['Strat_Simplified_Viro']
    newWellData['Prediction']=loaded_model.predict(newWellData)
    newWellData['WellName']=dfUnknownWells['WellName']
    newWellData[pred_para]=newWellData['Prediction']

    y_pred = loaded_model.predict(input_features)
    inputData['Prediction']=y_pred
    inputData['WellName']=dfExistingWells['WellName']
    inputData[pred_para]=dfExistingWells[pred_para]
    
    
    inputData = pd.concat([inputData, newWellData], ignore_index=True)
    
    
    unique_wells = inputData['WellName'].unique()
    n_wells = len(unique_wells)
    fig, axes = plt.subplots(1, n_wells, figsize=(5 * n_wells, 6))  # Adjust the size as needed


    for i, well in enumerate(unique_wells):
        well_data = inputData[inputData['WellName'] == well]
        if n_wells == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.plot(well_data[pred_para], well_data['TVDss'], label='Actual', marker='o')
        ax.plot(well_data['Prediction'], well_data['TVDss'], label='Prediction', marker='x')
        ax.set_title(f'Well Log for {well}')
        ax.set_xlabel(pred_para)
        ax.set_ylabel('TVD (True Vertical Depth)')
        ax.invert_yaxis()  # Invert y-axis as depth increases downwards
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.show()
    inputData.to_csv(pred_para+'.txt', sep='\t', index=False)
print('Finished--------------------------------')



