import os
import pandas as pd
import numpy as np

from enum import IntEnum

class DataModel(IntEnum):
    KPNEUMONIAE_33KTYPES = 0

def get_data_paths(dataModel):

    if dataModel == DataModel.KPNEUMONIAE_33KTYPES:
        print('Retrieving K. pneumoniae data')
        main_folder = "data\\K. pneumoniae"
        category_summary_filename = "K. pneumoniae - Categories.csv"

    categories_file = os.path.join(main_folder, category_summary_filename)

    return main_folder, categories_file

def get_dataframe_and_categories(dataModel):

    ## Get the data paths
    main_folder, categories_file = get_data_paths(dataModel)

    ## category file has the relationship between the categories and the samples
    category_summary_data = pd.read_csv(categories_file, sep = ',', header=0)

    ## Read preprocessed data and create main dataframe
    categories = {}
    main_df = pd.DataFrame(columns=['Wavenumber'])

    ## Iterate each row of category_summary_data which has the pair of [category, sample] relationship
    for cat, sample in category_summary_data.values:

        ## Fill categories dict
        if cat in categories.keys():
            categories[cat]['samples'].append(sample)
        else:
            categories[cat] = {'samples': [sample], 'name': cat, 'score': len(categories)}

        ## Read sample file
        file_df = pd.read_csv(os.path.join(main_folder, cat, sample + '.csv'), sep = ',', header=0, names=['Wavenumber', sample])
        
        ## Add data to main dataframe
        main_df = pd.merge(main_df, file_df, on='Wavenumber', how='outer')
    
    return main_df, list(categories.values())

def show_results_at_console(result):

    # Calculate accuracy
    accuracy = np.trace(result['confusion_matrix_kfold']['data']) / np.sum(result['confusion_matrix_kfold']['data'])

    # Get the confusion matrix as dataframe
    cm_columns = [cat['name'] for cat in result['confusion_matrix_kfold']['columns']]
    loo_cm = pd.DataFrame(data=result['confusion_matrix_kfold']['data'], columns=cm_columns, index=cm_columns)

    ## Print results
    print()
    print('Leave-One-Out Confusion matrix')
    print(loo_cm)
    print('Accuracy: ', str(round(accuracy * 100, 2)), '%') 
    print()
