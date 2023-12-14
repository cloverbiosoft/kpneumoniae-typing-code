from scripts.classification.randomforest import RandomForestAnalysis
from scripts.data_retrieval import DataModel, get_dataframe_and_categories, show_results_at_console


print(" ############################## ")
print(" # K pneumoniae - 33 KL-types # ")
print(" ############################## ")

## Get already preprocessed data and categories for K. pneumoniae 33 KL-types model
main_df_kpneumoniae, categories_kpneumoniae = get_dataframe_and_categories(dataModel=DataModel.KPNEUMONIAE_33KTYPES)

## Initialize RandomForestAnalysis
rfAnalysis = RandomForestAnalysis(
    categories = categories_kpneumoniae,
    n_estimators = 200,
    max_features = 12,
    max_depth = None,
    min_samples_split = 2,
    min_samples_leaf = 1) # type: RandomForestAnalysis

## Run model
result_kpneumoniae = rfAnalysis.run(peakmatrix_df = main_df_kpneumoniae)

## Print results
show_results_at_console(result_kpneumoniae)