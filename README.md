# matlab_regression_low_flow
Code and data used in the publication: 

"Low-flow characteristics across Switzerland – Snapshot campaigns, data evaluation and regression calculations" 

Pre-selected data used is combined in Inputvariablen_gesamt_REDUZIERT_FINAL.xls, this data was used to create input .txt files located in all/input.

Code has to be excuted in the following order using R and MATLAB:

-  1_filter_Data_zero (pre-processing of data and creation of input .txt files) [R]
-  2_COMPUTATION_zeroReplacement (execute MATLAB lasso and randomForest regression) [MATLAB]
-  3_compile_excel_output (summarize model predictions in single excel sheet) [R]
-  4_Rsquared_summary (summary of regression models)[R]
-  5_meanImportance_dotplot (plot random forest model variable mean importance) [R]
