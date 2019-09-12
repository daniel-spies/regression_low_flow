# matlab_regression_low_flow
Code and data used in the publication: 

"Low-flow characteristics across Switzerland – Snapshot campaigns, data evaluation and regression calculations" 

in order to run the code, input text files with response variable in the first column have to be located in all/input

Code has to be excuted in the following order using R and MATLAB:

1) 1_filter_Data_zero (pre-processing of data) [R]
2) COMPUTATION_zeroReplacement (execute MATLAB regression modelign) [MATLAB]
3) 3_compile_excel_output (summarize model predictions in single excel sheet) [R]
4) 4_Rsquared_summary (summary of regression models)[R]
5) 5_meanImportance_dotplot (plot random forest model variable mean importance) [R]
