
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPUTATION OF STATISTICS AND APPLICATION OF REGRESSION PROCEDURES %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clean up
close all;
clear all;
clc;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NEEDED INPUTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%input data has to be in a textfile (or multiple textfiles for automatic seperate computations)

%1) column number of the response variable in the input textfile
col_num_response = 1;

% STRUCTURAL DECISIONS

%1) exporting statistics of variables as excel-file?
I_stats_export = 1;  % 0 = NO ; 1 = YES

%2) creating variables-scatterplots (including cross-correlation coefficients)?
I_var_plot = 0;  % 0 = NO ; 1 = YES

%3) Including interaction terms in the model?
I_mdl_interactions = 0;  % 0 = NO ; 1 = YES

%4) Application of regression procedure also on log-transformed response variable?
% I_response_log = 1;  % 0 = NO ; 1 = YES

%5) Exporting validation plots (residual plots, leverage plot, etc.)?
I_mdl_validation_export = 0;  % 0 = NO ; 1 = YES

%6) Exporting regression model results (model estimates, measures of error, etc.)?
I_mdl_results_export = 1;  % 0 = NO ; 1 = YES

%7) definition of categorical variables:
% 
%   if certain variable(s) shall be considered as categorical variable(s),
%   put its/their column number(s) in the square brackets below, otherwise
%   leave the square brackets empty.
%   example: if the variables in the columns 6 and 8 of the input matrix
%   shall be categorical variables, you insert: categorical = [6 8];
categorical = [];

%8) definiton of categories of categorical variables:
% 
%   Per default every categorical variable gets two categories, one for
%   zero-values and the other for non-zero values. If you want to define how
%   many categories a categorical variable shall have and where their
%   boundaries should be, write following line of code (and see example further below for better understanding):
%   defined_categories("column number of categorical variable in input matrix",:) = ["numerical boundary-values"];

defined_categories = [];
 
%   IMPORTANT: all the categorical variables must have the same number of
%   categories, otherwise the code won't work. For example, if one variable
%   shall have four categories and another variable only two categories, you
%   can define two more categories for the latter, which you know that no
%   variable-values will fall into these two additional category-intervals.
%   example: The variable in the column 6 of the input matrix shall have
%   four categories defined as category 1 from 0 to below 10, category 2
%   from 10 to below 25, category 3 from 25 to below 50 and category 4 from
%   50 to below 100. The variable in the column 8 of the input matrix shall
%   have two categories, category 1 from 0 to below 15 and category 2 from
%   15 to below 25. Because the other variable shall have four categories,
%   this variable needs two more categories (that both variables have 4
%   categories). Therefore, you also define e.g. category 3 as 25 to below
%   27 and category 4 as 27 to below 30. But it won't make any difference
%   for the output for this variable as you (should) know that it has no
%   observation values above 25.
%   For this example you would write following lines of code:
%   defined_categories(6,:) = [0 10 25 50 100];
%   defined_categories(8,:) = [0 15 25 27 30];

%10) IF STEPWISE SELECTION CHOSEN:
% define the AIC-values for the enter- and remove criterion:
AIC_enter = -2; AIC_remove = -1; %(enter-value has to be lower than remove-value!)

%11) IF SEQUENTIAL FEATURE SELECTION CHOSEN:
% define the value by which the cross-validated MSE has to improve
% for fulfilling the enter-criterion:
sequentialfs_enter_value = 0.001;


%%%%%%%%%%%%%%%%% NO INPUTS AND DECISIONS NEEDED FROM HERE %%%%%%%%%%%%%%%%%
% data import
L = 1; %no clustering, therefore only one (L=1) output-file per input
dinfo = dir('inputs/*.txt'); %assigning that all and exclusively textfiles in the folder are considered
for K = 1 : length(dinfo) %going through all given textfiles in the folder individually
  filename = dinfo(K).name; %assigning name of currently considered textfile
  textfile = importdata(fullfile('inputs',filename));
  variable_names = textfile.textdata; %assigning response and vairable names
  input_data = textfile.data; %assigning all reponse and variable values as matrix
  
  % use log- response variable?
  I_response_log = 1;
  
  name_response = string(getfield(strsplit(filename, '_'),{1,3}));
  % apply regression procedures
  randomForest_model(input_data,variable_names,name_response,col_num_response,I_response_log,I_mdl_validation_export,I_mdl_results_export,categorical,defined_categories,I_mdl_interactions,K,L,filename); %calling respective function
  lasso_model(input_data,variable_names,name_response,col_num_response,I_response_log,I_mdl_validation_export,I_mdl_results_export,categorical,defined_categories,I_mdl_interactions,K,L,filename); %calling respective function
  sequentialfs_model(input_data,variable_names,name_response,col_num_response,I_response_log,I_mdl_validation_export,I_mdl_results_export,I_mdl_interactions,categorical,defined_categories,sequentialfs_enter_value,excel_language,K,L,filename); %calling respective function
  stepwise_model(input_data,variable_names,name_response,col_num_response,I_response_log,I_mdl_validation_export,I_mdl_results_export,I_mdl_interactions,categorical,defined_categories,AIC_enter,AIC_remove,K,L,filename); %calling respective function
end