
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% application of regression procedure with sequential feature selection %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%NEEDED INPUTS:
%input_data: values of the different variables over the different catchments
%variable_names: names of the input variables (including response)
%col_num_response: column number of response in the input-matrix (textfile)
%name_response: name of the response variable
%I_response_log: decision if with log-transformed response variable
%I_mdl_validation_export: decision if export of validation plots
%I_mdl_results_export: decision if export of model outputs
%I_mdl_interactions: decision if consideration of interaction terms
%categorical: column numbers of categorical variables in input-matrix
%defined_categories: defined boundaries between categories of categorical variables
%sequentialfs_enter_value: defining by how much new cross-validated MSE has to improve current one for including the currently considered input variable
%excel_language: default language in excel
%K: number of input-textfile (textfile in folder with data)
%L: number of cluster-interval

%VARIABLE RETURN:
%no return

%OUTPUTS (if chosen):
%model estimates, residuals, goodness-of-fit measures and more
%model validation plots


function [] = sequentialfs_model(input_data,variable_names,name_response,col_num_response,I_response_log,I_mdl_validation_export,I_mdl_results_export,I_mdl_interactions,categorical,defined_categories,sequentialfs_enter_value,excel_language,K,L,filename)

sprintf("sequential processing file: %s",filename)
%remove NaN-rows
input_data(any(isnan(input_data), 2), :) = [];

% defining untransformed response variable
input_raw = input_data(:,col_num_response);
response_types_as_matrix = input_raw; %response in vector-form
response_types = array2table(input_raw,'VariableNames',name_response);
response_names = {name_response};

% defining log-transformed response variable
if I_response_log == 1 %if chosen, defining the log-transformed response
    input_nonzero = input_raw+0.00001; %log-transformation not possible with zero values, therefore +0.00001 to all response-values
    input_log = log10(input_nonzero); %log-transformation
    response_types_as_matrix = horzcat(response_types_as_matrix, input_log); %merging of columns of untransformed and log-transformed response-values in vector-form
    response_log = array2table(input_log,'VariableNames',strcat(name_response,'_log')); %generating name of log-transformed response
    response_types = horzcat(response_types, response_log); %merging of columns of untransformed and log-transformed response-values as table
    response_names = [response_names {strcat(name_response,'_log')}]; %merging of untransformed response name with name of log-transformed response name (same name with "_log" at the end)
end


% DEFINING CATEGORICAL VARIABLES

input_stand = input_data;

%generating matrix consisting the DEFINED categorical boundaries of every variable.
%Its number of rows corresponds to the number of variables and its number
%of columns corresponds to the number of defined boundaries (all variables
%with defined categorical boundaries should have the same number of
%boundaries as stated in the main matlab-file (structurl decisions)). Every
%variable with no defined boundaries get only zeros assigned.
if size(defined_categories,2) > 0
    if size(defined_categories,2) < size(input_data,2)
        categories_boundaries = vertcat(defined_categories,zeros(size(input_data,2)-size(defined_categories,1),size(defined_categories,2)));
    else
        categories_boundaries = defined_categories;
    end
else
    categories_boundaries = zeros(size(input_data,2),1);
end
 
%generating vector containing column numbers of variables which did not get
%defined as being categorical
no_categoricals = 1:size(input_data,2);
no_categoricals(:,categorical) = [];

categorical_variable_names = [];


%transforming the values of the as categorical labelled variables to the 
%category-values according to the defined boundaries (or the default boundaries
%if none were defined) (e.g. if a variable got four different categories
%defined, the values will get transformed according to their "real" values
%to category-values of 1,2,3 or 4). The actual categorical variables will be
%generated during the fitting procedure within the sequential feature selection
%by defining which variables shall be categorical (see further below)
for j = 1:size(categorical,2)
    
    %names of categorical variables:
    categorical_variable_names = [categorical_variable_names variable_names(categorical(j))];
    
    if sum(abs(categories_boundaries(categorical(j),:))) == 0 
        categories = [-1 0.00000001 max(input_stand(:,categorical(j)))];
        input_stand(:,categorical(j)) = discretize(input_stand(:,categorical(j)),categories,'categorical');
    else
        categories = categories_boundaries(categorical(j),:);
        input_stand(:,categorical(j)) = discretize(input_stand(:,categorical(j)),categories,'categorical');
    end
end

%standardizing all variables which did not get defined as being categorical
input_stand(:,no_categoricals) = zscore(input_stand(:,no_categoricals)); %dann werden die Variablen/Spalten der Inputmatrix welche als keine categorical variables deklariert wurden standardisiert

%correcting categorical-postitions (necessary because the response variable
%gets removed from the latter used variable-matrix, meaning that the
%defined column numbers of the categorical variables have to be "moved" by
%one if their colum number is smaller than the one of the response in the
%input matrix (textfile)
categorical_cv = categorical;
if col_num_response == 1
    categorical_cv = categorical-1;
else 
    categorical_cv(categorical > col_num_response) = categorical(categorical>col_num_response)-1;
end

% correcting variable names
variable_names_cv = variable_names;
variable_names_cv(col_num_response) = [];

%defining (potential) predictor variables
ezg_variables_as_matrix = input_stand;
ezg_variables_as_matrix(:,col_num_response) = [];
ezg_variables = array2table(input_stand,'VariableNames',variable_names);
ezg_variables(:,col_num_response) = [];

%computing the leverage of the different catchments based on the
%standardized variables values
ezg_leverage = leverage(ezg_variables_as_matrix);


if I_mdl_interactions == 1
    %if chosen, adding interaction terms to predictor data and predictor names

    % contrary to the stepwise function, there is no possibilty to let matlab
    % automatically generate interaction terms within the fitting-function used here (further below).
    % However, defining which variables shall be considered as categorical is possible.
    % Therefore, following loops and if-statements transform the categorical variables
    % with their assigned "category-numbers" (see principle above) into the
    % real categorical variables (0/1-vector for every single category) and
    % multiply them with all other variables to get the interaction terms. The
    % categorical variable itself remain unchanged (still with their
    % "category-numbers" to let them then be transformed into the different
    % 0/1-categorical variables within the fitting function.
    categories_boundaries(col_num_response,:) = [];
    ezg_variables_with_interactions = ezg_variables_as_matrix;
    for j = 1:(size(ezg_variables,2)-1) %this for loop and the next below ensures that all variables get multiplied with one another (but no combination not more than ones, e.g. not Var1*Var2 and Var2*Var1 as well as it is the same) 
        for k = (j+1):size(ezg_variables,2)
            if ismember(j,categorical_cv) %checks if the "first" currently considered variable is defined as categorical
                for l = 1:max(ezg_variables_as_matrix(:,j)) %if yes the foor-loop takes as many "rounds" as it highest value which corresponds to its number of categories

                        logical_vector_1 = ezg_variables_as_matrix(:,j) == l; %generating the 0/1-categorical-variable for the currently considered category (l goes through all defined categories; one after another per round)
                        logical_variable_name_1 = strcat(variable_names_cv{j},'_',num2str(l)); %generating the according name of the categorical variable (e.g. category 2 of variable 3 gets the name "variable3_2")

                        if ismember(k,categorical_cv) %if the second currently considered variable is also categorical, the same principle gets carried out as above (always two variable sget considered obviously for generating the interaction-terms)
                            for m = 1:max(ezg_variables_as_matrix(:,k))
                                logical_vector_2 = ezg_variables_as_matrix(:,k) == m;
                                logical_variable_name_2 = strcat(variable_names_cv{k},'_',num2str(m));
                                ezg_variables_with_interactions = [ezg_variables_with_interactions logical_vector_1.*logical_vector_2];
                                variable_names_cv = [variable_names_cv {strcat(logical_variable_name_1,'_',logical_variable_name_2)}]; 
                            end
                        else %if the second variable is not categorical it can simply multiplied with the generated categorical 0/1-variables above
                            ezg_variables_with_interactions = [ezg_variables_with_interactions logical_vector_1.*ezg_variables_as_matrix(:,k)];
                            variable_names_cv = [variable_names_cv {strcat(logical_variable_name_1,'_',variable_names{k})}];
                        end
                end
            else %if the "first" variable is not defined as categorical, it simply can get used for multiplication with the second variable, which below gets reviewed as well if it is defined as categorical and should be transformed based on the same principle as above
                 if ismember(k,categorical_cv)
                      for n = 1:max(ezg_variables_as_matrix(:,k))
                          logical_vector_2 = ezg_variables_as_matrix(:,k) == n;
                          logical_variable_name_2 = strcat(variable_names_cv{k},'_',num2str(n));
                          ezg_variables_with_interactions = [ezg_variables_with_interactions ezg_variables_as_matrix(:,j).*logical_vector_2];
                          variable_names_cv = [variable_names_cv {strcat(variable_names_cv{j},'_',logical_variable_name_2)}];
                      end
                 else
                       ezg_variables_with_interactions = [ezg_variables_with_interactions ezg_variables_as_matrix(:,j).*ezg_variables_as_matrix(:,k)];
                       variable_names_cv = [variable_names_cv {strcat(variable_names_cv{j},'_',variable_names_cv{k})}];
                 end
            end
        end
    end

    %generating table of just created matrix with interaction terms
    ezg_variables = array2table(ezg_variables_with_interactions,'VariableNames',variable_names_cv);

    % assigning created matrix with interactions in "normal" form (not as table)
    ezg_variables_as_matrix = ezg_variables_with_interactions;

end

% removing columns with zeros
cols_with_all_zeros = find(all(ezg_variables_as_matrix==0));
ezg_variables(:,cols_with_all_zeros) = [];
ezg_variables_as_matrix(:,cols_with_all_zeros) = [];
variable_names_cv(cols_with_all_zeros) = [];


count = 0;

%now the sequential feature selection gets applied. If also the log-transformed
%response is considered (size(response_types,2)=2), then the whole following 
%procedures will be applied twice, once for the untransformed response and 
%once for the log-transformed one
for i = 1:size(response_types,2)
    
    count=count+1;
    
    % APPLYING SEQUENTIAL FEATURE SELECTION BASED ON MSE OF 10-FOLD CROSS-VALDIATION

    F = 10; %F=10 because 10-fold CV
    cv = cvpartition(size(response_types,1), 'kfold',F); %ten times partitioning of the data in nine training data folds and one testing data fold
    mse = zeros(F,1); %initialising vector for cross-validated MSEs for every testing fold
    
    for f=1:F %the sequential feature selection principle will be applied ten times according to the 10-fold cross-validation
        
        % training/testing indices for the current fold
        trainIdx = cv.training(f);
        testIdx = cv.test(f);
        
        %SEQUENTIAL FEATURE SELECTION
        
        % initialising of variables and setting of starting values
        
        CV_MSE_current = 10^28;
        
        potential_predictors = ezg_variables(trainIdx,:);
        potential_predictors_all = ezg_variables;
        variable_names_current = variable_names_cv;

        model_variables = response_types(trainIdx,i);
        model_variables_all = response_types(:,i); %needed for the "outter" cross-validation (the cross-validation of the outputs of the feature selection and not the cross-validation within the feature selection to select variables)
        model_variables_as_matrix = response_types_as_matrix(trainIdx,i);
        
        num_potential_predictors = size(potential_predictors,2);
        new_predictor = 0;
        new_predictor_all = 0;
        model_categorical_variables = [];
        
        %preparation of the folds for the 10-fold cross-validaiton
        
        F2 = 10; %F=10 because 10-fold CV
        cv2 = cvpartition(size(model_variables,1), 'kfold',F);
        
        while ~isempty(new_predictor) %feature selection continuous as long as one of the input variables fulfills enter-criterion
  
            new_predictor = [];
            new_predictor_all = []; %needed for the "outter" cross-validation (the cross-validation of the outputs of the feature selection and not the cross-validation within the feature selection to select variables)
            
            col_new_predictor = []; 
            new_categorical_variable = [];
            
            for k = 1:num_potential_predictors %in each round one of the potential model variables (input variables) gets individually considered...
       
                model_variables = horzcat(model_variables,potential_predictors(:,k)); %...by adding it to the current model variables (variables which are definitely in the model equation)
                model_variables_all = horzcat(model_variables_all,potential_predictors_all(:,k));
                mse=[];
                
                if any(strcmp(categorical_variable_names,variable_names_current(k))) %checks if the name of the currently considered variable belongs to one of the categorical ones, if yes it gets accordingly added to the categorical model variables (needed to "tell" the fitting function which variables are categorical)
                     model_categorical_variables = horzcat(model_categorical_variables,variable_names_current(k));
                end
    
                for f2=1:F2
                %applying cross-validation for the feature selection (so basically the training data for the cross-validation of the feature selection model outputs gets split again into training data and testing data for the actual feature selection)
                
                %the mean MSE of a 10-fold cross-validation gets computed
                %when the currently considered variable gets added to the
                %already included model variables.
                
                    % training/testing indices for this fold
                    trainIdx2 = cv2.training(f2);
                    testIdx2 = cv2.test(f2);
                    
                    % fitting function on training data
                    mdl = fitlm(model_variables(trainIdx2,:),'ResponseVar',response_names{i},'CategoricalVars',model_categorical_variables);

                    
                    % predict regression output on testing data
                    Y_hat = predict(mdl,model_variables(testIdx2,:));
                    
                    % computing 10^(estimate-value) with estimates based on log-transformed
                    % response to get estimates of untransformed response
                    if i==2
                        Y_hat = 10.^(Y_hat);
                    end
                    
                    % compute mean squared error
                    mse(f2,i) = mean((model_variables_as_matrix(testIdx2,1) - Y_hat).^2); %diese gesch�tzten Werte werden dann von den jeweils tats�chlichen response values abgezogen, dann das Resultat jeweils quadriert und davon den Durchnitt genommen, was somit schlussendlich den MSE des k-folds gibt, basierend auf der Modellsch�tzung mit der jeweiligen training data und dann Anwendung auf die test data
            
                end 
                % average MSE across k-folds
                CV_MSE_new = mean(mse(:,i));
                
                if CV_MSE_new + sequentialfs_enter_value < CV_MSE_current %cheks if with the inclusion of the currently considered variable the corrently best/lowest cross-validated MSE gets improved/"underbid" by a defined amount "sequentialfs_enter_value"  
                    CV_MSE_current = CV_MSE_new; %if enter-criterion gets fulfilled, the new cross-validated MSE gets defined as the current one
                    new_predictor = potential_predictors(:,k); %...and the input variable defined as the current best new predictor 
                    new_predictor_all = potential_predictors_all(:,k);
                    new_categorical_variable = []; %deleting "old" defined categorical variable
                    col_new_predictor = k; %assigning the colum number in the input matrix of the current best new predictor
                    if any(strcmp(categorical_variable_names,variable_names_current(k)))
                        new_categorical_variable = variable_names_current(k); %%if input variable is categorical, it gets separately assigned
                    end
                end

                model_variables(:,end)=[]; %deleting the currently added variable from the model variables for the next round to test the next potetnial predictor
                model_variables_all(:,end)=[];
                if any(strcmp(categorical_variable_names,variable_names_current(k)))
                    model_categorical_variables(:,end)=[]; %including the categorical list if it was a categorical variable
                end

            end
        
            model_variables = horzcat(model_variables,new_predictor); %at the end the variable which remained as the best choice considering the enter-criterion gets definitely added to the model variables and thereby included in the final regression model
            model_variables_all = horzcat(model_variables_all,new_predictor_all);
            model_categorical_variables = horzcat(model_categorical_variables,new_categorical_variable); %... the same thing seperately if it's a categorical variable
            potential_predictors(:,col_new_predictor) = []; %the now added variable into the model gets removed from the "list" of potential predictors
            potential_predictors_all(:,col_new_predictor) = [];
            variable_names_current(:,col_new_predictor) = []; %die Spalte dieser "besten" Variable im variable_anmes_cv-Vektor wird rausgel�scht da diese Variable ja nun definitiv in der definitiven predictor-Matrix X ist => muss gemacht werden sonst gibt es ein Durcheinander bez�glich der categorical variables
            num_potential_predictors = size(potential_predictors,2); %evaluationg how many input variables are left for testing the enter-criterion on so that the number of loops gets set accordingly
        end

        % appliyng linear regression on training data of model terms determined
        % through sequential feature selection

        % now the selected variables are used to fit them on the training
        % data for the "outer" cros-validation (the one to test the
        % performance of the model output itself)
        mdl = fitlm(model_variables_all(trainIdx,:),'ResponseVar',response_names{i},'CategoricalVars',model_categorical_variables);
        
        % predict response on testing data
        cv_Y_hat = predict(mdl,model_variables_all(testIdx,:));
        
        
        % compute measures of error and measure of correlation

        % computing 10^(estimate-value) with estimates based on log-transformed
        % response to get estimates of untransformed response
        if i==2
             cv_Y_hat = 10.^(cv_Y_hat);
        end

        % mean squared error
        cv_mse(f,i) = mean((response_types_as_matrix(testIdx,1) - cv_Y_hat).^2);
        
        % mean absolute error
        cv_mae(f,i) = mean(abs(response_types_as_matrix(testIdx,1) - cv_Y_hat));
        
        % explained variance
        cv_residuals_squared = (response_types_as_matrix(testIdx,1) - cv_Y_hat).^2;
        cv_observation_mean_differences_squared = (response_types_as_matrix(testIdx,1) - mean(response_types_as_matrix(testIdx,1))).^2; %"1" instead of "i" as also for the log-transformed response you want to refer to the untransformed response values as you transform back (see above)
        
        cv_RSS = sum(cv_residuals_squared);
        cv_TSS = sum(cv_observation_mean_differences_squared);
        rsquared(f,i) = 1 - cv_RSS/cv_TSS;   
    end
   

    % compute average measures of error across k-folds
    
    % average MSE
    CV_MSE(i) = mean(cv_mse(:,i));
    
    % average RMSE
    CV_RMSE(i) = sqrt(CV_MSE(i));
    
    % average MAE
    CV_MAE(i) = mean(cv_mae(:,i));
    
    % average R^2
    CV_Rsquared(i) = mean(rsquared(:,i));
    
    
    
    
    %SEQUENTIAL FEATURE SELECTION PROCEDURE ON ALL DATA
    
    %now the same regression procedure as before gets repeated, but this
    %time the model gets fit and tested as well on all data. Thereby, the
    %data used for model fitting and model testing is the same
    
    F = 10;
    cv = cvpartition(size(response_types,1), 'kfold',F);
    mse = zeros(F,1);
    
    CV_MSE_current = 10^28;
    
    potential_predictors = ezg_variables;
    variable_names_current = variable_names_cv;

    model_variables = response_types(:,i);
    
    num_potential_predictors = size(potential_predictors,2);
    new_predictor = 0;
    model_categorical_variables = [];
    col_num_modelterms = [];

    while ~isempty(new_predictor)
        
        new_predictor = [];
        col_new_predictor = []; 
        new_categorical_variable = [];
        
        for k = 1:num_potential_predictors
            
            model_variables = horzcat(model_variables,potential_predictors(:,k));
            
            if any(strcmp(categorical_variable_names,variable_names_current(k)))
                model_categorical_variables = horzcat(model_categorical_variables,variable_names_current(k));
            end
    
            for f=1:F
                    trainIdx = cv.training(f);
                    testIdx = cv.test(f);
                    
                    mdl = fitlm(model_variables(trainIdx,:),'ResponseVar',response_names{i},'CategoricalVars',model_categorical_variables);
        
                    Y_hat = predict(mdl,model_variables(testIdx,:));
                    
                    if i==2
                        Y_hat = 10.^(Y_hat);
                    end
                    
                    % compute mean squared error
                    mse(f,i) = mean((response_types_as_matrix(testIdx,1) - Y_hat).^2);
                    
            end 
            % average MSE across k-folds
            
            CV_MSE_new = mean(mse(:,i));
            
            if CV_MSE_new + sequentialfs_enter_value < CV_MSE_current
                CV_MSE_current = CV_MSE_new;
                new_predictor = potential_predictors(:,k);
                new_categorical_variable = [];
                col_new_predictor = k;
                if any(strcmp(categorical_variable_names,variable_names_current(k)))
                    new_categorical_variable = variable_names_current(k);
                end
            end

            model_variables(:,end)=[];
            if any(strcmp(categorical_variable_names,variable_names_current(k)))
                model_categorical_variables(:,end)=[];
            end

        end
        
        model_variables = horzcat(model_variables,new_predictor);
        model_categorical_variables = horzcat(model_categorical_variables,new_categorical_variable);
        potential_predictors(:,col_new_predictor) = [];
        variable_names_current(:,col_new_predictor) = [];
        num_potential_predictors = size(potential_predictors,2);
    end
    
    
    % appliyng linear regression on all data of model terms determined
    % through sequential feature selection

    mdl = fitglm(model_variables,'ResponseVar',response_names{i},'CategoricalVars',model_categorical_variables);
    
    
    %computation of model estimates and residuals
    
    model_estimates(:,i) = table2array(mdl.Fitted(:,1));
    
    % computing 10^(estimate-value) with estimates based on log-transformed
    % response to get estimates of untransformed response
    if i==2
        model_estimates(:,i) = 10.^(model_estimates(:,i));
    end
    
    residuals(:,i) = response_types_as_matrix(:,1) - model_estimates(:,i);
    
    observation_mean(i) = mean(response_types_as_matrix(:,1));
    
    residuals_squared(:,i) = residuals(:,i).^2;
    observation_mean_differences_squared(:,i) = (response_types_as_matrix(:,1) - observation_mean(1)).^2; %"1" instead of "i" as also for the log-transformed response you want to refer to the untransformed response values as you transform back (see above)
   
    
    %computation of measures of error and measures of correlation for fit on all data
    
    % mean squared error
    MSE(i) = mean(residuals_squared(:,i));
    
    % root mean squared error
    RMSE(i) = sqrt(MSE(i));
        
    % mean absolute error
    MAE(i) = mean(abs(residuals(:,i)));
    
    % explained variance
    RSS(i) = sum(residuals_squared(:,i));
    TSS(i) = sum(observation_mean_differences_squared(:,i));
    Rsquared(i) = 1 - RSS(i)/TSS(i);
    
    % summarizing all measures in one vector
    regression_output(count*2-1,:)=[response_names(i),observation_mean(1),CV_MSE(i),CV_RMSE(i),CV_MAE(i),CV_Rsquared(i)];
    regression_output(count*2,:)=[response_names(i),observation_mean(1),MSE(i),RMSE(i),MAE(i),Rsquared(i)];
    
    
    if I_mdl_validation_export == 1
    %GENERATING PLOTS
    
    % creating matrix without interaction-terms for certain validation plots
    ezg_variables_as_matrix_no_interactions = input_stand;
    ezg_variables_as_matrix_no_interactions(:,col_num_response) = [];
    

    %1) normal probability plot of residuals => https://ch.mathworks.com/help/stats/normplot.html
    
    % generating plot
    figure (1+6*(i-1));
    normplot(residuals(:,i))
    title("");
    
    % moving of the title
    %hLabel = get(gca,'title'); 
    %set(hLabel, 'Position', get(hLabel, 'Position') + [0 0.1 0]); 
    %set(hLabel, 'Units', 'pixels');
    
    % moving of x-axis label
    hLabel = get(gca,'xlabel'); 
    set(hLabel, 'Position', get(hLabel, 'Position') - [0 .01 0]); 
    set(hLabel, 'Units', 'pixels');
    
    % changing name of x-axis
    hLabel = get(gca,'xlabel'); 
    set(hLabel,'String'," residuals [mm]");
    
    
    
    %2) Plot estimated values vs. residuals with moving average of all
    %   residuals and moving 5%- and 95%-percentiles based on the respective
    %   last 30 values
    
    % adding smooth fit (polynomial function third grade) to investigate for
    % trends (non-linear relations)
    estimates_res_comb = [model_estimates(:,i) residuals(:,i)]; %merging of estimates and residuals
    estimates_res_comb_sorted = sortrows(estimates_res_comb,1); %sorting of the colums by value
    

    % computation of moving 5- and 95-percentiles of the residuals base on
    % the respective last 30 values
    k=0;
    for j = 31:size(residuals(:,i),1) %starts at the 31. residual
        k=k+1;
        residuals_percentiles_5_95(k,:) = prctile(estimates_res_comb_sorted((j-30):j,2),[5 95]); %computation of the moving percentiles (first based on residuals 1-30, then 2-31 and so on)
    end
    
    % applying smooth fit on 5-percentiles
    fit_res_quant5 = polyfit(estimates_res_comb_sorted(31:end,1),residuals_percentiles_5_95(:,1),4); %fitting => erstellt Polynomformel => Polynom 4. Ordnung
    x_linspace_partial = linspace(estimates_res_comb_sorted(31,1),max(model_estimates(:,i))); %linspace erst ab dem 10. (sortierten) model estimates Wert weil ja auch erst ab dann die Quantile berechnet wurden
    smoothfit_residuals_5_quantil = polyval(fit_res_quant5,x_linspace_partial); %Anwendung der Polynomformel auf die linspace-Werte gibt dann die smooth fit Werte welche mit linspace zusammen geplottet werden kann => gibt sooth line f�r 5er-Quantile im Plot
    
    % applying smooth fit on 95-percentiles
    fit_res_quant95 = polyfit(estimates_res_comb_sorted(31:end,1),residuals_percentiles_5_95(:,2),4); %fitting => erstellt Polynomformel => Polynom 4. Ordnung
    x_linspace_partial = linspace(estimates_res_comb_sorted(31,1),max(model_estimates(:,i))); %linspace erst ab dem 10. (sortierten) model estimates Wert weil ja auch erst ab dann die Quantile berechnet wurden
    smoothfit_residuals_95_quantil = polyval(fit_res_quant95,x_linspace_partial); %Anwendung der Polynomformel auf die linspace-Werte gibt dann die smooth fit Werte welche mit linspace zusammen geplottet werden kann => gibt sooth line f�r 5er-Quantile im Plot
    
    % moving average on all residuals
    fit_res = polyfit(estimates_res_comb_sorted(:,1),estimates_res_comb_sorted(:,2),3); %fitting => erstellt Polynomformel => Polynom 3. Ordnung
    x_linspace = linspace(min(model_estimates(:,i)),max(model_estimates(:,i))); %linspace �ber das Spektrum aller model estimates Werte
    smoothfit_residuals = polyval(fit_res,x_linspace);
    
    
    % plotting of residuals against model estimates
    figure (2+6*(i-1));
    plot(model_estimates(:,i),residuals(:,i),'Marker','o','Markersize',3,'LineStyle','none')

    % adding horizontal line at residuals = 0
    line_zero = line([min(xlim), max(xlim)-10^-10],[0,0]);
    line_zero.Color =  [0.5 0.5 0.5];
    line_zero.LineStyle = '- -';
    line_zero.LineWidth = 0.5;
    
    % plotting of the fitting lines
    hold on;
    plot(x_linspace,smoothfit_residuals,'b');
    plot(x_linspace_partial,smoothfit_residuals_5_quantil,'color',[0.9, 0.6, 0]);
    plot(x_linspace_partial,smoothfit_residuals_95_quantil,'color',[0.9, 0.6, 0]);
    
    % set axis-names:
    xlabel("model estimates [mm]",'FontSize',11);
    ylabel("residuals [mm]",'FontSize',11);
   
    % set plot title
    %title("model estimates vs. residuals",'FontSize',12,'FontWeight','bold');
    
    % moving of title, x- and y-axis label
    hLabel = get(gca,'xlabel'); 
    set(hLabel, 'Position', get(hLabel, 'Position') - [0 .005 0]); 
    set(hLabel, 'Units', 'pixels');
    
    hLabel = get(gca,'ylabel'); 
    set(hLabel, 'Position', get(hLabel, 'Position') - [0.01 0 0]); 
    set(hLabel, 'Units', 'pixels');
    
    %hLabel = get(gca,'title'); 
    %set(hLabel, 'Position', get(hLabel, 'Position') + [0 0.02 0]); 
    %set(hLabel, 'Units', 'pixels');
    

    
    %3) Plot estimated values vs. studentized residuals plot
    
    % computation of studentized residuals (https://ch.mathworks.com/help/stats/regression-and-anova.html)
    for j = 1:size(residuals(:,i),1)
        residuals_stud = residuals(:,i);
        model_estimates_stud = model_estimates(:,i);
        
        residuals_stud(j)=[];
        model_estimates_stud(j) = [];
        MSE_stud(j) = immse(residuals_stud,model_estimates_stud);
    
        residuals_studentized(j,i) = residuals(j,i) / ((MSE_stud(j)*(1-ezg_leverage(j)))^(1/2)); %computation of r-th studentized residual with following formula: https://ch.mathworks.com/help/stats/residuals.html (Mitte)
    end
    
    % plotting of studentized residuals against model estimates
    figure (3+6*(i-1));
    plot(model_estimates(:,i),residuals_studentized(:,i),'Marker','o','Markersize',3,'LineStyle','none')

    % adding horizontal line at studentized residuals = 0
    line_zero = line([min(xlim), max(xlim)-10^-10],[0,0]);
    line_zero.LineStyle = '- -';
    line_zero.LineWidth = 0.5;
    
    % set axis-names:
    xlabel("model estimates [mm]",'FontSize',11);
    ylabel("studentized residuals [-]",'FontSize',11);
   
    % set plot title
    %title("model estimates vs. studentized residuals",'FontSize',12,'FontWeight','bold');
    
    % moving of title, x- and y-axis label
    hLabel = get(gca,'xlabel'); 
    set(hLabel, 'Position', get(hLabel, 'Position') - [0 .01 0]); 
    set(hLabel, 'Units', 'pixels');
    
    hLabel = get(gca,'ylabel'); 
    set(hLabel, 'Position', get(hLabel, 'Position') - [0.01 0 0]); 
    set(hLabel, 'Units', 'pixels');
    
    %hLabel = get(gca,'title'); 
    %set(hLabel, 'Position', get(hLabel, 'Position') + [0 0.025 0]); 
    %set(hLabel, 'Units', 'pixels');
 
    
    
    %4) individual plots of every single variable against the residuals

    % generating variable names vector without response name
    variable_names_without_response = variable_names;
    variable_names_without_response(col_num_response) = [];
    
    if size(ezg_variables_as_matrix_no_interactions,2) < 101  %if plot number (thereby variable number) does not surpass 100, the plot specifics remain constant
        plotheight = 60;
        plotwidth = 75;
    else %otherwise they change depending on the number of plots/variables
        plotheight = 6*ceil(size(ezg_variables_as_matrix_no_interactions,2)*0.1);
        plotwidth = 75;
    end
    
    
    subplotsx = 10; %there shall always be 10 subplots per horizontal "plot-line"
    if size(ezg_variables_as_matrix_no_interactions,2) < 101  %if plot number (thereby variable number) does not surpass 100, the number of horziontal lines (subplotsy) is set to =10. Thereby all figures with subplot-numbers below 101 will have the same size (even if not all rows might be filled with subplots)
        subplotsy = 10;
    else
        subplotsy = ceil(size(ezg_variables_as_matrix_no_interactions,2)*0.1); %otherwise the number of horizotal lines depends of how many "lines" of 10 subplots can be generated based on the number of input variables (e.g. 112 variables give 11 complete lines (11*10) and one horizontal line consisting the remaining two subplots (=> subplotsy=12))
    end

    % spaces at the edges of the whole figure
    leftedge = 3;
    rightedge = 2;
    topedge = 2;
    bottomedge = 2;
    
    % horizontal and vertical spaces between subplots
    spacex = 0.6;
    spacey = 1.8;
    
    fontsize=25;


    % setting the Matlab figure
    f=figure('visible','off'); %visible off macht dass die figure(s) nicht automatisch hier angezeigt wird/werden nach der (jeweiligen) Erstellung
    clf(f); % clear current figure
    figure(4+6*(i-1));
    set(gcf, 'PaperUnits', 'centimeters');
    set(gcf, 'PaperSize', [plotwidth plotheight]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperPosition', [0 0 plotwidth plotheight]);
 

    % defining of subplot positions   
    subxsize=(plotwidth-leftedge-rightedge-spacex*(subplotsx-1.0))/subplotsx;
    subysize=(plotheight-topedge-bottomedge-spacey*(subplotsy-1.0))/subplotsy;

    %stepwise going through all variables and generating their residual
    %plot. Always along the horizontal until number of ten subplots is
    %reached, then on to the next horizontal line right below (ensured by the following if-statements)
    subplotsy = ceil(size(ezg_variables_as_matrix_no_interactions,2)*0.1);
    
    for j=1:subplotsy
     
        if size(ezg_variables_as_matrix_no_interactions,2) < j*10
            subplotsx = size(ezg_variables_as_matrix_no_interactions,2)-(subplotsy-1)*10;
        end
     
        if size(ezg_variables_as_matrix_no_interactions,2) > j*10
            subplotsx = 10;
        end
     
        for k=1:subplotsx   
            
            %xfirst and yfirst define the down-left corner point of the currently considered subplot
            xfirst=leftedge+(k-1.0)*(subxsize+spacex);
            yfirst=plotheight-topedge-subysize-(j-1.0)*(subysize+spacey);
      
            sub_pos{k,j}=[xfirst/plotwidth yfirst/plotheight subxsize/plotwidth subysize/plotheight]; %setting the whole (relative) measures of the subplot

        end
    end
   
    for j=1:subplotsy %stepwise going through all subplots with described positioning
     
        if size(ezg_variables_as_matrix_no_interactions,2) < j*10
             subplotsx = size(ezg_variables_as_matrix_no_interactions,2)-(subplotsy-1)*10; end
     
        if size(ezg_variables,2) > j*10
             subplotsx = 10; end
     
        for jj=1:subplotsx  
            num_plot = (j-1)*10 + jj; %"counts" which variable currently gets plottet
            
            % generating residual plot of currently considered variable
            ax=axes('position',sub_pos{jj,j},'XGrid','off','XMinorGrid','off','FontSize',fontsize,'Box','on','Layer','top');
            plot(ezg_variables_as_matrix_no_interactions(:,num_plot),residuals(:,i),'Marker','o','Markersize',2,'LineStyle','none')
            
            % set axis-names
            xlabel(strcat(sprintf(variable_names_without_response{num_plot}),{' [-]'}),'Interpreter', 'none','FontSize',22);
            if jj == 1
                ylabel("residuals [mm]",'FontSize',22);
            end
   
            if jj > 1
                set(ax,'yticklabel',[]);
                set(ax,'ytick',[]);
            end
            
        end
    end
    
    
    
    %5) Leverage values vs. studentized residuals plot
    
    figure(5+6*(i-1));
    plot(ezg_leverage,residuals_studentized(:,i),'Marker','o','Markersize',3,'LineStyle','none');
    
    %Einf�gen der waagrechten Linie bei residuals = 0
    line_zero = line([min(xlim), max(xlim)-10^-10],[0,0]);
    line_zero.Color =  [0.5 0.5 0.5];
    line_zero.LineStyle = '- -';
    line_zero.LineWidth = 0.5;
    
    % set axis-names:
    xlabel("leverage [-]",'FontSize',11);
    ylabel("studentized residuals [-]",'FontSize',11);
   
    %set plot title
    %title("leverage vs. studentized residuals",'FontSize',12,'FontWeight','bold');
    
    %Verschieben des  Titels ein bisschen nach oben, des x-Achsentitels ein bisschen nach unten und des y-Achsentitels ein bisschen nach lniks
    hLabel = get(gca,'xlabel'); 
    %set(hLabel, 'Position', get(hLabel, 'Position') - [0 .0 0]); 
    set(hLabel, 'Units', 'pixels');
    
    hLabel = get(gca,'ylabel'); 
    set(hLabel, 'Position', get(hLabel, 'Position') - [0.01 0 0]); 
    set(hLabel, 'Units', 'pixels');
    
    %hLabel = get(gca,'title'); 
    %set(hLabel, 'Position', get(hLabel, 'Position') + [0 0.005 0]); 
    %set(hLabel, 'Units', 'pixels');
    
    
    
    
    %5) Leverage values vs. studentized residuals plot
    
    % plotting leverage values against studentized residuals
    figure(5+6*(i-1));
    plot(ezg_leverage,residuals_studentized(:,i),'Marker','o','Markersize',3,'LineStyle','none');

    % adding horizontal line at studentized residuals = 0
    line_zero = line([min(xlim), max(xlim)-10^-10],[0,0]);
    line_zero.Color =  [0.5 0.5 0.5];
    line_zero.LineStyle = '- -';
    line_zero.LineWidth = 0.5;
    
    % set axis-names
    xlabel("leverage [-]",'FontSize',11);
    ylabel("studentized residuals [-]",'FontSize',11);
   
    % set plot title
    title("leverage vs. studentized residuals",'FontSize',12,'FontWeight','bold');
    
    % moving of title and axis labels
    hLabel = get(gca,'xlabel'); 
    set(hLabel, 'Units', 'pixels');
    
    hLabel = get(gca,'ylabel'); 
    set(hLabel, 'Position', get(hLabel, 'Position') - [0.01 0 0]); 
    set(hLabel, 'Units', 'pixels');
    
    hLabel = get(gca,'title'); 
    set(hLabel, 'Position', get(hLabel, 'Position') + [0 0.005 0]); 
    set(hLabel, 'Units', 'pixels');
    
    
    
    
    %6) catchment gridcodes (or variable row index) against leverage
    %values
    if i==1 %does only have to get done once (and not for log-transformed response model as well as it would be the exact same plot)
    
    %plotting of leverage values against row-numbers (in the input matrix) of considered
    %catchments
    figure(6);
    plot(1:size(ezg_leverage,1),ezg_leverage,'Marker','o','Markersize',3,'LineStyle','none')
    xlim([0 size(ezg_leverage,1)+1]);
 
    %adding horizontal line at mean of leverage values
    %(value always = (No. variables + 1) / No. observations)
    line_mean = line([min(xlim), max(xlim)-10^-10],[mean(ezg_leverage),mean(ezg_leverage)]);
    line_mean.Color =  [0.5 0.5 0.5];
    line_mean.LineStyle = '- -';
    line_mean.LineWidth = 0.5;
    
    % set axis-names:
    xlabel("row index catchment [-]",'FontSize',11);
    ylabel("leverage [-]",'FontSize',11);
   
    % set plot title
    title("catchment observations vs. leverage",'FontSize',12,'FontWeight','bold');
    
    % moving of title and axis labels
    hLabel = get(gca,'xlabel');  
    set(hLabel, 'Units', 'pixels');
    
    hLabel = get(gca,'ylabel'); 
    set(hLabel, 'Position', get(hLabel, 'Position') - [0.01 0 0]); 
    set(hLabel, 'Units', 'pixels');
    
    hLabel = get(gca,'title'); 
    set(hLabel, 'Position', get(hLabel, 'Position') + [0 0.005 0]); 
    set(hLabel, 'Units', 'pixels');
    
    end
    
    
    
    %export of validation plots
    
    if ~exist(['all\/outputs\'] ) mkdir(['all\outputs\']); %controls if folder for outputs exists, if not, one gets made
    end
    
    if ~exist(['all\outputs\sequentialfs_validation_plots\'] ) mkdir(['all\outputs\sequentialfs_validation_plots\']);  %controls if folder for validation plots within output-folder exists, if not, one gets made
    end
    
    if ~exist(fullfile('all/outputs/sequentialfs_validation_plots',strrep(filename,".txt",""))) mkdir(fullfile('all/outputs/sequentialfs_validation_plots',strrep(filename,".txt",""))); %controls if folder for validation plots within output-folder exists, if not, one gets made
    end
    
    for j = 1:6
    
    if i==1
        if j==1
        plot_name = sprintf('normal_probability_plot_%d_interval_%d',K,L); end
        if j==2
        plot_name = sprintf('residualplot_all_%d_interval_%d',K,L); end
        if j==3
        plot_name = sprintf('studentized_residualplot_all_%d_interval_%d',K,L); end
        if j==4
        plot_name = sprintf('residualplot_individual_%d_interval_%d',K,L); end
        if j==5
        plot_name = sprintf('leverage_vs_residualplot_%d_interval_%d',K,L); end
        if j==6
        plot_name = sprintf('leverageplot_%d_interval_%d',K,L); end
    end
    if i==2
        if j==1
        plot_name = sprintf('normal_probability_plot_log_q347_%d_interval_%d',K,L); end
        if j==2
        plot_name = sprintf('residualplot_all_log_q347_%d_interval_%d',K,L); end
        if j==3
        plot_name = sprintf('studentized_residualplot_all_log_q347_%d_interval_%d',K,L); end
        if j==4
        plot_name = sprintf('residualplot_individual_log_q347_%d_interval_%d',K,L); end
        if j==5
        plot_name = sprintf('leverage_vs_residualplot_log_q347_%d_interval_%d',K,L); end    
    end


    plot_path = fullfile('all/outputs/sequentialfs_validation_plots',strrep(filename,".txt",""),plot_name);
    
    if j*i < 12
    print(plot_path,'-deps');
    set(figure((i-1)*6+j),'color','w');
    set(figure((i-1)*6+j), 'InvertHardCopy', 'off');
    print('-loose','-dpng',plot_path,'-r400') ; %save figure as png
    print(gcf, '-depsc2','-loose',[plot_path,'.eps']);
    end
    
    end
    
    end
    

    if I_mdl_results_export == 1
    %EXCEL-EXPORT OF RESULTS
    
    outFile = strrep(strrep(filename,"InputDaten","results_"),".txt",".xls");
   
    if i==1
        if ~exist(['all/outputs/sequential'] ) mkdir(['all/outputs/sequential']); end
        excel_path = fullfile('all/outputs/sequential',outFile);
    end
    if i == size(response_types,2)
        
        xlwrite(excel_path,{'response variable','mean observation value',' ','MSE','RMSE','MAE','R^2'},'stats','B1');
        xlwrite(excel_path,regression_output(:,1:2),'stats','B2');
        xlwrite(excel_path,regression_output(:,3:end),'stats','E2');

        for n=1:count
            xlwrite(excel_path,{sprintf('model %d',n)},'stats',sprintf('A%d',2*n));
            xlwrite(excel_path,{'cross validated';'fit on all data'},'stats',sprintf('D%d',n*2));           
        end
    end
    % creating excel-file and inserting titles and accordingly (computed) values
    xlwrite(excel_path,{'formula'},sprintf('stats_model_%d',count),'A1');
    xlwrite(excel_path,strcat(mdl.Formula.ResponseName,{'  ~  '},mdl.Formula.LinearPredictor),sprintf('stats_model_%d',count),'A3');
    xlwrite(excel_path,{'predictors statistics'},sprintf('stats_model_%d',count),'A6');
    xlwrite(excel_path,mdl.Coefficients.Properties.VariableNames,sprintf('stats_model_%d',count),'B8');
    xlwrite(excel_path,mdl.Coefficients.Properties.RowNames,sprintf('stats_model_%d',count),'A9');
    xlwrite(excel_path,table2array(mdl.Coefficients),sprintf('stats_model_%d',count),'B9');
    xlwrite(excel_path,{'model estimations'},sprintf('stats_model_%d',count),'G6');
    xlwrite(excel_path,{'observation values'},sprintf('stats_model_%d',count),'G8');
    xlwrite(excel_path,{'estimated values'},sprintf('stats_model_%d',count),'H8');
    xlwrite(excel_path,{'residuals'},sprintf('stats_model_%d',count),'I8');
    xlwrite(excel_path,response_types_as_matrix(:,1),sprintf('stats_model_%d',count),'G9');
    xlwrite(excel_path,model_estimates(:,i),sprintf('stats_model_%d',count),'H9');
    xlwrite(excel_path,residuals(:,i),sprintf('stats_model_%d',count),'I9');
    
    
   
    
end

end



