
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% application of regression procedure with lasso principle %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%NEEDED INPUTS:
%input_data: values of the different variables over the different catchments
%variable_names: names of the input variables (including response)
%name_response: name of the response variable
%col_num_response: column number of response in the input-matrix (textfile)
%I_response_log: decision if with log-transformed response variable
%I_mdl_validation_export: decision if export of validation plots
%I_mdl_results_export: decision if export of model outputs
%I_mdl_interactions: decision if consideration of interaction terms
%categorical: column numbers of categorical variables in input-matrix
%defined_categories: defined boundaries between categories of categorical variables
%K: number of input-textfile (textfile in folder with data)
%L: number of cluster-interval

%VARIABLE RETURN:
%no return

%OUTPUTS (if chosen):
%model estimates, residuals, goodness-of-fit measures and more
%model validation plots

function [] = randomForest_model(input_data,variable_names,name_response,col_num_response,I_response_log,I_mdl_validation_export,I_mdl_results_export,categorical,defined_categories,I_mdl_interactions,K,L,filename)
    %remove NaN-rows
    input_data(any(isnan(input_data), 2), :) = [];

    %defining untransformed response variable
    input_raw = input_data(:,col_num_response); %assigning column of input matrix with response data
    response_types = input_raw;
    response_names = {name_response};

    %defining log-transformed response variable
    if I_response_log == 1
        input_nonzero = input_raw+0.00001; %log-transformation not possible with zero values, therefore +0.00001 to all response-values
        response_log = log10(input_nonzero); %log-transformation
        response_types = horzcat(response_types, response_log); %merging of columns of untransformed and log-transformed response-values in vector-form
        response_names = [response_names {strcat(name_response,'_log')}]; %merging of untransformed response name with name of log-transformed response name (same name with "_log" at the end)
    end


    % DEFINING CATEGORICAL VARIABLES

    input_stand = input_data;

    num_added_columns = 0;
    variable_names_new = variable_names;

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

    % initializing vector
    categorical_variable_names = [];

    %transforming the as categorical labelled variables into categorical variables
    %according to the defined boundaries (or the default boundaries if none
    %were defined). E.g. a variable with three assigned categories will result
    %in three categorical variables with 0/1-values, depending on if the
    %respective observation value lies within the respective category (1) or
    %not (0).
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


    %generating the categorical variables based on the assigned categorical
    %values (e.g. an as categorical defined variable consisting of the
    %assigned values 1,2,3,4 (for four categories) gets transformed into
    %four categorical variables with 0/1-values

    input_cat = input_stand;
    for j = 1:size(categorical,2)
        for l = 1:max(input_cat(:,categorical(j)+num_added_columns))

            logical_vector(:,l) = input_cat(:,categorical(j)+num_added_columns) == l;
            logical_variable_name{l} = strcat(variable_names{categorical(j)},'_',num2str(l)); %Namenserstellung der jeweiligen Kategorie l der categorical variable in column "categorical(j)+num_added_columns". Der Name setzt sich zusammen aus dem Variablennamen und dann via Bodenstrich verbunden die Kategoriennummer l (z.B. "P_MEAN_2" (falls P_MEAN als categorical variable definiert worden w�re))

        end

        num_added_columns_new = num_added_columns + max(input_cat(:,categorical(j)+num_added_columns))-1;

        input_cat = [input_cat(:,1:(categorical(j)+num_added_columns-1)) logical_vector input_cat(:,(categorical(j)+num_added_columns+1):end)];
        variable_names_new = [variable_names_new(:,1:(categorical(j)+num_added_columns-1)) logical_variable_name variable_names_new(:,(categorical(j)+num_added_columns+1):end)];
        num_added_columns = num_added_columns_new; 

        logical_vector = [];
        logical_variable_name = [];
    end

    %defining (potential) predictor variables
    ezg_variables = input_cat;
    ezg_variables(:,col_num_response) = []; 
    predictor_names = variable_names_new;
    predictor_names(:,col_num_response) = [];
    
    % RANDDOM FOREST REGRESSION WITH 10-FOLD CROSS-VALIDATION

    count = 0;

    %Now the random forest principle gets applied. If also the log-transformed
    %response is considered (size(response_types,2)=2), then the whole following 
    %procedures will be applied twice, once for the untransformed response and 
    %once for the log-transformed one
    x = size(response_types,2);
    
    % remove oultiers
    Mdl = TreeBagger(1000,ezg_variables,response_types(:,1),...
        'Method','regression',...
        'OOBPredictorImportance','on',...
        'PredictorSelection','curvature',...
        'Predictornames',predictor_names);
    Mdl = fillProximities(Mdl);
    outlierIdx = find((Mdl.OutlierMeasure>=3) == 1);
    
    if (length(outlierIdx) > 0)
        response_types(outlierIdx,:) = [];
        ezg_variables(outlierIdx,:) = [];
    end

    if I_mdl_interactions == 1
        %if chosen, adding interaction terms to predictor data and predictor names

        for j = 1:(size(ezg_variables,2)-1)
            for k = (j+1):size(ezg_variables,2)
                predictor_names = [predictor_names {strcat(predictor_names{j},'_',predictor_names{k})}];
            end
        end

        ezg_variables = x2fx(ezg_variables,'interaction');
        ezg_variables(:,1) = [];
    end

    % removing zero-columns
    cols_with_all_zeros = find(all(ezg_variables==0));
    ezg_variables(:,cols_with_all_zeros) = [];
    predictor_names(cols_with_all_zeros) = [];
    
    % 10-FOLD CROSS-VALIDATION OF LASSO-PROCEDURE
    F = 10; %F=10 because 10-fold CV
    cv = cvpartition(size(response_types,1), 'kfold',F); %ten times partitioning of the data in nine training data folds and one testing data fold
    mse = zeros(x);
    mae = zeros(x);
    rsquared = zeros(x); %initialising vector for cross-validated MSEs for every testing fold
    importance = cell(x);
    trees = cell(x);
    
    % obtain best hyperparameters
    maxMinLS = 5;
    minLS = optimizableVariable('minLS',[1,maxMinLS],'Type','integer');
    numPTS = optimizableVariable('numPTS',[1,size(ezg_variables,2)-1],'Type','integer');
    hyperparametersRF = [minLS; numPTS];
    oobErr = @(hparams) oobQuantileError(TreeBagger(1000,ezg_variables,response_types(:,1),...
                                                      'Method','regression',...
                                                      'OOBPrediction','on',...
                                                      'PredictorSelection','curvature',...
                                                      'MinLeafSize',hparams.minLS,...
                                                      'NumPredictorstoSample',hparams.numPTS));
                                                  
    results = bayesopt(oobErr,hyperparametersRF,...
                        'AcquisitionFunctionName','expected-improvement-plus',...
                        'Verbose',1);
                    
    bestOOBErr = results.MinObjective;
    bestHyperparameters = results.XAtMinObjective;
    
    % write hyperparameters to file
    formatSpec = '%s\t%d\t%d\t%d\n';
    outData = {filename;bestHyperparameters.minLS;bestHyperparameters.numPTS;outlierIdx};
    if exist('rf_parameters.txt', 'file') == 0
        fid = fopen('rf_parameters.txt', 'wt+');
        fprintf(fid, 'file\tminLeaveSize\tnumPredictorsToSample\toutlierIdx\n');
        fclose(fid);
    end
    fid = fopen('rf_parameters.txt', 'a+');
    fprintf(fid, formatSpec, outData{:,1});
    fclose(fid);
        
    for i = 1:x
        
        count = count + 1;     
        
        for f = 1:F %random forest principle gets applied ten times (F=10), once for every new constellation of training data and testing data
            % training/testing indices for this fold
            trainIdx = cv.training(f);
            testIdx = cv.test(f);
            
            % application of random forest on training data
            %[coeffs,FitInfo] 
            Mdl = TreeBagger(1000,ezg_variables(trainIdx,:),response_types(trainIdx,i),...
                'Method','regression',...
                'OOBPredictorImportance','on',...
                'PredictorSelection','interaction-curvature',...
                'Predictornames',predictor_names,...
                'MinLeafSize',bestHyperparameters.minLS,...
                'NumPredictorstoSample',bestHyperparameters.numPTS);
            
            %{ 
            %plot decision tree
            view(Mdl.Trees{1},'Mode','graph')

            % plot out of bag error
            figure;
            oobErrorBaggedEnsemble = oobError(Mdl);
            plot(oobErrorBaggedEnsemble)
            xlabel 'Number of grown trees';
            ylabel 'Out-of-bag classification error';
            %}

            %computation of mean squared error of trees using test data
            err = error(Mdl,ezg_variables(testIdx,:),response_types(testIdx,i));
            mse(f,i) = min(err);

            % extract importance of variables and fitted tree models
            importance{f,i} = Mdl.OOBPermutedPredictorDeltaError;
            trees{f,i} = Mdl;

            % get R2 and RMSE using test data
            pred = Mdl.predict(ezg_variables(testIdx,:));        

            % compute cross-validated measures of error and measure of correlation
            % mean squared error
            residuals(f,i) = mean([response_types(testIdx,i) - pred]);
            mse(f,i) = residuals(f,i).^2;

            % mean absolute error
            mae(f,i) = mean([abs(response_types(testIdx,i) - pred)]);

            % explained variance
            cv_residuals_squared = (response_types(testIdx,i) - pred).^2;
            cv_observation_mean_differences_squared = (response_types(testIdx,1) - mean(response_types(testIdx,1))).^2; %"1" instead of "i" as also for the log-transformed response you want to refer to the untransformed response values as you transform back (see above)

            cv_RSS = sum(cv_residuals_squared);
            cv_TSS = sum(cv_observation_mean_differences_squared);
            rsquared(f,i) = 1 - cv_RSS/cv_TSS;     
        end 

        % compute cross-validated average measures of error across k-folds
        % average MSE
        CV_MSE(i) = mean(mse(:,i));

        % average RMSE
        CV_RMSE(i) = sqrt(CV_MSE(i));

        % average MAE
        CV_MAE(i) = mean(mae(:,i));

        % average R^2
        CV_Rsquared(i) = mean(rsquared(:,i));
        
        % export mean importance of predictor variables
        if ~exist(['rf_importance'] )
            mkdir(['rf_importance']);
        end
        
        for row=1:length(predictor_names)
            meanPred(row) = mean(cellfun(@(x) x(row),importance(:,i)));
        end    
        outPred = [predictor_names;num2cell(meanPred)];
        %writetable(cell2table(outPred'),'outputs/meanImportancePredictors.txt')
        outFile = strcat('rf_importance/meanImportancePredictors_log_',string(i-1),'_',filename);
        fileID = fopen(outFile,'wt'); % create file and write to it
        formatSpec = '%s\t%1.4f\n';
        [nrows,ncols] = size(outPred);
        for col = 1:ncols
            fprintf(fileID,formatSpec,outPred{:,col});
        end
        fclose(fileID);
        
        % application of random forest on whole data
        pred = cell(x);
        residuals_all = cell(x);
  
        Mdl = TreeBagger(300,ezg_variables,response_types(:,i),...
            'Method','regression',...
            'OOBPredictorImportance','on',...
            'PredictorSelection','curvature',...
            'Predictornames',predictor_names,...
            'MinLeafSize',bestHyperparameters.minLS,...
            'NumPredictorstoSample',bestHyperparameters.numPTS);
        
        err = error(Mdl,ezg_variables,response_types(:,i));
        mse(i) = min(err);

        % extract importance of variables and fitted tree models
        importance_all = Mdl.OOBPermutedPredictorDeltaError;

        % get R2 and RMSE using test data
        pred{i} = predict(Mdl,ezg_variables);
        
        if(i == 2)
            pred{i} =  10.^pred{i};
        end
        
        % compute cross-validated measures of error and measure of correlation
        % mean squared error
        residuals_all{i} = response_types(:,1) - pred{i};

        % mean absolute error
        mae(i) = mean([abs(response_types(:,i) - pred{i})]);

        % explained variance
        cv_residuals_squared = (response_types(:,i) - pred{i}).^2;
        cv_observation_mean_differences_squared = (response_types(:,1) - mean(response_types(:,1))).^2; %"1" instead of "i" as also for the log-transformed response you want to refer to the untransformed response values as you transform back (see above)

        cv_RSS = sum(cv_residuals_squared);
        cv_TSS = sum(cv_observation_mean_differences_squared);
        rsquared(i) = 1 - cv_RSS/cv_TSS;     
        
        rmse(i) = sqrt(mse(i));
        
        % output formatting
        regression_output(count*2-1,:)=[response_names(i),CV_MSE(i),CV_RMSE(i),CV_MAE(i),CV_Rsquared(i)];
        regression_output(count*2,:)=[response_names(i),mse(i),rmse(i),mae(i),rsquared(i)];
        
        if I_mdl_results_export == 1
            
            outFile = strrep(strrep(filename,"InputDaten","results_"),".txt",".xls");

            if i==1
                if ~exist(['outputs/random_forest'] ) 
                    mkdir(['outputs/random_forest']); 
                end
                excel_path = fullfile('outputs/random_forest',outFile);
            end
            
            if i == size(response_types,2)
        
                xlwrite(excel_path,{'response variable',' ','MSE','RMSE','MAE','R^2'},'stats','B1');
                xlwrite(excel_path,{name_response{:}},'stats','B2');
                xlwrite(excel_path,regression_output(:,2:end),'stats','E2');
                
                for n=1:count
                    xlwrite(excel_path,{sprintf('model %d',n)},'stats',sprintf('A%d',2*n));
                    xlwrite(excel_path,{'cross validated';'fit on all data'},'stats',sprintf('D%d',n*2));
                end
            end 
            
            %EXCEL-EXPORT OF RESULTS
            % creating excel-file and inserting titles and accordingly (computed) values
            xlwrite(excel_path,{'predictors_statistics'},sprintf('stats_model_%d',count),'A6');
            xlwrite(excel_path,{'predictors'},sprintf('stats_model_%d',count),'A8');
            xlwrite(excel_path,{'relative_importance'},sprintf('stats_model_%d',count),'B8');
            xlwrite(excel_path,predictor_names',sprintf('stats_model_%d',count),'A9');
            xlwrite(excel_path,importance_all',sprintf('stats_model_%d',count),'B9');
            
            xlwrite(excel_path,{'model_estimations'},sprintf('stats_model_%d',count),'G6');
            xlwrite(excel_path,{'observation values'},sprintf('stats_model_%d',count),'G8');
            xlwrite(excel_path,{'estimated values'},sprintf('stats_model_%d',count),'H8');
            xlwrite(excel_path,{'residuals'},sprintf('stats_model_%d',count),'I8');
            xlwrite(excel_path,response_types(:,1),sprintf('stats_model_%d',count),'G9');
            xlwrite(excel_path,pred{i},sprintf('stats_model_%d',count),'H9');
            xlwrite(excel_path,residuals_all{i},sprintf('stats_model_%d',count),'I9');  
        end 
end