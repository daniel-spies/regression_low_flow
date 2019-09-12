
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


function [] = lasso_model(input_data,variable_names,name_response,col_num_response,I_response_log,I_mdl_validation_export,I_mdl_results_export,categorical,defined_categories,I_mdl_interactions,K,L,filename)

    sprintf("lasso processing file: %s",filename)

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


    %needed later for individual residual plot (see further below)
    ezg_variables_as_matrix = input_stand;
    ezg_variables_as_matrix(:,no_categoricals) = zscore(ezg_variables_as_matrix(:,no_categoricals));
    ezg_variables_as_matrix(:,col_num_response) = [];

    %defining (potential) predictor variables
    ezg_variables = input_cat;
    ezg_variables(:,col_num_response) = []; 
    predictor_names = variable_names_new;
    predictor_names(:,col_num_response) = [];


    %computing the leverage of the different catchments based on the
    %standardized variables values
    ezg_leverage = leverage(input_stand);


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


    % LASSO REGRESSION WITH 10-FOLD CROSS-VALIDATION

    count = 0;



    %Now the lasso principle gets applied. If also the log-transformed
    %response is considered (size(response_types,2)=2), then the whole following 
    %procedures will be applied twice, once for the untransformed response and 
    %once for the log-transformed one
    for i = 1:size(response_types,2)

        count=count+1;

        % 10-FOLD CROSS-VALIDATION OF LASSO-PROCEDURE

        F = 10; %F=10 because 10-fold CV
        cv = cvpartition(size(response_types,1), 'kfold',F); %ten times partitioning of the data in nine training data folds and one testing data fold
        mse = zeros(F,1); %initialising vector for cross-validated MSEs for every testing fold

        for f=1:F %lasso principle gets applied ten times (F=10), once for every new constellation of training data and testing data
            % training/testing indices for this fold
            trainIdx = cv.training(f);
            testIdx = cv.test(f);

            % application of lasso-model on training data
            [coeffs,FitInfo] = lasso(ezg_variables(trainIdx,:),response_types(trainIdx,i),'CV',10,'PredictorNames',predictor_names); %input variables do not need to be standardized beforehand as this gets done within the function

            %computation of model estimates and residuals

            coeffs_minMSE = coeffs(:,FitInfo.IndexMinMSE); %assigning the calibrated coefficient values...
            intercept_minMSE = FitInfo.Intercept(FitInfo.IndexMinMSE); %... and intercept value

            ezg_variables_testdata = ezg_variables(testIdx,:);
            Y_hat = []; %has to be set zero first at each round

            % predict regression output (%just on training data fitted regression model
            % gets used to estimate the q347-values based on the testing data)
            for j = 1:size(ezg_variables_testdata,1)
                Y_hat(j,1) = intercept_minMSE + sum(coeffs_minMSE.*transpose(ezg_variables_testdata(j,:))); %using assigned intercept and coefficent values to generate estimates of the response
            end      


            % compute cross-validated measures of error and measure of correlation

            % mean squared error
            mse(f,i) = mean((response_types(testIdx,i) - Y_hat).^2);

            % mean absolute error
            mae(f,i) = mean(abs(response_types(testIdx,i) - Y_hat));

            % explained variance
            cv_residuals_squared = (response_types(testIdx,i) - Y_hat).^2;
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


        % APPLYING LASSO-MODEL ON ALL DATA

        %now the same regression procedure as before gets repeated, but this
        %time the model gets fit and tested as well on all data. Thereby, the
        %data used for model fitting and model testing is the same

        %applying the lasso regression
        [coeffs,FitInfo] = lasso(ezg_variables,response_types(:,i),'CV',10,'PredictorNames',predictor_names); %input variables do not need to be standardized beforehand as this gets done within the function

        %computation of model estimates and residuals

        coeffs_minMSE = coeffs(:,FitInfo.IndexMinMSE);
        intercept_minMSE = FitInfo.Intercept(FitInfo.IndexMinMSE);
        observation_mean(i) = mean(response_types(:,i));

        for j = 1:size(ezg_variables,1)
            model_estimates(j,i) = intercept_minMSE + sum(coeffs_minMSE.*transpose(ezg_variables(j,:)));
        end

        % computing 10^(estimate-value) with estimates based on log-transformed
        % response to get estimates of untransformed response
        if i==2
            model_estimates(:,i) = 10.^(model_estimates(:,i));
        end

        residuals(:,i) = response_types(:,1) - model_estimates(:,i);
        residuals_squared(:,i) = residuals(:,i).^2;
        observation_mean_differences_squared(:,i) = (response_types(:,1) - observation_mean(1)).^2; %"1" instead of "i" as also for the log-transformed response you want to refer to the untransformed response values as you transform back (see above)


        %computation of measures of error and measures of correlation for fit on all data

        % mean squared error
        MSE(i) = mean(residuals_squared(:,i)); %diese gesch�tzten Werte werden dann von den jeweils tats�chlichen response values abgezogen, dann das Resultat jeweils quadriert und davon den Durchnitt genommen, was somit schlussendlich den MSE des k-folds gibt, basierend auf der Modellsch�tzung mit der jeweiligen training data und dann Anwendung auf die test data

        % root mean squared error
        RMSE(i) = sqrt(MSE(i));

        % mean absolute error
        MAE(i) = mean(abs(residuals(:,i)));

        % explained variance
        RSS(i) = sum(residuals_squared(:,i));
        TSS(i) = sum(observation_mean_differences_squared(:,i));
        Rsquared(i) = 1 - RSS(i)/TSS(i);


        % summarizing all measures in one vector

        regression_output(count*2-1,:)=[response_names(i),observation_mean(i),CV_MSE(i),CV_RMSE(i),CV_MAE(i),CV_Rsquared(i)];
        regression_output(count*2,:)=[response_names(i),observation_mean(i),MSE(i),RMSE(i),MAE(i),Rsquared(i)];



        if I_mdl_validation_export == 1
        %GENERATING PLOTS


            %1) cross-validation plot

            % plotting of the lambdas against the resulting MSEs based on 10-fold
            % cross-validation
            figure (2+7*(i-1))
            lassoPlot(coeffs,FitInfo,'PlotType','CV');
            legend('show') % Show legend

            % moving of title
            hLabel = get(gca,'title'); 
            set(hLabel, 'Position', get(hLabel, 'Position') + [0 0.1 0]); 
            set(hLabel, 'Units', 'pixels');



            %2) normal probability plot of residuals => https://ch.mathworks.com/help/stats/normplot.html

            % generating plot
            figure (2+7*(i-1));
            normplot(residuals(:,i))
            title(""); %�nderung des Titels

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



            %3) Plot estimated values vs. residuals with moving average of all
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
            figure (3+7*(i-1));
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



            %4) Plot estimated values vs. studentized residuals plot

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
            figure (4+7*(i-1));
            plot(model_estimates(:,i),residuals_studentized(:,i),'Marker','o','Markersize',3,'LineStyle','none')

            % adding horizontal line at studentized residuals = 0
            line_zero = line([min(xlim), max(xlim)-10^-10],[0,0]);
            line_zero.Color =  [0.5 0.5 0.5];
            line_zero.LineStyle = '- -';
            line_zero.LineWidth = 0.5;

            % set axis-names:
            xlabel("model estimates [mm]",'FontSize',11);
            ylabel("studentized residuals [-]",'FontSize',11);

            %set plot title
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



            %5) individual plots of every single variable against the residuals

            % generating variable names vector without response name
            variable_names_without_response = variable_names;
            variable_names_without_response(col_num_response) = [];

            if size(ezg_variables_as_matrix,2) < 101 %if plot number (thereby variable number) does not surpass 100, the plot specifics remain constant
                plotheight = 60;
                plotwidth = 75;
            else %otherwise they change depending on the number of plots/variables
                plotheight = 6*ceil(size(ezg_variables_as_matrix,2)*0.1);
                plotwidth = 75;
            end

            subplotsx = 10; %there shall always be 10 subplots per horizontal "plot-line"
            if size(ezg_variables_as_matrix,2) < 101  %if plot number (thereby variable number) does not surpass 100, the number of horziontal lines (subplotsy) is set to =10. Thereby all figures with subplot-numbers below 101 will have the same size (even if not all rows might be filled with subplots)
                subplotsy = 10;
            else
                subplotsy = ceil(size(ezg_variables_as_matrix,2)*0.1); %otherwise the number of horizotal lines depends of how many "lines" of 10 subplots can be generated based on the number of input variables (e.g. 112 variables give 11 complete lines (11*10) and one horizontal line consisting the remaining two subplots (=> subplotsy=12))
            end

            %spaces at the edges of the whole figure
            leftedge = 3;
            rightedge = 2;
            topedge = 2;
            bottomedge = 2;

            %horizontal and vertical spaces between subplots
            spacex = 0.6;
            spacey = 1.8;

            fontsize=25;


            %setting the Matlab figure
            f=figure('visible','off');
            clf(f);
            figure(5+7*(i-1));
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
            subplotsy = ceil(size(ezg_variables_as_matrix,2)*0.1); %der Plan ist dass pro y-Reihe 10 Plots geplottet werden. D.h. es gibt immer 10 Plots in x-Richtung. Diese Formel sorgt daf�r dass es entsprechend eine korrekte Anzahl in y-Richtung geht, d.h. wenn z.B. insgesamt 27 Plots betrachtet werden sollen, folgt daraus subplotsy=3 (weil 10+10+7 gibt drei y-Reihen). Jede Inputvariable (ezg_variable(:,2)) soll dabei einen eigenen Plot haben

            for j=1:subplotsy

                if size(ezg_variables_as_matrix,2) < j*10
                    subplotsx = size(ezg_variables_as_matrix,2)-(subplotsy-1)*10;
                end

                if size(ezg_variables_as_matrix,2) > j*10
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

                if size(ezg_variables_as_matrix,2) < j*10
                     subplotsx = size(ezg_variables_as_matrix,2)-(subplotsy-1)*10; end

                if size(ezg_variables_as_matrix,2) > j*10
                     subplotsx = 10; end

                for jj=1:subplotsx  
                    num_plot = (j-1)*10 + jj; %"counts" which variable currently gets plottet

                    %generating residual plot of currently considered variable
                    ax=axes('position',sub_pos{jj,j},'XGrid','off','XMinorGrid','off','FontSize',fontsize,'Box','on','Layer','top');
                    plot(ezg_variables_as_matrix(:,num_plot),residuals(:,i),'Marker','o','Markersize',2,'LineStyle','none')

                    % set axis-names:
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

            %6) Leverage values vs. studentized residuals plot

            % plotting leverage values against studentized residuals
            figure(6+7*(i-1));
            plot(ezg_leverage,residuals_studentized(:,i),'Marker','o','Markersize',3,'LineStyle','none');

            % adding horizontal line at studentized residuals = 0
            line_zero = line([min(xlim), max(xlim)-10^-10],[0,0]);
            line_zero.Color =  [0.5 0.5 0.5];
            line_zero.LineStyle = '- -';
            line_zero.LineWidth = 0.5;

            % set axis-names:
            xlabel("leverage [-]",'FontSize',11);
            ylabel("studentized residuals [-]",'FontSize',11);

            % set plot title
            %title("leverage vs. studentized residuals",'FontSize',12,'FontWeight','bold');

            % moving of title and axis labelshLabel = get(gca,'xlabel'); 
            set(hLabel, 'Units', 'pixels');

            hLabel = get(gca,'ylabel'); 
            set(hLabel, 'Position', get(hLabel, 'Position') - [0.01 0 0]); 
            set(hLabel, 'Units', 'pixels');

            %hLabel = get(gca,'title'); 
            %set(hLabel, 'Position', get(hLabel, 'Position') + [0 0.005 0]); 

            %7) catchment gridcodes (or variable row index) against leverage
            %values
            if i==1 %does only have to get done once (and not for log-transformed response model as well as it would be the exact same plot)
                % plotting of leverage values against row-numbers (in the input matrix) of considered
                % catchments
                figure (7);
                plot(1:size(ezg_leverage,1),ezg_leverage,'Marker','o','Markersize',3,'LineStyle','none')
                xlim([0 size(ezg_leverage,1)+1]);

                % adding horizontal line at mean of leverage values
                %(value always = (No. variables + 1) / No. observations)
                line_mean = line([min(xlim), max(xlim)-10^-10],[mean(ezg_leverage),mean(ezg_leverage)]);
                line_mean.Color =  [0.5 0.5 0.5];
                line_mean.LineStyle = '- -';
                line_mean.LineWidth = 0.5;

                % set axis-names:
                xlabel("row index catchment [-]",'FontSize',11);
                ylabel("leverage [-]",'FontSize',11);

                %set plot title
                %title("catchment observations vs. leverage",'FontSize',12,'FontWeight','bold');

                % moving of title and axis labelshLabel = get(gca,'xlabel'); 
                set(hLabel, 'Units', 'pixels');

                hLabel = get(gca,'ylabel'); 
                set(hLabel, 'Position', get(hLabel, 'Position') - [0.1 0 0]); 
                set(hLabel, 'Units', 'pixels');

                %hLabel = get(gca,'title'); 
                %set(hLabel, 'Position', get(hLabel, 'Position') + [0 0.005 0]); 
                %set(hLabel, 'Units', 'pixels');
            end

            % export of validation plots

            if ~exist(['outputs'] ) mkdir(['outputs']); %controls if folder for outputs exists, if not, one gets made
            end

            if ~exist(['outputs/lasso_validation_plots'] ) mkdir(['outputs/lasso_validation_plots']); %controls if folder for validation plots within output-folder exists, if not, one gets made
            end

            if ~exist(fullfile('outputs/lasso_validation_plots',strrep(filename,".txt",""))) mkdir(fullfile('outputs/lasso_validation_plots',strrep(filename,".txt",""))); %controls if folder for validation plots within output-folder exists, if not, one gets made
            end

            for j = 1:7

                if i==1
                    if j==1
                    plot_name = sprintf('CVplot_q347_%d_interval_%d',K,L); end
                    if j==2
                    plot_name = sprintf('normal_probability_plot_%d_interval_%d',K,L); end
                    if j==3
                    plot_name = sprintf('residualplot_all_%d_interval_%d',K,L); end
                    if j==4
                    plot_name = sprintf('studentized_residualplot_all_%d_interval_%d',K,L); end
                    if j==5
                    plot_name = sprintf('residualplot_individual_%d_interval_%d',K,L); end
                    if j==6
                    plot_name = sprintf('leverage_vs_residualplot_%d_interval_%d',K,L); end
                    if j==7
                    plot_name = sprintf('leverageplot_%d_interval_%d',K,L); end
                end
                if i==2
                    if j==1
                    plot_name = sprintf('CVplot_q347_log_%d_interval_%d',K,L); end
                    if j==2
                    plot_name = sprintf('normal_probability_plot_log_q347_%d_interval_%d',K,L); end
                    if j==3
                    plot_name = sprintf('residualplot_all_log_q347_%d_interval_%d',K,L); end
                    if j==4
                    plot_name = sprintf('studentized_residualplot_all_log_q347_%d_interval_%d',K,L); end
                    if j==5
                    plot_name = sprintf('residualplot_individual_log_q347_%d_interval_%d',K,L); end
                    if j==6
                    plot_name = sprintf('leverage_vs_residualplot_log_q347_%d_interval_%d',K,L); end

                end


                plot_path = fullfile('outputs/lasso_validation_plots',strrep(filename,".txt",""),plot_name);

                if j*i < 14
                    print(plot_path,'-deps');
                    set(figure((i-1)*7+j),'color','w');
                    set(figure((i-1)*7+j), 'InvertHardCopy', 'off');
                    print('-loose','-dpng',plot_path,'-r400') ; %save figure as png
                    print(gcf, '-depsc2','-loose',[plot_path,'.eps']);
                end
            end
        end

        if I_mdl_results_export == 1
        %EXCEL-EXPORT OF RESULTS

            outFile = strrep(strrep(filename,"InputDaten","results_"),".txt",".xls");

            if i==1
                if ~exist(['outputs/lasso'] )
                    mkdir(['outputs/lasso']); 
                end
                excel_path = fullfile('outputs/lasso',outFile);
            end

            %generating the model formula
            predictors_nonzero = FitInfo.PredictorNames(coeffs(:,FitInfo.IndexMinMSE)~=0);
            
             if i == size(response_types,2)
                xlwrite(excel_path,{'response variable','mean observation value',' ','MSE','RMSE','MAE','R^2'},'stats','B1'); %der eine Leerschlag ist dabei damit der Titel "mean observation value" nicht �ber die Zelle rechts davon hinwegragt im Excel-File
                xlwrite(excel_path,regression_output(:,1:2),'stats','B2');
                xlwrite(excel_path,regression_output(:,3:end),'stats','E2');

                for n=1:count
                    xlwrite(excel_path,{sprintf('model %d',n)},'stats',sprintf('A%d',2*n));
                    xlwrite(excel_path,{'cross validated';'fit on all data'},'stats',sprintf('D%d',n*2));           
                end
            end
            
            if size(predictors_nonzero,2) > 0
                formula_predictors = strcat({'  ~  intercept + '},predictors_nonzero(1));
                for m = 1:size(predictors_nonzero,2)-1
                    formula_predictors = strcat(formula_predictors,{' + '},predictors_nonzero{m+1});
                end

                formula_complete = strcat(response_names{i},formula_predictors);
                
            
                % creating excel-file and inserting titles and accordingly (computed) values
                xlwrite(excel_path,{'formula'},sprintf('stats_model_%d',count),'A1');
                xlwrite(excel_path,formula_complete(1),sprintf('stats_model_%d',count),'A3');
                xlwrite(excel_path,{'predictors statistics'},sprintf('stats_model_%d',count),'A6');
                xlwrite(excel_path,{'predictors'},sprintf('stats_model_%d',count),'A8');
                xlwrite(excel_path,{'coefficient values'},sprintf('stats_model_%d',count),'B8');
                xlwrite(excel_path,{'intercept'},sprintf('stats_model_%d',count),'A9');
                xlwrite(excel_path,transpose(predictors_nonzero),sprintf('stats_model_%d',count),'A10');
                xlwrite(excel_path,intercept_minMSE,sprintf('stats_model_%d',count),'B9');
                xlwrite(excel_path,coeffs_minMSE(coeffs(:,FitInfo.IndexMinMSE)~=0),sprintf('stats_model_%d',count),'B10');
                xlwrite(excel_path,{'model estimations'},sprintf('stats_model_%d',count),'D6');
                xlwrite(excel_path,{'observation values'},sprintf('stats_model_%d',count),'D8');
                xlwrite(excel_path,{'estimated values'},sprintf('stats_model_%d',count),'E8');
                xlwrite(excel_path,{'residuals'},sprintf('stats_model_%d',count),'F8');
                xlwrite(excel_path,response_types(:,1),sprintf('stats_model_%d',count),'D9');
                xlwrite(excel_path,model_estimates(:,i),sprintf('stats_model_%d',count),'E9');
                xlwrite(excel_path,residuals(:,i),sprintf('stats_model_%d',count),'F9');
            else
               sprintf("no positive MSE predictors found for i: %d",i)
               continue
            end
            
        end

    end
end

