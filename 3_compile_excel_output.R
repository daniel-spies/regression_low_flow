library(xlsx)

readxls <- function(dir,file,sheet="stats_model_1",startRow=8){
  xlsx::read.xlsx(file.path(dir,file),sheetName=sheet,startRow=startRow)
}

index_files <- list.files("index",full.names=T)
index_lists <- lapply(index_files, read.table,header=T)
names(index_lists) <- sub(".*_qual_(.*)_zero_index.txt","\\1",index_files)

lapply(c("all/outputs","randomForest_reduced/outputs"), function(analysis){
    rf_outliers <- read.table(file.path(analysis,"../rf_parameters.txt"),header=T)
    data <- lapply(dir(analysis), function(method){
        files <- list.files(file.path(analysis,method))
        data_method <- lapply(files, function(file){
            # get EZG identifiers by GIS_ID
            name <- sub("results__(.*)_(.*)_zero.*",paste0(method,"_\\1_\\2_"),file)
            region <- sub("results__(plateau|alps).*","\\1",file)
            GIS_ID <- index_lists[[region]][,"GIS_ID"]

            #if( method == "random_forest"){
            #    outlierIdx <- rf_outliers[grep(sub(paste0(method,"_"),"",name),rf_outliers$file),"outlierIdx"]
            #    GIS_ID <- GIS_ID[,-outlierIdx]
            #}
            # check which model is the best
            x <- readxls(dir=file.path(analysis,method),file=file,sheet="stats",startRow=1)
            model <- which.max(x$R.2[c(2,4)])
            input <- tryCatch(readxls(dir=file.path(analysis,method),file=file,sheet=paste0("stats_model_",model)),
                         error=function(e) tryCatch(readxls(dir=file.path(analysis,method),file=file,sheet="stats_model_1"),
                                               error = function(e2) NULL))
            if( is.null(input)){
                return(NULL)
            } else {
                retData <- cbind(GIS_ID,input[,c("observation.values","estimated.values","residuals")])
                colnames(retData) <- c("GIS_ID",paste0(name,c("obs","est","residual")))
                return(retData)
            }
        })
        data_method <- data_method[unlist(sapply(data_method,length))>0]
        retData <- Reduce(function(x,y) merge(x,y,all=T,by="GIS_ID"),data_method)
    })
    outData <- Reduce(function(x,y) merge(x,y,all=T,by="GIS_ID"),data)
    write.table(outData,file.path(analysis,paste0(dirname(analysis),"_obs_vs_est.tsv")),row.names=F,quote=F,sep="\t")
})