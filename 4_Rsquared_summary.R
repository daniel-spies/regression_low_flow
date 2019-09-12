#!/R
library(xlsx)

lapply(c("all/outputs","randomForest_reduced/outputs"), function(analysis){
    files <- list.files(analysis,recursive=T,pattern="*.xls")
    data <- lapply(files, function(file){
        # get EZG identifiers by GIS_ID
        region <- sub("results__(plateau|alps).*","\\1",basename(file))
        response <- sub("results__(plateau|alps)_(.*)_zero.*","\\2",basename(file))
        resp_vec <- rep(c(response,paste0(response,"_log")),each=2)
        fit_vec <- rep(c("cross-validated","fit_on_all_data"),2)
        method <- dirname(file)
        # check which model is the best
        x <- read.xlsx(file.path(analysis,file),sheetName="stats")
        retData <- cbind(region,method,resp_vec,fit_vec,round(as.numeric(x$R.2),4))
        colnames(retData) <- c("region","method","response","fit","Rsquared")
        return(retData)
    })
    outData <- as.data.frame(Reduce(rbind,data))
    write.table(outData,file.path(analysis,paste0(dirname(analysis),"_Rsquared_summary.tsv")),row.names=F,quote=F,sep="\t")
})