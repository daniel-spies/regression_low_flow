#!/R

inputFolder <- file.path(path,"all/inputs")
outDir <- file.path(path,"randomForest_reduced/inputs")
system(paste0("mkdir -p ",outDir))

predictorFiles <- list.files(file.path(path,"all/rf_importance/all"),pattern="meanImportance.*.txt",full.names=T)
predictorList <- lapply(predictorFiles,read.csv,sep="\t",header=F)
names(predictorList) <- basename(predictorFiles)

## extract best predictors of input files obtained by random forest
## for the best overall prediction that replaced missing values with 0
redPred <- lapply(predictorFiles, function(predictor){
    file <- file <- basename(predictor)
    # extract used parameters
    log_response <- sapply(strsplit(file,"_"),"[[",3)
    area <- sapply(strsplit(file,"_"),"[[",5)
    responseVar <- sapply(strsplit(file,"_"),"[[",6)
    qualFilter <- sub(".*_qualThresh([12]).txt","\\1",file)
    
    # extract best variables
    varThresh <- quantile(predictorList[[file]][,2])[4]
    bestPred <- predictorList[[file]][predictorList[[file]][,2]>=varThresh,1]
    
    # create new reduced input file
    inFile <- sub(".*_(InputDaten.*.txt)","\\1",file)
    inputData <- read.table(file.path(inputFolder,inFile),header=T)
    outputData <- inputData[,c(responseVar,bestPred)]
    outFile <- sub(".txt$",paste0("_log",log_response,"_randomForest_reduced.txt"),inFile)
    write.table(outputData,file.path(outDir,outFile),row.names=F,quote=F,sep="\t")
    return(outputData)
})