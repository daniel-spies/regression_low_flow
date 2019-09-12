#!/R
############################### library loading ############################### 
library(data.table)
library(xlsx)
############################### paths ############################### 
inputDir <- "all/inputs"
indexDir <- "all/index"
############################### data loading ############################### 
# manually remove NA rows
alpine <- setDT(read.xlsx("Inputvariablen_gesamt_REDUZIERT_FINAL.xlsx",sheetIndex=3,startRow = 4))
midland <- setDT(read.xlsx("Inputvariablen_gesamt_REDUZIERT_FINAL.xlsx",sheetIndex=2,startRow = 4))

dataList <- list(alps=alpine,plateau=midland)

response_var <- c("q95","Qmin","C")
data_qual_thresh <- c(2)

# for each region type split data further
lapply(names(dataList), function(region){
  data <- dataList[[region]][,11:ncol(dataList[[region]])]
  response <-  dataList[[region]][,8:10]
  quality <- unlist(dataList[[region]][,6])
  quality <- quality[!is.na(quality)]
  # split per response variable
  
  sapply(response_var, function(resp){
    all <- setDT(cbind(response[,resp,with=F],data))
    colnames(all) <- c(resp,colnames(data))
    all <- all[,grep("NA",colnames(all),invert = T,value=T),with=F] # remove excel NA columns
    all <- all[1:length(quality),] # remove NA rows
    all[is.na(all)] <- 0 # missing values will be replaced by median values
    all_thresh <- all[quality <= 2,]
    outFile <- paste0("InputDaten_",region,"_",resp,"_zero.txt")
    write.table(all_thresh,file.path(inputDir,outFile),row.names=F,quote=F,sep="\t")
  })
})

# write index files for later merging
lapply(names(dataList), function(region){
  data <- dataList[[region]][,c(3,6)]
  data <- data[complete.cases(data), ]
  data <- data[Quality <= 2,]
  outFile <- paste0("Index_qual_",region,"_zero_index.txt")
  write.table(data,file.path(indexDir,outFile),row.names=F,quote=F,sep="\t")
})