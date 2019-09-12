#!/R
library(data.table)
library(ggplot2)
setwd("all/rf_importance/best")

rowOrder <- c("Pmean","Psummer","Pwinter","PETmean","PETsummer","PETwinter",
              "size","Hmin","Hmax","Hmean","relief","aspectnorth","aspecteast","aspectsouth","aspectwest",
              "SLP_MEAN","slope<10","slope10_30","slope30_60","slope>60",
              "roughness","channel length","UCA","SCA","TWI","drainage_density",
              "SP_EG","SP_SG","SP_G","SP_M","SP_GUT","SP_SGUT","storagemean",
              "D_SG","D_G","D_SCHG","D_N","D_UE","permeabilitymean","GRU_SF","GRU_F","GRU_M","GRU_T","GRU_ST","soil_depthmean",
              "BOD_KVF","BOD_DVF","BOD_SVEG","BOD_AWW","BOD_WAL","BOD_DW","BOD_FELS","BOD_GLE","BOD_SUM","BOD_FLIESS","BOD_STEH",
              "BED_FELS","BED_FLIESS","BED_GEB","BED_LOCK","BED_GLET","BED_STEH","BED_FEUCH","BED_WALD","BED_WALDO","bedrock","unconsolidated","waters","forests",
              "depth0","depth0.1_10","depth>10",
              "Jura","OSM","OMM","USM","cystalin_rocks","sedimentary_rocks","total","artificial","other","alluvial","glacial","swamp","debris","landslide")

files <- list.files(pattern="meanImportancePredictors.*")
data <- lapply(files, function(file){
  data <- read.table(file,header=F)
  colnames(data) <- c("variable","relImportance")
  data$variable <- sub("_$","",data$variable)
  name <- sub(".*InputDaten_(.*)_(.*)_zero.txt.*","\\1_\\2",file)
  return(list(data,name))
})

mergedData <- Reduce(function(x,y) merge(x,y,all=T,by="variable"),lapply(data,"[[",1))
#mergedData[is.na(mergedData)] <- 0
colnames(mergedData) <- c("variable",sapply(data,"[[",2))
mergedData <- mergedData[,c(1,order(colnames(mergedData)[2:ncol(mergedData)])+1)]

dotPlotData <- melt(setDT(mergedData),id.vars="variable")
dotPlotData[, area := sub("(.*)_(.*)","\\1",variable.1)]
dotPlotData[, response := sub("(.*)_(.*)","\\2",variable.1)]
dotPlotData[, col := c("plateau"="darkred","alps"="darkblue")[area]]
colnames(dotPlotData) <- c("variable","category","value","area","response","col")

# set order of plotting variables
dotPlotData[area == "alps", area := "Swiss Alps"]
dotPlotData[area == "plateau", area := "Swiss Plateau"]
dotPlotData$area <- factor(dotPlotData$area,levels=c("Swiss Plateau","Swiss Alps"))
dotPlotData[variable == "depth0_1_10", variable := "depth0.1_10"]
dotPlotData[variable == "depth_10", variable := "depth>10"]
dotPlotData[variable == "slope_10", variable := "slope<10"]
dotPlotData[variable == "slope_60", variable := "slope>60"]
dotPlotData$variable <- factor(dotPlotData$variable,levels=rev(rowOrder))#
#dotPlotData[category == "q95", category := expression(q["95"])]
#dotPlotData[category == "Qmin", category := expression(Q[min])]

ggplot(dotPlotData,aes(x=response,y=variable,size=value,col=col,alpha=value)) +
  geom_point() +
  scale_size_continuous(name = "importance", breaks = seq(0.1,0.5,by=0.1),labels = c("<0.1","0.2","0.3","0.4",">0.4")) +
  scale_colour_manual(breaks = c("1", "2"), values = c("#990000", "#000099")) +
  facet_grid("area") +
  guides(col = F,alpha=F) +
  theme_bw() + 
  theme(text = element_text(color="black", size=16, family="sans"),
        axis.text.x = element_text(angle=45,hjust=1,vjust=1),
        axis.title.x = element_text(color="black", size=24, family="sans", face="bold"),
        axis.title.y = element_text(color="black", size=24, family="sans", face="bold"))
dev.print(cairo_pdf,"dotplot_relativeImportance_RF.pdf",height=15,width=8)
