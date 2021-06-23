library("scmamp")

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

result.matrix <- read.csv("../../results/combination_data_shd.csv", row.names = 1, header= TRUE)

plotCD(result.matrix, alpha = 0.05, cex = 0.75)
