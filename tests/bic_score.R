library(bnlearn)

true = as.bn("[A][B][C][D][E]")
data = read.csv("data.csv")
data[] = lapply(data, as.factor)

print(bnlearn::score(true, data, type="bic"))