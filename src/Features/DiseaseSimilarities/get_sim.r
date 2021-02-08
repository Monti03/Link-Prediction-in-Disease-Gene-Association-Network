library(meshes)
library(MeSH.Hsa.eg.db)
processFile = function(filepath) {
   con = file(filepath, "r")
   vec = c()
   while ( TRUE ) {
      line = readLines(con, n = 1)
      if (length(line) == 0 ) {
         break
      }
      vec = c(vec, c(line))
  }

  close(con)
  return(vec)
}
print('defining hsamd')
hsamd <- meshdata("MeSH.Hsa.eg.db", category=c('C', 'D'), computeIC=T, database="gendoo")
print('read file')
v <- processFile('diseases.txt')
i <- 0
print('starting for loop')
for (disease1 in v) {
   for (disease2 in v) {
      if(disease1 >= disease2){
         val <- meshSim(disease1, disease2, semData=hsamd, measure="Wang")
         if(val != 0){
            print(paste(i, val), sep= ' ')
         }
         to_print <- paste(disease1, disease2, val, '\n', sep="\t")

         cat(to_print, file="outfile_new.txt",append=TRUE)
         #print(i)
         i <- i + 1
      }
   }
}
