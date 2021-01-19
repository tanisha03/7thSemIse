# #library for fviz_cluster
# library(factoextra)

# #importing the Iris dataset and removing categorical attributes
# data <- iris
# data$Species <- NULL

# #K-Means Clustering
# distance1 <- scale(dist(data,method="euclidian"))
# kModel <- kmeans(distance1,3)
# fviz_cluster(kModel, data, geom="point")


# #Hierarchical Clustering
# distance2 <- dist(data[,3:4],method="euclidian")
# hModel <- hclust(distance2,method="average")
# plot(hModel)

library(cluster)

iris_1 <- iris[,-5]

km <- kmeans(iris_1, centers = 3, nstart = 20)

table(iris$Species,km$cluster)

plot(iris_1[c("Sepal.Length","Sepal.Width")],col=km$cluster,main="Kmeans with 3 clusters")
