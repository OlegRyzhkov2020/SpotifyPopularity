# Adjusts artist.rank_sum in the target dataset
#
# Currently, artist.rank_sum includes the current song. This causes collinearity in our data
# as - for example - if an artist only has 1 song, artist.rank_sum is a perfect predictor of
# popularity.
# 
# This approach calculates artist.rank_sum as the sum all songs by the artist not including the current song.
# Artists with 1 or fewer ranked songs are represented by a value of 0.

#read in target dataset
target_numeric_popularity = read.csv("target_numeric_popularity.csv")

#subtract current song's popularity from total sum
target_numeric_popularity$artist.rank_sum = 
              target_numeric_popularity$artist.rank_sum - 
              target_numeric_popularity$popularity

target_numeric_popularity$artist.trackcount = 
              target_numeric_popularity$artist.trackcount-1

target_numeric_popularity[target_numeric_popularity$popularity>0,]$artist.trackrankedcount = 
              target_numeric_popularity[target_numeric_popularity$popularity>0,]$artist.trackrankedcount-1

target_numeric_popularity[is.na(target_numeric_popularity$artist.rank_mean),]$artist.rank_mean = 0

write.csv(target_numeric_popularity,file="target_numeric_popularity.csv")