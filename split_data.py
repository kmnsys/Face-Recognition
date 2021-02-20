import splitfolders

#To split dataset https://github.com/jfilter/split-folders library has been used.
splitfolders.ratio("D:/CroppedYale", output="D:/data2", seed=1337, ratio=(.7, .0, .3), group_prefix=None) # default values
