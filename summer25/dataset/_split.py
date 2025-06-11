"""
TODO: split dataset into random train/val/test sets 

take train/val/test proportions - ensure it adds up to 1
split at audio level AND task level AND subject level depends how data is given 
if directory: split only audio level 
if csv: split based on task/subject

randomly split, optionally set a seed 

create a json? create a csv? - save to splits with name specifying split level, proportions, seed, and if it exists already and there is NO SEED, append a number

"""