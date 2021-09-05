import sys
import os
from wand.image import Image


print("This is the name of the script: ", sys.argv[0])
print("Number of arguments: ", len(sys.argv))
print("The arguments are: ", str(sys.argv))

if len(sys.argv) != 3:
    sys.exit('Need to give source and target directory as in:\n' \
             'python heic_to_jpg.py PATH_TO_SOURCE PATH_TO_TARGET')
if not(os.path.isdir(sys.argv[-2])):
    sys.exit('PATH_TO_SOURCE needs to be path to a directory.')
if not(os.path.isdir(sys.argv[-1])):
    sys.exit('PATH_TO_TARGET needs to be path to a directory.')

source_dir = str(sys.argv[-2])
target_dir = str(sys.argv[-3])

