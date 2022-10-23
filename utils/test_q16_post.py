import requests
import os

os.chdir(r'C:\Users\Calvin Howard\OneDrive\Documents\Work\KiTH Solutions\KiTHPython\Python 3.7.7 Repository\Server\utils')
mydir = os.getcwd()
cubefile = 'cube.png'
infinityfile = 'infinity.png'
clockfile = 'clock.png'
cube_path = os.path.join(mydir, cubefile)
infinity_path = os.path.join(mydir, infinityfile)
clock_path = os.path.join(mydir, clockfile)

cubehandle = open(cube_path, 'rb')
infinityhandle = open(infinity_path, 'rb')
clockhandle = open(clock_path, 'rb')

files = {
    'cube_file': ('cube.png', open(cube_path, 'rb')), 
    'infinity_file': ('infinity.png', open(infinity_path, 'rb')), 
    'clock_file': ('clock.png', open(clock_path, 'rb'))}
test_post = requests.post('http://127.0.0.1:8000/question_sixteen_screen', files = files)
test_post.text