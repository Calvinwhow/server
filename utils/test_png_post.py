import requests
import os

os.chdir(r'C:\Users\Calvin Howard\OneDrive\Documents\Work\KiTH Solutions\KiTHPython\Python 3.7.7 Repository\Server\utils')
mydir = os.getcwd()
myfile = 'cube.png'
png_path = os.path.join(mydir, myfile)

handle = open(png_path, 'rb')
files = {'file':(myfile, open(png_path, 'rb'))}
test_post = requests.post('http://127.0.0.1:8000/upload_png', files = files)
test_post.text

q16_score = dict(test_score = 0)
q16_score_post = requests.post('http://127.0.0.1:8000/question_sixteenC_screen', json = q16_score)
print(q16_score_post.json())

#as a function
def upload_drawing(drawing_name, *args):
    mydir = os.getcwd()
    myfile = drawing_name + '.png'
    png_path = os.path.join(mydir, myfile)
    files = {'file':(myfile, open(png_path, 'rb'))}
    test_post = requests.post('http://127.0.0.1:8000/upload_png', files = files) #THIS UPLOADS TO A LOCALHOST
    test_post.text
    pass