#run in cmd > uvicorn server_test:app --reload
from fastapi import FastAPI, File, UploadFile, Request
from matplotlib.pyplot import text
from pydantic import BaseModel
import os
from typing import Text, Union 
import utils.server_utils as pkg

class KeyInfo(BaseModel):
    uuid: str
    data: str

class Text_Results(BaseModel):
    text_one: Union[str, None] = None
    text_two: Union[str, None] = None
    text_three: Union[str, None] = None
    text_four: Union[str, None] = None
    text_five: Union[str, None] = None
    text_six: Union[str, None] = None
    text_seven: Union[str, None] = None
    text_eight: Union[str, None] = None
    text_nine: Union[str, None] = None
    text_ten: Union[str, None] = None

class Audio_Results(BaseModel):
    audio_one: Union[str, None] = None

class Image_Results(BaseModel):
    image_one: Union[str, None] = None
    image_two: Union[str, None] = None
    image_three: Union[str, None] = None

class Score_Output(Text_Results):
    score: Union[int, None] = None

app = FastAPI()

@app.get("/")
def root():
    return 'this is some debugging text'

@app.post("/question_two_screen")
def score(text_results: Text_Results): 
    province = text_results.text_one; country = text_results.text_two
    continent = text_results.text_three; modality = text_results.text_four
    day = text_results.text_five; month = text_results.text_six
    year = text_results.text_seven; date = text_results.text_eight
    season = text_results.text_nine; city = text_results.text_ten
    q2_score = pkg.score_q2(province, country, continent, modality, day, month, year, date, season, city)
    return {'score': int(q2_score)}

@app.post('/question_four_screen')
def score(text_results: Text_Results):
    one = text_results.text_one; two = text_results.text_two; three = text_results.text_three; four = text_results.text_four; five = text_results.text_five
    q4_score = pkg.score_q4(one, two, three, four, five)
    return {'score': int(q4_score)}

@app.post('/question_ten_screen')
def score(text_results: Text_Results):
    sentence = text_results.text_one
    q10_score = pkg.score_q10(sentence)
    return {'score': int(q10_score)}

@app.post('/question_sixteenC_screen')
def score(text_results: Text_Results):
    q16_score = pkg.score_q16()
    return {'score': int(q16_score)}

@app.post('/question_twenty_screen')
def score(text_results: Text_Results):
    q20_score = pkg.score_q20(text_results.text_one, text_results.text_two, text_results.text_three, text_results.text_four, text_results.text_five, text_results.text_six, text_results.text_seven, text_results.text_eight)
    return {'score': int(q20_score)}

@app.post('/upload_png')
async def upload(file: UploadFile = File(...)):
    if not file:
        return {'message': 'no file sent'}
    else:
        try:
            file_location = f"{file.filename}"
            print(file_location)
            with open(file_location, 'wb+') as file_object:
                file_object.write(file.file.read())
        finally:
            x = file.filename
            file.file.close()
        return {'info': f'file"{x}" saved at"{file_location}"'}

@app.post("/audio_question/{question}")
async def upload(question: str, file: UploadFile = File(...)):
    score = 0
    score_qn = 'score_' + question
    print(score_qn)
    print(file.file, 'type: ',  type(file.file))
    if not file:
        return {'message': 'no file sent'}
    else:
        try:
            score = eval('pkg.' + score_qn + '(file.file)')
        except:
            print('Error')
    return {'score': score}

@app.post('/upload_wav/{qn}')
async def upload(file: UploadFile = File(...)):
    if not file:
        return {'message': 'no file sent'}
    else:
        try:
            file_location = f"{file.filename}"
            my_dir = os.getcwd()
            dest_dir = 'audio'
            save_path = os.path.join(my_dir, dest_dir)
            with open(file_location, 'wb+') as file_object:
                os.chdir(save_path)
                file_object.write(file.file.read())
        finally:
            x = file.filename
            file.file.close()
        return {'info': f'file: "{x}" saved at: "{file_location}"'}

#Functional CSV posting function
@app.post('/results/')
async def test_results(result: KeyInfo):
    with open(f'{result.uuid}.csv', 'a') as f:
        f.write(result.data)
        print('file saved at: ', os.getcwd())
    pass

# #Testing dict posting function
# @app.post('/results/')
# async def test_results(result: Result):
#     with open(f'{result.uuid}.csv', 'a') as f:
#         f.write(result.data)
#         print('file written')
#         print(os.getcwd())
#         print('code worded')
#     pass