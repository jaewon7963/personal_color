from rembg import remove
from PIL import Image
import os

input_dir = './musinsa'

for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.PNG'):
        print(filename)
        input = Image.open(input_dir + "/" + filename) 
        output = remove(input) 
        output.save('./nut/' + filename.split('.')[0] + '.PNG') # rembg의 결과는 png로