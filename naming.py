import os

target_directory = './image/chaeyeong2'
desired_name = 'chaeyeong2'


file_list = os.listdir(target_directory)


for i, filename in enumerate(file_list):

    file_extension = os.path.splitext(filename)[-1]
    

    new_filename = f"{desired_name}_{i:03d}{file_extension}"
    
    os.rename(
        os.path.join(target_directory, filename),
        os.path.join(target_directory, new_filename)
    )
