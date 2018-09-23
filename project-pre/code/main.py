import os


os.system("python get_video_frame.py")
os.system("python feature/extract_image.py")
os.system("python feature/extract_question.py")
os.system("python train.py")
os.system("python predict.py")
