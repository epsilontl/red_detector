import py7zr
import os

zip_file_path = "/home/wds/Downloads/prophesee_gen4/testset/testfilelist02.7z"
unzip_file_path = "/home/wds/Downloads/prophesee_gen4/testset"

with py7zr.SevenZipFile(zip_file_path, mode='r') as z:
    z.extractall(unzip_file_path)


# f = open("/home/wds/Downloads/prophesee_gen4/testset/test.txt", mode='w')
# for path in os.listdir("/home/wds/Downloads/prophesee_gen4/testset/test"):
#     if path[-3:] == "dat":
#         content = path + '\n'
#         f.write(content)

