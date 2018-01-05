import os

file = open("imagelist.txt", "w")
filenames = os.listdir("images")

for name in filenames:
	if not name == ".keep":
		file.write("images/" + name + "\n")		

file.close()