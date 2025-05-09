import json

category = "education"
# f = open(f"{category}/posts.json", "r")
# f = json.load(f)
#
# print(len(f))

f = open(f"{category}/valid_posts.json", "r")
f = json.load(f)
for key in f.keys():
    print(len(f[key]))
