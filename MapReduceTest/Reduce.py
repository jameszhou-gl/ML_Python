# wordcount reduce阶段

cur_word = None
sum = 0

ff = open("file02.txt", "r")
for line in ff.readlines():
    x = line.strip()
    if cur_word == None:
        cur_word = x
    if cur_word != x:
        print('\t'.join([cur_word, str(sum)]))
        cur_word = x
        sum = 0
    sum += 1
print('\t'.join([cur_word, str(sum)]))
