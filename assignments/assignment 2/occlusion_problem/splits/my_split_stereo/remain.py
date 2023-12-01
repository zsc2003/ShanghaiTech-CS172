# open the val_files.txt
# read each line
# remain things after the first '/'
# write to val.txt

with open('val_files.txt', 'r') as f:
    with open('val.txt', 'w') as f1:
        for line in f.readlines():
            f1.write(line.split(' ')[0] + '\n')

with open('train_files.txt', 'r') as f:
    with open('train.txt', 'w') as f1:
        for line in f.readlines():
            f1.write(line.split(' ')[0] + '\n')