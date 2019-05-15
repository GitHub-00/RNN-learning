
'''数据预处理与数据生成'''
import sys
import os
import jieba

train_file = os.path.abspath(os.path.join(os.getcwd(),'../cnews/cnews.train.txt'))
val_file = os.path.abspath(os.path.join(os.getcwd(),'../cnews/cnews.val.txt'))
test_file=os.path.abspath(os.path.join(os.getcwd(),'../cnews/cnews.test.txt'))

seg_train_file = os.path.abspath(os.path.join(os.getcwd(),'../cnews/cnews.train.seg.txt'))
seg_val_file = os.path.abspath(os.path.join(os.getcwd(),'../cnews/cnews.val.seg.txt'))
seg_test_file = os.path.abspath(os.path.join(os.getcwd(),'../cnews/cnews.test.seg.txt'))

vocab_file = os.path.abspath(os.path.join(os.getcwd(),'../cnews/cnews.vocab.txt'))
category_file = os.path.abspath(os.path.join(os.getcwd(),'../cnews/cnews.category.txt'))

with open(val_file,'r') as f:
    lines = f.readlines()

#print(lines[0])
#print(type(lines[0]))
label, content = lines[0].strip('\r\n').split('\t')
word_iter = jieba.cut(content)

#print(word_iter)
#print('/'.join(word_iter))

def generate_seg_file(input_file, output_seg_file):
    #按行对文件内容分词
    with open(input_file,'r') as f:
        lines = f.readlines()
    with open(output_seg_file,'w') as f:
        for line in lines:
            label, content = line.strip('\r\n').split('\t')
            word_iter = jieba.cut(content)
            word_content = ''
            for word in word_iter:
                word = word.strip(' ')
                if word != '':
                    word_content += word + ' '

            out_line = '%s\t%s\n' %(label,word_content.strip(' '))
            f.write(out_line)


#generate_seg_file(train_file, seg_train_file)
#generate_seg_file(val_file, seg_val_file)
#generate_seg_file(test_file, seg_test_file)

def generate_vocab_file(input_seg_file, output_vocab_file):
    with open(input_seg_file,'r') as f:
        lines = f.readlines()

    word_dict = {}
    for line in lines:
        label, content = line.strip('\r\n').split('\t')
        for word in content.split():
            word_dict.setdefault(word, 0 )
            word_dict[word] += 1
    #[(词,频次),,,,,()]
    sorted_word_dict = sorted(word_dict.items(), key = lambda d:d[1])

    with open(output_vocab_file,'w') as f:
        f.write('<UNK>\t10000000\n')
        for item in sorted_word_dict:
            f.write('%s\t%d\n' %(item[0], item[1]))

#generate_vocab_file(seg_train_file, vocab_file)


def generate_category_file(input_file, category_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    category_dict = {}
    for line in lines:
        label, content = line.strip('\r\n').split('\t')
        category_dict.setdefault(label, 0)
        category_dict[label] += 1
    category_number = len(category_dict)

    with open(category_file,'w') as f:
        for category in category_dict:
            line = '%s\n' %category
            print('%s\t%d' %(category,category_dict[category]))

            f.write(line)

generate_category_file(train_file, category_file)


