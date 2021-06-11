import random
import csv
import os.path
from os import path
import string
import math

def multinomial_sample(n,p,k=1):
    '''
    Return samples from a multinomial distribution.

    n:= number of trials
    p:= list of probabilities
    k:= number of desired samples
    '''
    assert isinstance(n, int)
    assert n > 0
    assert isinstance(k, int)
    assert k > 0
    assert isinstance(p, list)
    sum = 0
    for prob in p:
        assert isinstance(prob, float)
        assert prob >=0 and prob <= 1
        sum += prob
    assert sum == 1
    samples = []
    for i in range(k):
        sample = [0]*len(p)
        for j in range(n):
            num = random.random()
            lower = 0
            upper = 0
            for z in range(len(p)):
                upper += p[z]
                #print(lower, upper)
                if (num >= lower and num <= upper):
                    sample[z] += 1
                lower += p[z]

        samples.append(sample)
    return samples

def encrypt_message(message, fname):
    '''
    Given `message`, which is a lowercase string without any punctuation, and `fname` which is the
    name of a text file source for the codebook, generate a sequence of 2-tuples that
    represents the `(line number, word number)` of each word in the message. The output is a list
    of 2-tuples for the entire message. Repeated words in the message should not have the same 2-tuple.

    :param message: message to encrypt
    :type message: str
    :param fname: filename for source text
    :type fname: str
    :returns: list of 2-tuples
    '''
    assert isinstance(message, str)
    assert isinstance(fname, str)
    assert path.exists(fname)
    dict = {}
    codebook = open(fname, 'r')
    line_num = 0
    total_words = 0
    for line in codebook:
        line = line.translate(str.maketrans('', '', string.punctuation))
        line = line.replace('\n', '')
        line = line.replace('  ', ' ')
        line = line.lower()
        line_list = line.split(' ')
        line_num += 1
        word_num = 0
        for word in line_list:
            not_word = False
            for char in word:
                if(not char.isalpha()):
                    not_word = True
            if(not not_word):
                word_num += 1
                total_words += 1
                if word in dict.keys():
                    dict[word].append((line_num, word_num))
                else:
                    dict[word] = [(line_num, word_num)]
    #print(total_words)
    #print(line_num)
    #print(dict.keys())
    message = message.translate(str.maketrans('', '', string.punctuation))
    message = message.replace('\n', '')
    message = message.replace('  ', ' ')
    message = message.lower()
    message_list = message.split(' ')
    encoded = []
    count_dict = {}
    for word in message_list:
        if(word in count_dict.keys()):
            count_dict[word] += 1
        else:
            count_dict[word]= 1
    for key in count_dict.keys():
        assert len(dict[key]) >= count_dict[key]
    for i in range(len(message_list)):
        not_found = True
        while(not_found):
            index = random.randint(0, len(dict[message_list[i]])-1)
            if(dict[message_list[i]][index] not in encoded):
                encoded.append(dict[message_list[i]][index])
                not_found = False
    return encoded

def decrypt_message(inlist, fname):
    '''
    Given `inlist`, which is a list of 2-tuples`fname` which is the
    name of a text file source for the codebook, return the encrypted message.

    :param message: inlist to decrypt
    :type message: list
    :param fname: filename for source text
    :type fname: str
    :returns: string decrypted message
    '''
    assert isinstance(inlist, list)
    for tup in inlist:
        assert isinstance(tup, tuple)
        assert len(tup) == 2
        assert tup[0] > 0
        assert tup[1] > 0
    assert isinstance(fname, str)
    assert path.exists(fname)
    dict = {}
    codebook = open(fname, 'r')
    line_num = 0
    total_words = 0
    for line in codebook:
        line = line.translate(str.maketrans('', '', string.punctuation))
        line = line.replace('\n', '')
        line = line.replace('  ', ' ')
        line = line.lower()
        line_list = line.split(' ')
        line_num += 1
        word_num = 0
        for word in line_list:
            not_word = False
            for char in word:
                if(not char.isalpha()):
                    not_word = True
            if(not not_word):
                word_num += 1
                total_words += 1
                if word in dict.keys():
                    dict[word].append((line_num, word_num))
                else:
                    dict[word] = [(line_num, word_num)]
    key_list = list(dict.keys())
    val_list = list(dict.values())
    message = ''
    for i in range(len(inlist)):
        for tups in val_list:
            if(inlist[i] in tups):
                position = val_list.index(tups)
                message += key_list[position] + ' '
    return message[:-1]


def split_by_n(fname,n=3):
    '''
    Split files into sub files of near same size
    fname : Input file name
    n is the number of segments
    '''
    assert isinstance(fname, str)
    assert path.exists(fname)
    assert isinstance(n, int)
    assert n > 0
    nums = []
    for i in range(n):
        num = str(i)
        if(len(str(i)) < 3):
            for j in range(3 - len(str(i))):
                num = '0' + num
        nums.append(num)
    size = os.path.getsize(fname)
    divide = size/n #math.ceil(size/n)
    #print(divide)
    f = open(fname)
    file_num = 0
    currFileSize = 0
    line_num = 0
    for line in f:
        line_num += 1
        line = line.replace('\n', '')
        #first file open
        if(currFileSize == 0 or currFileSize > divide and file_num != n-1):
            new_file = open(fname + '_' + nums[file_num] + '.txt', 'wt')
            line_size = len(line)
            new_file.write(line)
            currFileSize += line_size
        #still printing to current file
        else:
            line_size = len(line)
            if(line_size + currFileSize + 1 < divide or file_num == n-1):
                new_file.write('\n')
                currFileSize += 1
                new_file.write(line)
                currFileSize += line_size
            else:
                new_file.close()
                print(os.path.getsize(fname + '_' + nums[file_num] + '.txt'))
                file_num += 1
                new_file = open(fname + '_' + nums[file_num] + '.txt', 'wt')
                line_size = len(line)
                new_file.write(line)
                currFileSize = line_size
    print(os.path.getsize(fname + '_' + nums[file_num] + '.txt'))
    new_file.close()
