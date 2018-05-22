import wikipedia
import urllib
import requests
import json
from lxml import html
from bs4 import BeautifulSoup
import re
import numpy as np
import codecs

urlfile = 'urls.txt'

url1 = 'https://en.wiktionary.org/wiki/Category:Arabic_terms_belonging_to_the_root'

urlex = 'https://en.wiktionary.org/wiki/Category:Arabic_terms_belonging_to_the_root_%D8%A1_%D8%A8_%D9%82'

special = ['/wiki/Special:Categories', '/wiki/Category:Arabic_roots', '/wiki/Category:Arabic_terms_by_etymology']
special2 = ['Special:Categories', 'Category:Arabic_terms_by_root', 'Category:Arabic_3-letter_roots', 'Category:Arabic_language', 'Category:Arabic_4-letter_roots', 'Category:Empty_categories']

rootfile = 'root.txt'
datafile = 'data.txt'
datafile2 = 'data2.txt'
trainfile = 'train_b.tsv'
evalfile = 'eval_b.tsv'

# urllist = []
# with open(urlfile) as infile:
#     for line in infile:
#         line = line.strip('\n')
#         urllist.append(line)
#
# dict = {}
#
# # file = open(rootfile,"w")
#
# for url in urllist:
#     html = urllib.request.urlopen(url)
#     bsObj = BeautifulSoup(html,'html.parser')
#
#     for link in bsObj.find('div',id = "bodyContent").find_all('a',href = re.compile("^(/wiki/)((?!;)\S)*$")):
#         root = link.attrs['href'].replace('/wiki/Category:Arabic_terms_belonging_to_the_root', '')
#         if root not in special:
#             print(root)
            # file.write(root + '\n')

#
#             url2 = url1 + root
#             html2 = urllib.request.urlopen(url2)
#             bsObj2 = BeautifulSoup(html2, 'html.parser')
#
#             list = []
#             for link in bsObj2.find('div', id="bodyContent").find_all('a', href=re.compile("^(/wiki/)((?!;)\S)*$")):
#                 result = link.attrs['href'].replace('/wiki/', '')
#                 if result not in special2 and 'Category' not in result:
#                     result = result.replace('#Arabic', '')
#                     print(result)
#                     list.append(result)
#                     # file.write(result + '\t')
#             dict[root] = set(list)
#
# with open(datafile2, 'w') as outfile:
#     for (k, v) in dict.items():
#         outfile.write(k + '\t')
#         outfile.write(','.join(v) + '\n')
# file.close()
#
#

# dict = {}
# with open(datafile) as infile, open(datafile2, 'w') as outfile:
#     for line in infile:
#         line = urllib.parse.unquote(line.replace('_', ''))
#         outfile.write(line)


list = []
with open(datafile2) as infile:
    for line in infile:
        [root, words] = line.strip('\n').split('\t')
        root = root.encode('utf-8')
        words = words.encode('utf-8')
        for word in words.split(b','):
            list += [([root[i:i+1] for i in range(len(root))], [word[j:j+1] for j in range(len(word))])]

cutoff = round(len(list) * 0.8)
order = np.random.permutation(np.arange(len(list)))
list = [list[i] for i in order]
train = list[:cutoff]
eval = list[cutoff:]


# with codecs.open(trainfile, 'w', "utf-8") as outfile1, codecs.open(evalfile, 'w', "utf-8") as outfile2:
with open(trainfile, 'wb') as outfile1, open(evalfile, 'wb') as outfile2:
    for (k, v) in train:
        if len(k) > 0 and len(v) > 0:
            outfile1.write(b' '.join(v) + b'\t' + b' '.join(k) + b'\n')
    for (k, v) in eval:
        if len(k) > 0 and len(v) > 0:
            outfile2.write(b' '.join(v) + b'\t' + b' '.join(k) + b'\n')