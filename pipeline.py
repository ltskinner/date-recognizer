



import os
import re 
import numpy as np 
import pickle

from datetime import datetime

import spacy
nlp = spacy.load('en_vectors_web_lg')
nlp.add_pipe(nlp.create_pipe('sentencizer'))
print("[+] spacy imported")

"""
Something to keep in mind abbreviations of months is throwing false positives in labeling
index filter to see if there is a neighbor label? years ok to not have neighbors
top priority for this coming week, network works. NETWORK WORKS!
    unfortunately may need to regex the hell out for his

Also, there are far too many [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
so realiztically, a corpus to find each word isnt super reasonable, so lets tinker with fuzzy and see what we can get
    honestly, I think some weight is better than ZERO weight
"""

def stFinder(text):
    st = [
        '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th', 
        '11th', '12th', '13th', '14th', '15th', '16th', '17th', '18th', '19th', 
        '20th', '21st', '22nd', '23rd', '24th', '25th', '26th', '27th', '28th', '29th', '30th', '31st'
    ]

    for i in st:
        if i in text.lower():
            return True
    return False

def moAbbr(raw):
    text = raw.lower()
    # 'may', 
    mo = ['jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    # ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    alf = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    for i in mo:
        if i in text:
            if len(text) == 3:
                return True
            else:
                s = text.find(i)
                if s == 0:
                    #  0  1  2  3
                    # [d, e, c, i]
                    if text[3] not in alf:
                        return True
                elif text[s-1] not in alf:
                    if len(text) == 4:
                        return True
                    else:
                        try:
                            if text[s+3] not in alf:
                                return True
                        except:
                            return True
    return False


def textMonthFinder(text):
    mo = [ 'january', 'february', 'march', 'april', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
    #'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'

    for i in mo:
        if i in text.lower():
            return True
    return False

# Universal activate
def year4d(text): 
    # ensuring nothing longer than 4 dig date passes now
    if len(re.findall(r"[0-9]{4}", text)) == 1 and len(re.findall(r"[0-9]{5}", text)) == 0:
        return True
    return False

def yrmo2d(text):
    if len(re.findall(r"[0-9]{2}", text)) == 1:
        return True
    return False

def noDash(p1, p2):
    if '-' not in p1.text and '-' not in p2.text:
        return True
    return False

"""
Need to index not literal iteration
Rules:
1) Any 4 digit number is a year, index is activated
2) Any literal month except for may, index is activated
3) Need exceptions for '-'
4) lat longs are thowing it off too lmao
5) Better word vectors? https://spacy.io/models/en#en_vectors_web_lg
"""

def label2text(label, text):
    buffer = []
    for c, word in enumerate(nlp(text)):
        if label[c] > .5:
            buffer.append(word.text)
            #print("-->", word.text)
            #print(word.vector)
    
    print("--> " + str(' '.join(buffer)).replace(' - ', '-'))

# 3) Label the data
def labelMaker(sentences):
    # dtype as spacy objects remember
    labels = []
    for text in sentences:
        #print("----------------------------------------------------")
        label = np.zeros(56)
        """
        [!] Please dont judge the horrendus if/else tree below [!]
        [!] Efficiency and clarity went out the window, I just want this damn data labeled ahah [!]
        """
        # finds the ends lel
        for i in [0, len(text)-1]:
            word = text[i]
            if year4d(word.text):
                label[i] = 1
                #print("[+] YEAR")
                #print(">>", word.text)
            elif yrmo2d(word.text):
                if '/' in word.text:
                    label[i] = 1
                    #print("[+] YEAR or MONTH")
                    #print(">>", word.text)
                else:
                    if year4d(text[i+1].text) or year4d(text[i-1].text):
                        label[i] = 1
                        #print("[+] Lone 2D")
                        #print(">>", word.text)
                    elif yrmo2d(text[i+1].text) or yrmo2d(text[i-1].text):
                        label[i] = 1
                        #print("[+] Lone 2D")
                        #print(">>", word.text)
                    elif textMonthFinder(text[i+1].text) or textMonthFinder(text[i-1].text):
                        label[i] = 1
                        #print("[+] Lone 2D")
                        #print(">>", word.text)

            elif textMonthFinder(word.text):
                label[i] = 1
                #print("[+] MONTH")
                #print(">>", word.text)
            elif moAbbr(word.text):
                label[i] = 1
            elif stFinder(word.text):
                label[i] = 1
                #print("[+] ST")
                #print(">>", word.text)
            
            #print(">>>", word.text)


        for i in range(1, len(text)-1):
            word = text[i]
            nodash = noDash(text[i-1], text[i+1])
            if nodash and year4d(word.text):
                label[i] = 1
                #print("[+] YEAR")
                #print(">>", word.text)
            elif nodash and yrmo2d(word.text):
                if '/' in word.text:
                    label[i] = 1
                    #print("[+] YEAR or MONTH")
                    #print(">>", word.text)
                else:
                    if year4d(text[i+1].text) or year4d(text[i-1].text):
                        label[i] = 1
                        #print("[+] Lone 2D")
                        #print(">>", word.text)
                    elif yrmo2d(text[i+1].text) or yrmo2d(text[i-1].text):
                        label[i] = 1
                        #print("[+] Lone 2D")
                        #print(">>", word.text)
                    elif textMonthFinder(text[i+1].text) or textMonthFinder(text[i-1].text):
                        label[i] = 1
                        #print("[+] Lone 2D")
                        #print(">>", word.text)
            elif nodash and textMonthFinder(word.text):
                label[i] = 1
                #print("[+] MONTH")
                #print(">>", word.text)
            elif moAbbr(word.text):
                label[i] = 1
            elif nodash and stFinder(word.text):
                if textMonthFinder(text[i-1].text) or textMonthFinder(text[i+1].text) or moAbbr(text[i-1].text) or moAbbr(text[i+1].text) or text[i-1].text.lower() == 'may' or text[i+1].text.lower() == 'may':
                    label[i] = 1
                    #print("[+] ST")
                    #print(">>", word.text)
            
            #print(">>>", word.text)
            elif '-' in word.text:
                # 1d's
                if text[i-1].text in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    label[i] = 1
                    label[i-1] = 1
                    #print("[+] XX-XXXX")
                    #print(">>", text[i-1].text)
                if text[i+1].text in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    label[i] = 1
                    label[i+1] = 1
                    #print("[+] XX-XXXX")
                    #print(">>", text[i+1].text)
                # 2d's
                if yrmo2d(text[i-1].text):
                    label[i] = 1
                    label[i-1] = 1
                    #print("[+] XX-XXXX")
                    #print(">>", text[i-1].text)
                if yrmo2d(text[i+1].text):
                    label[i] = 1
                    label[i+1] = 1
                    #print("[+] XX-XXXX")
                    #print(">>", text[i+1].text)
                # 4d's
                if year4d(text[i-1].text):
                    label[i] = 1
                    label[i-1] = 1
                    #print("[+] XX-XXXX")
                    #print(">>", text[i-1].text)
                if year4d(text[i+1].text):
                    label[i] = 1
                    label[i+1] = 1
                    #print("[+] XX-XXXX")
                    #print(">>", text[i+1].text)
                
                # texts
                if textMonthFinder(text[i-1].text):
                    label[i] = 1
                    label[i-1] = 1
                    #print("[+] XX-XXXX")
                    #print(">>", text[i-1].text)
                elif moAbbr(text[i-1].text):
                    label[i] = 1
                    label[i-1] = 1
                if textMonthFinder(text[i+1].text):
                    label[i] = 1
                    label[i+1] = 1
                    #print("[+] XX-XXXX")
                    #print(">>", text[i+1].text)
                elif moAbbr(text[i+1].text):
                    label[i] = 1
                    label[i+1] = 1
                
            elif year4d(word.text):
                label[i] = 1
                #print("[+] YEAR")
                #print(">>", word.text)
            

            """
            if isdate(word.text):
                label.append(1)
            else:
                label.append()
            p2 = p1
            p1 = word.text
            """
        for i in range(len(text)):
            if text[i].text in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                try:
                    if label[i+1] == 1 or text[i+1].text.lower() == 'may':
                        label[i] = 1
                except:
                    pass
                try:
                    if label[i-1] == 1 or text[i-1].text.lower() == 'may':
                        label[i] = 1
                except:
                    pass 


        for i in range(len(text)):
            if text[i].text.lower() == "may":
                try:
                    if label[i+1] == 1:
                        label[i] = 1
                except:
                    pass
                try:
                    if label[i-1] == 1:
                        label[i] = 1
                except:
                    pass 
        
        for i in range(len(text)):
            try:
                if label[i+1] == 1 and label[i-1] == 1:
                    label[i] = 1
            except:
                pass


        # remember to un nlp() the function if using in this instance
        #label2text(label, text)

        labels.append(label)


    return labels


# 2) break into sentences using spacy
def sentsplit(text, nlp):
    doc = nlp(text)
    touse = []
    for sent in doc.sents:
        if len(sent) <= 56:
            touse.append(sent)
    return touse


# 1) Index all the subfolders in the seed data dir
def findText(rootdir, nlp):
    start = datetime.now()
    sentences = []

    for subdir in os.listdir(os.getcwd() + '/' + rootdir):
        for f in os.listdir(os.getcwd() + '/' + rootdir + subdir):
            fpath = os.getcwd() + '/' + rootdir + subdir + '/' + f
            print(fpath)
            with open(fpath, 'r', encoding="utf-8") as f:
                sents = sentsplit(f.read(), nlp)
                for s in sents:
                    sentences.append(s)
            """
            f = open(fpath, 'r', encoding="utf-8")
            data = f.read()
            f.close()
            sentences.append(sentsplit(data, nlp))
            """

    labeled = labelMaker(sentences)

    texts = [sent.text for sent in sentences]

    saveframe = []
    zeroframe = []
    for i in range(len(labeled)):
        if 1 in labeled[i]:
            saveframe.append([labeled[i], texts[i]])
        else:
            zeroframe.append([labeled[i], texts[i]])


    print("Time:", datetime.now() - start)
    pickle.dump(saveframe, open("labeled-texts.dat", 'wb'), -1)
    pickle.dump(zeroframe, open("zeros.dat", 'wb'), -1)

    print("Labeled:", len(saveframe), "Zeros:", len(zeroframe), "pct:", len(saveframe)/(len(saveframe) + len(zeroframe)))



def ttSplit(filename, pct):
    data = pickle.load(open(filename, 'rb'))

    instance_indecies = list(range(len(data)))
    np.random.shuffle(instance_indecies)
    
    shuff = []
    for i in instance_indecies:
        if len(data[i][1]) > 0:
            shuff.append(data[i])

    idex = int(len(data)*pct)
    pickle.dump(shuff[:idex], open("train.dat", 'wb'), -1)
    pickle.dump(shuff[idex:], open("valid.dat", 'wb'), -1)

    #return shuff[:idex], shuff[idex:]



# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------- M*A*I*N --------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

# 1) Make the labels
#findText('bucket/', nlp) # LMAO took 30:41 to run ahha



# 2) Split into testing and training
ttSplit("labeled-texts.dat", .85)




"""
def testRes():
    data = pickle.load(open("labeled-texts.dat", 'rb'))

    for i in data:
        print("-----------------------")
        print(i[1])
        label2text(i[0], i[1])
        print(i[0])

testRes()
"""
