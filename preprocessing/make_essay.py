import random
import pandas as pd
import numpy as np
from tqdm import tqdm

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

df = pd.read_csv('/home/hjjung/ailab/VAE/data/essay.csv', index_col=False, encoding='utf-8')

##Make sentence##
from nltk.tokenize import sent_tokenize
import pandas as pd

new_df = {'ID': [], 'TEXT': [], 'OPN': [], 'CON': [], 'EXT': [], 'AGR': [], 'NEU': []}

for i in range(len(df)):
    ID = df['#AUTHID'].iloc[i]
    text = sent_tokenize(df['TEXT'].iloc[i])
    OPN = df['cOPN'].iloc[i]
    CON = df['cCON'].iloc[i]
    EXT = df['cEXT'].iloc[i]
    AGR = df['cAGR'].iloc[i]
    NEU = df['cNEU'].iloc[i]
    for sen in text:
        new_df['TEXT'].append(sen)
        new_df['ID'].append(ID)
        new_df['OPN'].append(OPN)
        new_df['CON'].append(CON)
        new_df['EXT'].append(EXT)
        new_df['AGR'].append(AGR)
        new_df['NEU'].append(NEU)
new_df = pd.DataFrame(new_df)
print("----Make Sentence----")
print(len(df))
print(len(new_df))
# new_df.to_csv('line_essay.csv', index=False)

## float ##
df = new_df.copy()
df = df.reset_index(drop=True)
del new_df
float_idx = []
for i in range(len(df)):
    if type(df['TEXT'].iloc[i]) == float:
        float_idx.append(i)
print("----Float----")
print(len(df))
new_df = df.drop(float_idx, axis=0)
print(len(new_df))
# new_df.to_csv('comments.csv', index=False)

## Check ASCII ##
df = new_df.copy()
df = df.reset_index(drop=True)
del new_df
def is_not_ascii(string):
    return string is not None and any([ord(s) >= 128 for s in string])

not_ascii_idx = []
for i in range(len(df)):
    if is_not_ascii(df['TEXT'].iloc[i]):
        not_ascii_idx.append(i)
print("----ASCII----")
print(len(df))
new_df = df.drop(not_ascii_idx, axis=0)
print(len(new_df))
# new_df.to_csv('line_ascii_essay.csv', index=False)

## delete nan ##
df = new_df.copy()
df = df.reset_index(drop=True)
del new_df
print("----NAN----")
print(len(df))
new_df = df.dropna(subset=['TEXT'])
print(len(new_df))
# new_df.to_csv('line_ascii_null_essay.csv',index=False)

## drop 3 tokens##
df = new_df.copy()
df = df.reset_index(drop=True)
del new_df
drop_idx = []
for i in range(len(df)):
    text = df['TEXT'].iloc[i]
    s = text.split()
    if len(s) < 3:
        drop_idx.append(i)
new_df = df.drop(drop_idx, axis=0)
print("----3 tokens----")
print(len(df))
print(len(new_df))

# ## drop duplicate ##
# df = new_df.copy()
# del new_df
# IDs = df['ID'].unique()
# new_df = []
# new_df = pd.DataFrame(new_df)
# for ID in IDs:
#     if (df['ID'] == ID).any():
#         index = list(df[df['ID'] == ID].index)
#         temp = df['TEXT'][index]
#         temp = pd.DataFrame(temp)
#         temp['ID'] = df['ID'][index]
#         temp = temp.drop_duplicates('TEXT')
#         new_df = pd.concat([new_df, temp])
# print("----duplicate----")
# print(len(df))
# print(len(new_df))
# new_df.to_csv('line_ascii_null_dupli_essay.csv',index=False)

## Type 2 85551 ##
df = new_df.copy()
df = df.reset_index(drop=True)
del new_df
next_df = []
next_df = pd.DataFrame(next_df)
new_df = []
new_df = pd.DataFrame(new_df)
IDs = df['ID'].unique().tolist()
for ID in tqdm(IDs):
    author = ID
    if (df['ID'] == author).any():
        index = list(df[df['ID'] == author].index)
        num_sample = index
        for i in range(len(index)):
            temp = {'TEXT' : []}

            if len(num_sample) > 30:
                temp_text = ""
                a = random.sample(num_sample, random.randint(2,30))
                for idx, j in enumerate(a):
                    temp_text += df['TEXT'].iloc[j]
                    if idx != len(a)-1:
                        temp_text += " "
                temp['TEXT'].append(temp_text)
                num_sample = [x for x in num_sample if x not in a]

            elif 2 <= len(num_sample) <= 30:
                temp_text = ""
                a = random.sample(num_sample, random.randint(2,len(num_sample)))
                for idx, j in enumerate(a):
                    temp_text += df['TEXT'].iloc[j]
                    if idx != len(a)-1:
                        temp_text += " "
                temp['TEXT'].append(temp_text)

                num_sample = [x for x in num_sample if x not in a]

            elif len(num_sample) < 2:
                # temp_text = ""
                # a = random.sample(num_sample, (len(num_sample)))
                # for idx, j in enumerate(a):
                #     temp_text += df['TEXT'].iloc[j]
                #     if idx != len(a)-1:
                #         temp_text += " "
                # temp['TEXT'].append(temp_text)

                break
            temp_df = pd.DataFrame(temp)
            temp_df['ID'] = author

            new_df = temp_df.reset_index()

            new_df['OPN'] = df[df['ID'] == author]['OPN'].iloc[0]
            new_df['CON'] = df[df['ID'] == author]['CON'].iloc[0]
            new_df['EXT'] = df[df['ID'] == author]['EXT'].iloc[0]
            new_df['AGR'] = df[df['ID'] == author]['AGR'].iloc[0]
            new_df['NEU'] = df[df['ID'] == author]['NEU'].iloc[0]
            next_df = pd.concat([next_df, new_df])

next_df = next_df.drop(['index'], axis=1)
next_df = next_df[['ID', 'TEXT', 'OPN', 'CON', 'EXT', 'AGR', 'NEU']]
print(len(next_df))
    
next_df.to_csv('../new_essay.csv',index=False)











