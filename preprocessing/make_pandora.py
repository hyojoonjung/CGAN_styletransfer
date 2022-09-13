import random
import pandas as pd
import numpy as np
from tqdm import tqdm

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

##AUTHOR LIST ##
author_profiles = pd.read_csv('../data/pandora/raw/author_profiles.csv')
author_df = author_profiles[['author', 'agreeableness','openness', 'conscientiousness', 'extraversion', 'neuroticism']]
# df = author_profiles[['author', 'mbti']]
author_df = author_df.dropna(thresh=2)
# df.to_csv('pandora/author_MBTI.csv', index=False)
print("----AUTHOR LIST----")
print(len(author_profiles))
print(len(author_df))


df = pd.read_csv('../data/pandora/new_comments.csv', index_col=False)
author_df = pd.read_csv('../data/pandora/new_author_OCEAN.csv',index_col=False)


## Type 1 35702 ##
# next_df = []
# next_df = pd.DataFrame(next_df)
# new_df = []
# new_df = pd.DataFrame(new_df)
# for i in range(len(author_df)):
#     author = author_df['author'].iloc[i]
#     if (df['author'] == author).any():
#         index = df[df['author'] == author]
#         temp = pd.DataFrame(index)
#         temp = temp.drop_duplicates(['body'])
#         new_df = temp.reset_index()
#         new_df['OPN'] = [author_df[author_df['author'] == author]['openness'].item() for i in range(len(temp))]
#         new_df['CON'] = [author_df[author_df['author'] == author]['conscientiousness'].item() for i in range(len(temp))]
#         new_df['EXT'] = [author_df[author_df['author'] == author]['extraversion'].item() for i in range(len(temp))]
#         new_df['AGR'] = [author_df[author_df['author'] == author]['agreeableness'].item() for i in range(len(temp))]
#         new_df['NEU'] = [author_df[author_df['author'] == author]['neuroticism'].item() for i in range(len(temp))]
#         next_df = pd.concat([next_df, new_df])
# next_df = next_df.drop(['index'], axis=1)
    
# next_df.to_csv('data/pandora/pandora_type_1.csv',index=False)

##Make sentence##
from nltk.tokenize import sent_tokenize
import pandas as pd

new_df = {'author': [], 'body': []}

for i in range(len(df)):
    author = df['author'].iloc[i]
    text = sent_tokenize(df['body'].iloc[i])
    for sen in text:
        new_df['body'].append(sen)
        new_df['author'].append(author)

new_df = pd.DataFrame(new_df)
print("----Make Sentence----")
print(len(df))
print(len(new_df))
# new_df.to_csv('line_comments.csv', index=False)

## float ##
df = new_df.copy()
df = df.reset_index(drop=True)
del new_df
float_idx = []
for i in range(len(df)):
    if type(df['body'].iloc[i]) == float:
        float_idx.append(i)
print("----float----")
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
    if is_not_ascii(df['body'].iloc[i]):
        not_ascii_idx.append(i)
print("----ASCII----")
print(len(df))
new_df = df.drop(not_ascii_idx, axis=0)
print(len(new_df))
# new_df.to_csv('line_comments_ascii.csv', index=False)

## delete nan ##
df = new_df.copy()
df = df.reset_index(drop=True)
del new_df
print("----nan----")
print(len(df))
new_df = df.dropna(subset=['body'])
print(len(new_df)) #-5
# new_df.to_csv('data/pandora/line_comments_ascii_null.csv',index=False)

## drop duplicate ##
df = new_df.copy()
df = df.reset_index(drop=True)
del new_df
authors = df['author'].unique()
new_df = []
new_df = pd.DataFrame(new_df)
for author in authors:
    if (df['author'] == author).any():
        index = list(df[df['author'] == author].index)
        temp = df['body'][index]
        temp = pd.DataFrame(temp)
        temp['author'] = df['author'][index]
        temp = temp.drop_duplicates('body')
        new_df = pd.concat([new_df, temp])
print("----duplicate----")
print(len(df))
print(len(new_df))
# new_df.to_csv('line_ascii_null_dupli_comments.csv',index=False)

## drop 3 tokens##
df = new_df.copy()
df = df.reset_index(drop=True)
del new_df
drop_idx = []
for i in range(len(df)):
    text = df['body'].iloc[i]
    s = text.split()
    if len(s) < 3:
        drop_idx.append(i)
new_df = df.drop(drop_idx, axis=0)
print("----3 tokens----")
print(len(df))
print(len(new_df))

## Type 2 85551 ##
df = new_df.copy()
df = df.reset_index(drop=True)
del new_df
next_df = []
next_df = pd.DataFrame(next_df)
new_df = []
new_df = pd.DataFrame(new_df)
for i in tqdm(range(len(author_df))):
    author = author_df['author'].iloc[i]
    if (df['author'] == author).any():
        index = list(df[df['author'] == author].index)
        num_sample = index
        for i in range(len(index)):
            temp = {'TEXT' : []}

            if len(num_sample) > 30:
                temp_text = ""
                a = random.sample(num_sample, random.randint(2,30))
                for idx, j in enumerate(a):
                    temp_text += df['body'].iloc[j]
                    if idx != len(a)-1:
                        temp_text += " "
                temp['TEXT'].append(temp_text)
                num_sample = [x for x in num_sample if x not in a]

            elif 2 <= len(num_sample) <= 30:
                temp_text = ""
                a = random.sample(num_sample, random.randint(2,len(num_sample)))
                for idx, j in enumerate(a):
                    temp_text += df['body'].iloc[j]
                    if idx != len(a)-1:
                        temp_text += " "
                temp['TEXT'].append(temp_text)
                num_sample = [x for x in num_sample if x not in a]

            elif len(num_sample) < 2:
                # temp_text = ""
                # a = random.sample(num_sample, (len(num_sample)))
                # for idx, j in enumerate(a):
                #     temp_text += df['body'].iloc[j]
                #     if idx != len(a)-1:
                #         temp_text += " "
                # temp['TEXT'].append(temp_text)

                break
            temp_df = pd.DataFrame(temp)
            temp_df['author'] = author

            new_df = temp_df.reset_index()
            new_df['OPN'] = [author_df[author_df['author'] == author]['openness'].item() for i in range(len(temp_df))]
            new_df['CON'] = [author_df[author_df['author'] == author]['conscientiousness'].item() for i in range(len(temp_df))]
            new_df['EXT'] = [author_df[author_df['author'] == author]['extraversion'].item() for i in range(len(temp_df))]
            new_df['AGR'] = [author_df[author_df['author'] == author]['agreeableness'].item() for i in range(len(temp_df))]
            new_df['NEU'] = [author_df[author_df['author'] == author]['neuroticism'].item() for i in range(len(temp_df))]
            next_df = pd.concat([next_df, new_df])

next_df = next_df.drop(['index'], axis=1)
next_df = next_df[['author', 'TEXT', 'OPN', 'CON', 'EXT', 'AGR', 'NEU']]

print("----dropna----")
print(len(next_df))
next_df = next_df.dropna()
print(len(next_df))

next_df.to_csv('../pandora.csv',index=False)