import pandas as pd
import os
import re
import itertools
import Levenshtein
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
import string
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem.porter import *
import numpy as np
stop_words = set(stopwords.words('english'))    # Remove stopwords
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

# Read all texts from 15 files and remove duplicate texts.
# directory_path = 'C:/Users/srajwal/Downloads/Semester 1/BioNLP/Assignment 1/Annotations'
#
# data = pd.DataFrame()
# for fname in os.listdir(directory_path):
#     if fname.endswith('.xlsx') and not fname.startswith('~$'):
#         df_temp = pd.read_excel(os.path.join(directory_path, fname))
#         data = pd.concat([data,df_temp],ignore_index=True)
#
data = pd.read_excel('UnlabeledSet.xlsx')
data = data.drop(['DATE','Symptom Expressions','Standard Symptom','Symptom CUIs','Negation Flag'], axis=1)
data = data.drop_duplicates(subset = ['ID'],keep='first')
data.reset_index(drop = True, inplace = True)
# data.to_excel('unique_s1_to_s15.xlsx', index = False)
print(data.shape)
# print("Repeated texts droped!")
def in_scope(neg_end, text,symptom_expression):
    negated = False
    text_following_negation = text[neg_end:]
    tokenized_text_following_negation = list(nltk.word_tokenize(text_following_negation))
    three_terms_following_negation = ' '.join(tokenized_text_following_negation[:min(len(tokenized_text_following_negation),3)])
    match_object = re.search(symptom_expression,three_terms_following_negation)
    if match_object:
        period_check = re.search('\.',three_terms_following_negation)
        next_negation = 100000  #starting with a very large number
        for neg in negations:
            if re.search(neg,text_following_negation):
                index = text_following_negation.find(neg)
                if index<next_negation:
                    next_negation = index
        if period_check:
            #if the period occurs after the symptom expression
            if period_check.start() > match_object.start() and next_negation > match_object.start():
                negated = True
        else:
            negated = True
    return negated

symptom_dict = {}
symptom_lexicons = open('COVID-Twitter-Symptom-Lexicon.txt')
for line in symptom_lexicons:
  listt = line.split('\t')
  # print(listt)   # example output: ['Lymphadenopathy', 'C0497156', 'swollen nodes']
  key = str.strip(listt[2])   # non-standard expressions / Symptom expressions
  value = [str.strip(listt[1]),str.strip(listt[0])]   # CUI, standard symptom
  symptom_dict[key.lower()] = value
  symptom_dict = dict(sorted(symptom_dict.items()))

def preprocessor(text): # Not doing any preprocessing because we need to match the exact symptom from the given list that contains stopwords
  text = text.replace("\n", " ").lower()
  # Note that . had to be retained.
  punctuations_to_remove = ['!', '"', '#', "'", '$', '%', '&', '(', ')', '*', '+', ',',
                            '-', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\',
                            ']', '^', '_', '`', '{', '|', '}', '~']
  result_string = ""
  for char in text:   # Iterate through each character in the input string
      if char not in punctuations_to_remove:    # Check if the character is not in the list of punctuations
          result_string += char   # If it's not a punctuation, add it to the result string
  return result_string
  # return (' '.join(str(x) for x in processed_string))

# removed apostrophe mark from negations list 
negations = ['no', 'not', 'without', 'absence of', 'cannot', 'couldnt', 'could not', 'didnt', 'did not', 'denied',
             'denies', 'free of', 'negative for', 'never had', 'resolved', 'exclude', 'with no', 'rule out', 'free',
             'aside from', 'except', 'apart from']

final_df = pd.DataFrame(columns=['ID', 'Symptom Expressions', 'Standard Symptom', 'Symptom CUIs', 'Negation Flag'])
final_fuzzy_df = pd.DataFrame(columns=['ID', 'Symptom Expressions', 'Standard Symptom', 'Symptom CUIs', 'Negation Flag'])

def run_sliding_window_through_text(words, window_size):
    '''
    Generate a window sliding through a sequence of words
    '''
    word_iterator = iter(words) # creates an object which can be iterated one element at a time
    word_window = tuple(itertools.islice(word_iterator, window_size)) #islice() makes an iterator that returns selected elements from the the word_iterator
    yield word_window
    #now to move the window forward, one word at a time
    for w in word_iterator:
        word_window = word_window[1:] + (w,)
        yield word_window

for i in range(len(data)):
    print("*********************************")
    print("Loop no: ",i)
    post_id = data['ID'][i]
    post_org = data['TEXT'][i]  # Original post
    print(post_id)
    posts1 = preprocessor(post_org)  # Lowercased post (no stopwords removed)
    # print(posts1)
    sentences = sent_tokenize(posts1)  # Posts converted to sentences
    matched_tuples = []   # Matched_tuples will be a list of tuples that will store the match information so that it can be processed later by a negation scoping function
    fuzzy_matched = []
    for s in sentences:
        threshold = 0.70
        max_similarity_obtained = -1
        best_match = ''
        for symptom in symptom_dict.keys():   # symptom here is a Non standard expression
            CUI = symptom_dict[symptom][0]
            standard_symptom = symptom_dict[symptom][1]

            for match in re.finditer(r'\b'+symptom+r'\b',s):
                # CUI = symptom_dict[symptom][0]
                # standard_symptom = symptom_dict[symptom][1]
                match_tuple = (s,CUI,match.group(),match.start(),match.end(),post_id, standard_symptom)
                matched_tuples.append(match_tuple)

            size_of_window = len(symptom.split())
            tokenized_text = list(nltk.word_tokenize(s))
            # print(tokenized_text)
            for window in run_sliding_window_through_text(tokenized_text, size_of_window):
                window_string = ' '.join(window)
                similarity_score = Levenshtein.ratio(window_string, symptom)
                if similarity_score >= threshold:
                    # print(similarity_score, '\t', symptom, '\t', window_string)
                    if similarity_score > max_similarity_obtained:
                        max_similarity_obtained = similarity_score
                        best_match = window_string
                        match = re.search(best_match, s)
                        if match:
                            start_index = match.start()
                            end_index = match.end()
                        fuzz_match_tuple = (s,CUI,best_match, start_index,end_index,post_id, standard_symptom)
                        # print(fuzz_match_tuple)
                        fuzzy_matched.append(fuzz_match_tuple)

    cui_text_list = []
    for mt in matched_tuples:
        is_negated = False
        text = mt[0]
        cui = mt[1]
        expression = mt[2]   # Non standard expression
        start = mt[3]
        end = mt[4]
        post_id = mt[5]
        standard_symp = mt[6]
        for neg in negations:
            for match in re.finditer(r'\b'+neg+r'\b', text):          #check if the negation matches anything in the text of the tuple
                is_negated = in_scope(match.end(),text,expression)
                if is_negated:
                    cui_text_list.append((post_id,expression,standard_symp,cui,1))
                    break
        if not is_negated:
            cui_text_list.append((post_id,expression,standard_symp,cui,0))
    if(cui_text_list):
        temp = pd.DataFrame(cui_text_list)
        temp_result_df = temp.groupby(0).agg(lambda x: '$$$'+'$$$'.join(map(str, x))+'$$$')
        temp_result_df = temp_result_df.reset_index()
        print(cui_text_list)
        temp_result_df.columns = ['ID', 'Symptom Expressions', 'Standard Symptom', 'Symptom CUIs', 'Negation Flag']
        final_df = pd.concat([final_df,temp_result_df], axis=0)
    else:
        new_record = pd.DataFrame([{'ID': post_id, 'Symptom Expressions': '$$$$$$', 'Standard Symptom': '$$$$$$', 'Symptom CUIs': '$$$$$$', 'Negation Flag': '$$$$$$'}])
        final_df = pd.concat([final_df, new_record], ignore_index=True)

    cui_text_list = []
    for mt in fuzzy_matched:
        is_negated = False
        text = mt[0]
        cui = mt[1]
        expression = mt[2]  # Non standard expression
        start = mt[3]
        end = mt[4]
        post_id = mt[5]
        standard_symp = mt[6]
        for neg in negations:
            for match in re.finditer(r'\b' + neg + r'\b',
                                     text):  # check if the negation matches anything in the text of the tuple
                is_negated = in_scope(match.end(), text, expression)
                if is_negated:
                    cui_text_list.append((post_id, expression, standard_symp, cui, 1))
                    break
        if not is_negated:
            cui_text_list.append((post_id, expression, standard_symp, cui, 0))
    if (cui_text_list):
        temp = pd.DataFrame(cui_text_list)
        temp_result_df = temp.groupby(0).agg(lambda x: '$$$' + '$$$'.join(map(str, x)) + '$$$')
        temp_result_df = temp_result_df.reset_index()
        temp_result_df.columns = ['ID', 'Symptom Expressions', 'Standard Symptom', 'Symptom CUIs', 'Negation Flag']
        final_fuzzy_df = pd.concat([final_fuzzy_df, temp_result_df], axis=0)

'''Given a set of Reddit posts, the system will output a text file that contains, tab separated, the id of a post, the symptom expression 
(or entire negated symptom expression), the CUI for the symptom, and a flag indicating negation (i.e., 1 if the symptom expression is a negated one, 0 otherwise)'''

final_df.to_excel('final_annotated_file_on_unlabeled_dataset2.xlsx',index = False)
# final_fuzzy_df.to_excel('unique_threshold_70_fuzzy_final_annotated_file.xlsx',index = False)

## Run EvaluationScript.py
################# Following code was used to plot certain charts for the report! ####################
# df_plot = pd.read_excel('EvaluationGoldStandardSet.xlsx')
# # df_plot = pd.read_excel('unique_final_annotated_file.xlsx')
# # dictionary = {}
# dictionary_negation_flags = {}
# for i in range(len(df_plot)):
#     # temp_symps = df_plot['Standard Symptom'][i]
#     temp_flags = df_plot['Negation Flag'][i]
#     print(i)
#     print(temp_flags)
#     print("************************")
#
#     # elements = temp_symps.split("$$$")
#     # elements = [element for element in elements if element.strip()]
#     neg_flags = temp_flags.split("$$$")
#     neg_flags = [neg_flag for neg_flag in neg_flags if neg_flag.strip()]
#
#     # for element in elements:
#     #     if element in dictionary:
#     #         dictionary[element] +=1
#     #     else:
#     #         dictionary[element] = 1
#     for neg_flag in neg_flags:
#         if neg_flag in dictionary_negation_flags:
#             dictionary_negation_flags[neg_flag] +=1
#         else:
#             dictionary_negation_flags[neg_flag] = 1

# 'Anxiety, stress & general mental health symptoms' is a very big title. So reduced it to 'ASG mental health symptoms'

# import matplotlib.pyplot as plt
#
# dictionary['ASG mental health symptoms'] = dictionary.pop('Anxiety, stress & general mental health symptoms')
# dictionary = dict(sorted(dictionary.items(), key=lambda item: item[1],reverse=True))
# print(dictionary)
#
# Extract keys (categories) and values (data) from the dictionary
# categories = list(dictionary.keys())
# values = list(dictionary.values())
#
# # Create a bar chart
# plt.barh(categories, values)
# plt.yticks(fontsize = 8)
# plt.grid(True, axis='x', linestyle='--', alpha=0.7)
#
# # Add labels and a title
# plt.ylabel('Standard Symptom')
# plt.xlabel('Frequency')
# plt.title('Bar Chart of Symptom Distribution')
#
# # Show the chart
# plt.show()
# print(dictionary_negation_flags)
# labels = list(dictionary_negation_flags.keys())
# sizes = list(dictionary_negation_flags.values())
# plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
# plt.title("Negation flag distributions: Given dataset")
# plt.show()
#
#
