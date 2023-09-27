'''
BMI 550: Applied BioNLP
Assignment 1: Evaluation script

The file simply loads a submission file and a gold standard and
computes the recall, precision and F-score for the system

@author Abeed Sarker
email: abeed.sarker@dbmi.emory.edu


Created: 09/14/2023
'''
import pandas as pd
from collections import defaultdict



def load_labels(f_path):
    '''
    Loads the labels

    :param f_path:
    :return:
    '''
    labeled_df = pd.read_excel(f_path)
    labeled_dict = defaultdict(list)
    print("Size before processing: ",len(labeled_df))
    for index,row in labeled_df.iterrows():
        id_ = row['ID']
        if not pd.isna(row['Symptom CUIs']) and not pd.isna(row['Negation Flag']):
            cuis = row['Symptom CUIs'].split('$$$')[1:-1]
            neg_flags = row['Negation Flag'].split('$$$')[1:-1]
            for cui,neg_flag in zip(cuis,neg_flags):
                labeled_dict[id_].append(cui + '-' + str(neg_flag))
    return labeled_dict

infile = 'EvaluationGoldStandardSet.xlsx'
# infile = 's1_to_s15.xlsx'
# infile = 'fuzzy_final_annotated_file.xlsx'

# outfile = 'final_annotated_file.xlsx'
outfile = 'unique_final_annotated_file.xlsx'
# outfile = 's1_to_s15.xlsx'
# outfile = 'fuzzy_final_annotated_file.xlsx'
# outfile = 'unique_threshold_95_fuzzy_final_annotated_file.xlsx'
gold_standard_dict = load_labels(infile)
submission_dict = load_labels(outfile)

print("Size of gold standard dictionary:",len(gold_standard_dict))
print("Size of submission_dict:",len(submission_dict))

tp = 0
tn = 0
fp = 0
fn = 0
print(gold_standard_dict)
for k,v in gold_standard_dict.items():
    for c in v:
        try:
            if c in submission_dict[k]:
               tp+=1
            else:
                fn+=1
                print('{}\t{}\tfn'.format(k, c))
        except KeyError:#if the key is not found in the submission file, each is considered
                        #to be a false negative..
            fn+=1
            print('{}\t{}\tfn'.format(k, c))
    for c2 in submission_dict[k]:
        if not c2 in gold_standard_dict[k]:
            fp+=1
            print('{}\t{}\tfp'.format(k, c2))
print('True Positives:',tp, 'False Positives: ', fp, 'False Negatives:', fn)
recall = tp/(tp+fn)
precision = tp/(tp+fp)
f1 = (2*recall*precision)/(recall+precision)
print('Recall: ',recall,'\nPrecision:',precision,'\nF1-Score:',f1)
print('{}\t{}\t{}'.format(precision, recall, f1))

