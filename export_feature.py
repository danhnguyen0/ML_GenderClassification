import pickle
from text_processing import *
from data_process import *

columns = read_csv()
columns = remove_rows_with_none_elements(columns)

with open('columns.pkl', 'wb') as f:
    pickle.dump(columns, f)

final_correct_spelling_data = []
final_incorrect_spelling_count = []    
correct_spelling_data = []
incorrect_spelling_count_data = [] 

for i in range(len(columns[0])):
    correct_string, incorrect_count = remove_incorrectly_spelled_words(columns[0][i])
    print(correct_string, incorrect_count)
    correct_spelling_data.append(correct_string)
    incorrect_spelling_count_data.append([incorrect_count])

final_correct_spelling_data.append(correct_spelling_data)
final_correct_spelling_data.append(columns[1])
final_incorrect_spelling_count.append(incorrect_spelling_count_data)
final_incorrect_spelling_count.append(columns[1])    
with open('correct_spelling_data.pkl', 'wb') as f:
    pickle.dump(final_correct_spelling_data, f)
    
with open('incorrect_spelling_count_data.pkl', 'wb') as f:
    pickle.dump(final_incorrect_spelling_count, f)