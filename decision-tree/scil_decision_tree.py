import pandas as pd
from sklearn import tree
import graphviz
from graphviz import Source
import numpy as np
from sklearn import preprocessing
from sklearn.tree import _tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import argparse
import os
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning)

def get_time(time_str):
    step1 = time_str.split(':')
    step2 = step1[1].split('.')
    minutes = int(step1[0])
    seconds = int(step2[0])
    milli = int(step2[1])
    if minutes == 0 and seconds == 0:
      seconds = 1
      milli = 0
    return minutes * 60 + seconds


parser = argparse.ArgumentParser(description='Build a decision tree')
parser.add_argument('compression_input',
                   help='CSV file containing compression methods and corresponding variables')
parser.add_argument('tree_output',
                   help='Decision tree output file')
parser.add_argument('method', type=int, choices=[1, 2, 3, 4],
                   help='Method to optimize for. 1: Group by kJ/CR 2: Group by Watt')
parser.add_argument('--image',
                   help='Export additionally as an image')
args = parser.parse_args()


dataset = pd.read_csv(args.compression_input)
# Drop row when Energy is Nan
dataset = dataset[np.isfinite(dataset['Energy_Joules'])]
dataset['DM4'].fillna(0, inplace=True)
dataset['DM3'].fillna(0, inplace=True)
dataset['DM2'].fillna(0, inplace=True)
dataset['DM1'].fillna(0, inplace=True)

# INFO: removes mafisc as this filter is not implemented in scil
dataset = dataset[dataset['Compressor'] != 'mafisc']

print('Energy consumtion: {}'.format(dataset['Energy_Joules'].sum()))


dataset['CR'] = dataset['FileSize_in_Folder_Before'] / dataset['FileSize_in_Folder_After']

if args.method == 1:
    dataset['kJ/CR'] = dataset['Energy_Joules'] / dataset['CR']
    dataset = dataset.sort_values("kJ/CR").groupby(["Variable_Name"], as_index=False).first()
elif args.method == 2:
    dataset['Watt'] = dataset['Energy_Joules'] / dataset['Elapsed_Time'].apply(get_time)
    dataset = dataset.sort_values("Watt").groupby(["Variable_Name"], as_index=False).first()
elif args.method == 3:
    dataset['CR/Time'] = dataset['CR'] / dataset['Elapsed_Time'].apply(get_time)
    dataset = dataset.sort_values("CR/Time").groupby(["Variable_Name"], as_index=False).last()

dataset = dataset.drop('FileSize_in_Folder_After', axis=1)
dataset = dataset.drop('FileSize_in_Folder_Before', axis=1)
dataset = dataset.drop('Nano_Seconds', axis=1)
dataset = dataset.drop('CPU_Usage', axis=1)
dataset = dataset.drop('Variable_Name', axis=1)
dataset = dataset.drop('Min_Value', axis=1)
dataset = dataset.drop('Max_Value', axis=1)
dataset = dataset.drop('Elapsed_Time', axis=1)
dataset = dataset.drop('Energy_Joules', axis=1)
dataset = dataset.drop('CR', axis=1)
dataset = dataset.drop('kJ/CR', axis=1, errors='ignore')
dataset = dataset.drop('Watt', axis=1, errors='ignore')
dataset = dataset.drop('Fill_Time', axis=1, errors='ignore')
dataset = dataset.drop('Allocation_Time', axis=1, errors='ignore')
dataset = dataset.drop('DataSpace_Class', axis=1, errors='ignore')
dataset = dataset.drop('CR/Time', axis=1, errors='ignore')


dataset = dataset.drop('Fill_Value', axis=1, errors='ignore')

dataset = dataset.drop('Type_Precision', axis=1, errors='ignore')
#dataset = dataset.drop('Number_of_Elements', axis=1, errors='ignore')
dataset = dataset.drop('DataType_Size', axis=1, errors='ignore')
dataset = dataset.drop('Number_of_Attributes', axis=1, errors='ignore')
dataset = dataset.drop('Storage_Layout', axis=1, errors='ignore')
dataset = dataset.drop('Fill_Value_Definition', axis=1, errors='ignore')


#dataset = dataset.drop('Elapsed_Time', axis=1)
dataset = dataset.drop('System_Time', axis=1)
dataset = dataset.drop('User_Time', axis=1)
#dataset = dataset.drop('Energy_Joules', axis=1)
print(dataset.columns)
# Generate numerical data for categorical values
X = pd.get_dummies(data=dataset, columns=['Data_Type']) # Fill_Value_Definition, Storage_Layout
X = X.drop('Compressor', axis=1)

Y = dataset['Compressor']
Y.tolist()
le = preprocessing.LabelEncoder()
le.fit(Y) 
Yy = le.transform(Y)

# Split test and training data
X_train, X_test, y_train, y_test = train_test_split(X, Y)


clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best') #  splitter='best', max_depth=2
# Generate the decision tree
print("Generating tree...")
clf = clf.fit(X_train, y_train)

# Accuracy score
y_predict = clf.predict(X_test)
print("Accuracy score: {}".format(accuracy_score(y_test, y_predict)))

# Export tree as an image
if args.image:
    graph = Source(tree.export_graphviz(clf, out_file=None, 
        feature_names=X.columns.values,  
        class_names=clf.classes_,  
        filled=True, rounded=True,  
        special_characters=True))
    pdf_bytes = graph.pipe(format='pdf')
    with open(args.image,'wb') as f:
        f.write(pdf_bytes)



# Process tree
columns = ';'.join(map(str, X_train.columns.values))
tree_classes = ''
first = True
for c in range(0, len(le.classes_)):
    if not first:
        tree_classes += ';'
    tree_classes += ''.join(map(str, le.inverse_transform([c])))
    first = False

# Export tree
# Get values from tree
left = clf.tree_.children_left
right = clf.tree_.children_right
thresholds = clf.tree_.threshold
features = clf.tree_.feature
values = clf.tree_.value

left_e = ';'.join(map(str, left))
right_e = ';'.join(map(str, right))
thresholds_e = ';'.join(map(str, thresholds))
features_e = ';'.join(map(str, features))
values_e = ';'.join(map(str, values))
values_e = values_e.replace('[','').replace(']','').replace('  ', ' ')

file = open(args.tree_output,"w")
file.write("#classes\n")
file.write(tree_classes)
file.write("\n#columns\n") 
file.write(columns)
file.write("\n#left\n") 
file.write(left_e)
file.write("\n#right\n") 
file.write(right_e)
file.write("\n#thresholds\n") 
file.write(thresholds_e)
file.write("\n#indices\n") 
file.write(features_e)
file.write("\n#values\n") 
file.write(values_e)
file.close() 

print()
print("Use by exporting path to tree e.g.")
print("$ export SCIL_DECISION_TREE_FILE={}".format(os.path.abspath(args.tree_output)))

