# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:48:04 2019

@author: Skanda
"""

import javalang
import os
import pandas as pd
import pickle


def get_children(node, children):
    if isinstance(node, javalang.ast.Node):
        children_candidates = node.children
    else:
        children_candidates = node
    for candidate in children_candidates:
        if isinstance(candidate, javalang.ast.Node):
            children.append(candidate)
        if isinstance(candidate, (list, tuple)):
            get_children(candidate, children)
    return children
    
    
def SBT(node):
    sequence = ''
    children = get_children(node, [])
    if len(children) == 0:
        sequence += '( ' + str(node).split('(')[0] + ' ' + '( ' + '('.join(str(node).split('(')[1:]) + ')' + str(node).split('(')[0]
    else:
        sequence += '( ' + str(node).split('(')[0] + ' ' + '( ' + '('.join(str(node).split('(')[1:])
        for c in children:
            sequence += SBT(c)
        sequence += ') ' + str(node).split('(')[0]
    return sequence
        
        
def find_file(search_file, root):
    for subdir, dirs, files in os.walk(root):
        for file in files:
            if file == search_file:
                contents = ''
                file = subdir + '\\' + file
                try:
                    with open(file, 'r', encoding='utf-8') as fp:
                        contents = fp.read()
                        tree = javalang.parse.parse(contents)
                        sbt_string = SBT(tree)
                        return sbt_string
                except FileNotFoundError:
                    print('EXCEPTION')
                    return None
  
    
def main():
    filenames = ['./camel-1.6.csv', './lucene-2.4.csv', './ant-1.7.csv']
    roots = ['./apache-camel-1.6.0-src', './lucene-2.4.0-src', './apache-ant-1.7.0-src']
    sbt_data = {}
    count = 1
    for filename, root in zip(filenames, roots):
        data = pd.read_csv(filename)
        for index, row in data.iterrows():	
            search_file = row['name.1'].split('.')[-1].replace(' ', '')
            search_file += '.java'
            sbt_data[row['name.1']] = find_file(search_file, root)
            print(count)
            count += 1
        print('FILE {} DONE'.format(filename))
    with open('sbt_data_combined_minus_velocity.pickle', 'wb') as fp:
        pickle.dump(sbt_data, fp)
      

    
    
        
if __name__ == '__main__':
    main()
    
        
        
        
        
