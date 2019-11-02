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
        
        
def find_file(search_file):
    root = './apache-camel-1.6.0-src'
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
    data = pd.read_csv('./camel-1.6.csv')
    sbt_data = {}
    for index, row in data.iterrows():	
        search_file = row['name.1'].split('.')[-1].replace(' ', '')
        search_file += '.java'
        sbt_data[row['name.1']] = find_file(search_file)
        print(index)
    with open('sbt_data_camel.pickle', 'wb') as fp:
        pickle.dump(sbt_data, fp)
      

    
    
        
if __name__ == '__main__':
    main()
    
        
        
        
        
