import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
              
class billu():

    def __init__(self, column):
        self.column=column

    def getColumn(self):
        print (self.column)

    def descriptive_stats(self):
        return (self.column.describe())

    def variance(self):
        return (self.column.std())

    def min_value(self):
        return(self.column.min())
    
    def max_value(self):
        return(self.column.max())

    def frequency(self):
        return(self.column.count())

    def distribution(self):
        plt.hist(self.column)
        plt.xlabel('Frequency',fontsize=18)
        return (plt.show())
    
    
class yelp(billu):
    def __init__(self,column):
        billu.__init__(self,column)

    def col_average(self):
        return(self.column.mean())

    def del_item(self):
        stack=list(self.column)
        removed=stack.pop()
        print(removed)
        

               
