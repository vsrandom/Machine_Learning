import pandas as pd

reviews=pd.read_csv("ign.csv")
#print(reviews.head())
#print(reviews.shape)
'''
reviews=reviews.iloc[0:,1:]
print(reviews.shape)
print(reviews.head())

#print(reviews.loc[:5,["title","score"]])
print(reviews["score"])
'''

frame=pd.DataFrame([[1,2],["Loda","Lassan"]],columns=["col1","col2"],index=["row1","row2"])

print(frame.head())
