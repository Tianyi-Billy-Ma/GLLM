




Dataset Statistics:

Number of tables: 18496 
Num of unique tables: 2100
Num of nodes: 363509
Num of hyperedges: 71000

Average number of cols/rows: 6.35/25.38
Max number of cols/rows: 25.00/753.00
Min number of cols/rows: 3.00/4.00

Memo:

Table too long issue: idx 844, 1171, 1285, 1316 




Performance:

Match Table + Llama 3: 
    f1: 56.82 acc: 66.87

RAG + Rephrased Questions 1360 + rest orignal Questions + Llama 3:
    f1: 14.54, acc: 13.59

RAG + Rephrased Questions 1360 + Llama 3:
    f1: 13.38 acc: 12.87

RAG + Original Question 1360 + Llama 3:
    f1: 11.51 acc: 10.85

RAG + Rephrased Questions All + Llama 3:
    f1: 15.34 acc: 14.68

RAG + Original Table Embedding + All Rephrased Questions + Llama 3: 
    f1: 30.24 acc: 29.28




Hypergraph Embeddings 

Title Acc: {1: 67.27, 3: 77.7, 5: 81.68, 10: 86.09}
Summary Acc: {1: 61.83, 3: 71.88, 5: 75.75, 10: 80.5}


Table Linearlize Embedding

Title Acc: {1: 67.05, 3: 77.61, 5: 81.68, 10: 85.95}
Summary Acc: {1: 60.86, 3: 71.23, 5: 75.01, 10: 79.86}





TODO: 
    Table + Summary 
    Linearlization Embedding (table only / table + title / table + summary / table + title + summary)
    Hypergraph table embeddings
    Hypergraph embeddings + summary embedding/ title embedding


**With Rephrased Questions**

md format:                          

{"1": 50.11, "3": 64.83, "5": 71.15, "10": 78.6}

md format + summary                 

{"1": 69.27, "3": 80.47, "5": 84.2, "10": 88.35}

md format + title                   

{"1": 67.96, "3": 79.38, "5": 83.16, "10": 87.24}

md format + summary + title         

{"1": 70.43, "3": 80.98, "5": 84.62, "10": 88.64}

dict format                         

{"1": 54.27, "3": 69.23, "5": 74.98, "10": 81.87}

dict format + summary               

{"1": 70.25, "3": 81.28, "5": 84.96, "10": 89.17}

dict format + title                 

{"1": 69.39, "3": 80.94, "5": 84.49, "10": 88.65}

dict format + summary + title       

{"1": 71.19, "3": 81.81, "5": 85.53, "10": 89.52}  ## Best 

html format                         

{"1": 54.07, "3": 69.28, "5": 75.05, "10": 81.87}

html format + summary               

{"1": 70.38, "3": 81.36, "5": 84.86, "10": 89.07}

html format + title                 

{"1": 69.68, "3": 80.82, "5": 84.55, "10": 88.59}

html format + summary + title       

{"1": 71.10, "3": 81.76, "5": 85.32, "10": 89.34}

sentence format                     

{"1": 53.26, "3": 67.44, "5": 73.21, "10": 80.8}

sentence format + summary           

{"1": 68.87, "3": 80.33, "5": 84.24, "10": 88.85}

sentence format + title             

{"1": 68.92, "3": 80.16, "5": 83.97, "10": 88.34}

sentence format + summary + title   

{"1": 70.75, "3": 81.38, "5": 85.06, "10": 89.32}

**With Original Questions**

md format:                          

{"1": 33.79, "3": 47.18, "5": 52.96, "10": 61.11}

md format + summary                 

{"1": 37.76, "3": 51.01, "5": 56.81, "10": 64.72}

md format + title                   

{"1": 35.16, "3": 48.52, "5": 54.41, "10": 62.35}

md format + summary + title         

{"1": 37.47, "3": 50.8, "5": 56.66, "10": 64.55}

dict format                         

{"1": 37.26, "3": 51.04, "5": 57.08, "10": 65.23}

dict format + summary               

{"1": 39.82, "3": 53.39, "5": 59.29, "10": 67.07}

dict format + title                 

{"1": 38.75, "3": 52.11, "5": 58.22, "10": 66.3}

dict format + summary + title       

{"1": 39.68, "3": 52.9, "5": 59.13, "10": 67.07}

html format                         

{"1": 36.92, "3": 50.39, "5": 56.92, "10": 64.89}

html format + summary               

{"1": 39.4, "3": 52.87, "5": 58.68, "10": 66.91}

html format + title                 

{"1": 38.28, "3": 51.85, "5": 57.76, "10": 65.79}

html format + summary + title       

{"1": 39.07, "3": 52.6, "5": 58.73, "10": 66.82}

sentence format                     

{"1": 37.52, "3": 50.74, "5": 56.88, "10": 65.14}

sentence format + summary           

{"1": 39.71, "3": 53.33, "5": 59.61, "10": 67.35}

sentence format + title             

{"1": 39.05, "3": 52.48, "5": 58.56, "10": 66.51}

sentence format + summary + title   

{"1": 39.98, "3": 53.48, "5": 59.5, "10": 67.31}
                     


**With Rephrased Questions + Hypergraph Embedding**

md + node embeddings:               

{"1": 42.40, "3": 57.03, "5": 63.69, "10": 72.84}

md + summary + node:                

{"1": 60.23, "3": 72.50, "5": 77.1, "10": 83.54}

md + summary + title + node:        

{"1": 61.34, "3": 73.15, "5": 77.54, "10": 84.06}

md + title + node:                  

{"1": 58.95, "3": 71.4, "5": 75.98, "10": 82.43}



Hypergraph Embeddings + dict + summary + title + Origianl Questons:

{"1": 45.82, "3": 56.13, "5": 64.28, "10": 69.26}

dict format + summary + title + Original Questions

{"1": 39.68, "3": 52.9, "5": 59.13, "10": 67.07}

Hypergraph Embeddings + dict + summary + title + Rephrased Questons:

{"1": 76.89, "3": 85.31, "5": 87.84, "10": 90.42}  ## Best Model 

dict format + summary + title + Rephrased Questions:   

{"1": 71.19, "3": 81.81, "5": 85.53, "10": 89.52}  ## Best Baseline 

10 epochs loss: 5~ =>2~