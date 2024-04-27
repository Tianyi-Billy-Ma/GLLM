




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