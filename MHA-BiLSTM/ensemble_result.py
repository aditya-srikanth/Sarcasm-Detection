import numpy as np
import pandas as pd
import csv
import re
import os

## path variables ##
rootDir="Your_Root_Dir"
output="output/"
att_bal_output_file_path=rootDir+output+"Multi_Att_Best_UnBalanced_output.tsv" # here Multi_Att_Best_UnBalanced_output.tsv is output file of multi attention model
svm_output_file_path=rootDir+output+"SVM_Bal_Unbal.xlsx"                       # here MSVM_Bal_Unbal.xlsx is output file of SVM model

df_att_bal=pd.read_csv(att_bal_output_file_path, sep='\t')
df_svm_bal=pd.read_excel(svm_output_file_path, sheet_name='Balanced_Test')
result=pd.merge(df_svm_bal,df_att_bal,on='id') 
df_final=result[['id','actual_label','Feature_Based Pred','predicted_label']]
df_final_new=df_final.rename(columns={"Feature_Based Pred": "SVM_Pred", "predicted_label": "Attn_Pred"})


tp=tn=fn=fp=0
#y=76000
metrics=dict()
for i in df_final.id:
  y=i
  if y <42670:
    
    actual_label=df_final['actual_label'][y]
    attn_pred=df_final['predicted_label'][y]
    svm_pred=df_final['Feature_Based Pred'][y]
    predicted_label=0
    if  attn_pred ==1 or svm_pred==1 :
      predicted_label=1
    if (predicted_label==1 and actual_label==1):
      tp += 1
    if (predicted_label==0 and actual_label==0):
      tn += 1
    if (predicted_label==0 and actual_label==1):
      fn += 1
    if (predicted_label==1 and actual_label==0):
      fp += 1
print("tp,tn,fn,fp is : ",tp,tn,fn,fp)
if (tp>0 ):
      
  metrics["precision"]=(tp/(tp+fp))
  metrics["recall"]=(tp/(tp+fn))
  metrics["f1_score"]=(2*metrics["precision"]*metrics["recall"])/(metrics["precision"]+metrics["recall"])
  print("F1 score is ",metrics["f1_score"] )
  print("Ensembled Results are ",metrics )
  
