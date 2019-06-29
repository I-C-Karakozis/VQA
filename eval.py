import json

from tools.eval_utils import VQAEval
from tools.log_utils import print_border
from tools.vqa_utils import VQA

# set up file names and paths
annFile   ='annotations/dev_answers.json'
quesFile  ='annotations/dev_questions.json'
fileTypes = ['accuracy', 'evalQA', 'evalQuesType', 'evalAnsType'] 

[accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = ['results/train_%s.json'%(fileType) for fileType in fileTypes]  
resFile = 'results/vqa.json'

# create vqa object and vqaRes object
vqa = VQA(annFile, quesFile)
vqaRes = vqa.loadRes(resFile, quesFile)

# evaluate results
vqaEval = VQAEval(vqa, vqaRes, n=2)
vqaEval.evaluate() 

# print accuracies
print("Overall Accuracy is: %.02f" %(vqaEval.accuracy['overall']))
print_border()
print("Per Question Type Accuracy is the following:")
for quesType in vqaEval.accuracy['perQuestionType']:
	print("%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType]))
print_border()
print("Per Answer Type Accuracy is the following:")
for ansType in vqaEval.accuracy['perAnswerType']:
	print("%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType]))
print_border()
