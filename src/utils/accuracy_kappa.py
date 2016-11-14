import os, sys
import util
from skll import metrics

# Results from the CNN trained for referral (o_O)
lines_p = open("data/sub__2015-10-01 00:18:38.csv",'rb').readlines()
lines_p = open("data/sub__2015-10-15 16:01:06.csv",'rb').readlines() #per_patient

# Results from the CNN trained for severity and converted to referral (Jeffrey De Fauw)
#lines_p = open("data/2015_07_27_182759_25_log_mean_rank.csv",'rb').readlines()

lines_t = open("data/retinopathy_solutionReferral.csv",'rb').readlines()

predict = {}
true = {}

for l in lines_p[1:]:
	img, label = l.split(',')
	predict[img] = int(label.strip())
	
for l in lines_t[1:]:
	img, label = l.split(',')
	true[img] = int(label.strip())
	
p = []
t = []
for k in predict.keys():
	p.append(predict[k])
	t.append(true[k])	

#print true
#print true[:100]
print metrics.kappa(t, p)

a = 0
for i in range(len(p)):
	if p[i] == t[i]:
		a += 1
		
print a/float(len(p))
