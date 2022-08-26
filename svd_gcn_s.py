import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import time
import gc


#gowalla
#user,item=29858,40981
#yelp
user,item=25677,25815
#ml-1m
#user,item=6040,3952

result=[]
dataset='./yelp'


#瀵煎叆鏁版嵁
df_train=pd.read_csv(dataset+ r'/train_sparse.csv')
df_test=pd.read_csv(dataset+ r'/test_sparse.csv')

#load the train/test data
#load the data
train_samples=0
#train_data=[[] for i in range(user)]
test_data=[[] for i in range(user)]
for row in df_train.itertuples():
	#train_data[row[1]].append(row[2])
	train_samples+=1
for row in df_test.itertuples():
	test_data[row[1]].append(row[2])
##########################################
#interaction matrix
rate_matrix=torch.Tensor(np.load(dataset+ r'/rate_sparse.npy')).cuda()


class SVD_GCN(nn.Module):
	def __init__(self, user_size, item_size, beta=4.0, req_vec=20):
		super(SVD_GCN, self).__init__()

		self.beta=beta
		self.user_size=user_size
		self.item_size=item_size

		svd_filter=self.weight_func(torch.Tensor(np.load(dataset+ r'/svd_value.npy')[:req_vec]).cuda())
		self.user_vector=(torch.Tensor(np.load(dataset+ r'/svd_u.npy')[:,:req_vec])).cuda()*svd_filter
		self.item_vector=(torch.Tensor(np.load(dataset+ r'/svd_v.npy')[:,:req_vec])).cuda()*svd_filter

	def weight_func(self,sig):
		return torch.exp(self.beta*sig)



	def predict(self):


		return (self.user_vector.mm(self.item_vector.t())).sigmoid()-rate_matrix*1000



	def test(self):
		#calculate idcg@k(k={1,...,20})
		def cal_idcg(k=20):
			idcg_set=[0]
			scores=0.0
			for i in range(1,k+1):
				scores+=1/np.log2(1+i)
				idcg_set.append(scores)

			return idcg_set

		def cal_score(topn,now_user,trunc=20):
			dcg10,dcg20,hit10,hit20=0.0,0.0,0.0,0.0
			for k in range(trunc):
				max_item=topn[k]
				if test_data[now_user].count(max_item)!=0:
					if k<=10:
						dcg10+=1/np.log2(2+k)
						hit10+=1
					dcg20+=1/np.log2(2+k)
					hit20+=1

			return dcg10,dcg20,hit10,hit20



		#accuracy on test data
		ndcg10,ndcg20,recall10,recall20=0.0,0.0,0.0,0.0
		predict=self.predict()

		idcg_set=cal_idcg()
		for now_user in range(user):
			test_lens=len(test_data[now_user])

			#number of test items truncated at k
			all10=10 if(test_lens>10) else test_lens
			all20=20 if(test_lens>20) else test_lens
		
			#calculate dcg
			topn=predict[now_user].topk(20)[1]

			dcg10,dcg20,hit10,hit20=cal_score(topn,now_user)


			ndcg10+=(dcg10/idcg_set[all10])
			ndcg20+=(dcg20/idcg_set[all20])
			recall10+=(hit10/all10)
			recall20+=(hit20/all20)			

		ndcg10,ndcg20,recall10,recall20=round(ndcg10/user,4),round(ndcg20/user,4),round(recall10/user,4),round(recall20/user,4)
		print(ndcg10,ndcg20,recall10,recall20)

		result.append([ndcg10,ndcg20,recall10,recall20])



#Model training and test

model = SVD_GCN(user, item)

model.test()




output=pd.DataFrame(result)
output.to_csv(r'./svd_gcn_s.csv')

