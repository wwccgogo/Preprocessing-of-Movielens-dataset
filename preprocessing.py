import numpy as np 
import pandas as pd 
import scipy.sparse as sp

import torch.utils.data as data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
import os
import pickle
import random
import re
import argparse
#arguments
parser = argparse.ArgumentParser()
parser.add_argument("--raw_dir",default='/home/wc/datasets/movielense/')
parser.add_argument("--dataset_name",default='ml-1m')
parser.add_argument("--process_dir",default='./dataset/pre_process/')
parser.add_argument("--same_space",default = True)
args = parser.parse_args()

raw_dir = args.raw_dir
datasetname = args.dataset_name
processdir = args.process_dir
user_movie_same_space = args.same_space

raw_data_dir = raw_dir + datasetname+"/"
#create output dir
if not os.path.exists(processdir):
    os.makedirs(dirs)

cat_name_set = set()
with open(raw_data_dir+"movies.dat",'r',encoding='latin-1')as rfile:
    for l in rfile.readlines():
        l = l.strip("\n")
        for c in l.split("::")[-1].split("|"):
            cat_name_set.add(c)
cat_names = list(cat_name_set)
cat_count_dict = {}
count_cat_dict = {}
for i in range(len(cat_names)):
    if cat_names[i] not in cat_count_dict:
        cat_count_dict[cat_names[i]] = i+1
        count_cat_dict[i+1] = cat_names[i]
# processdir = "/home/wc/workspace/recommend/movielens_reconstruct/dataset/pre_process"
#/10m/ml-10M100K
with open(processdir+'cat_count_dict.pickle', 'wb') as handle:
    pickle.dump(cat_count_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(processdir+'count_cat_dict.pickle', 'wb') as handle:
    pickle.dump(count_cat_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
punc = '[,.!\'()\"]'
movie_word_count_dict = {}
movie_count_word_dict = {}
count = 1

with open (raw_data_dir+"movies.dat","r",encoding='latin-1')as rfile:
    lines = rfile.readlines();
    for l in lines:
        attrs = l.split("::")
        mname_terms = attrs[1].split(" ")[:-1]
        for t in mname_terms:
        
            t = re.sub(punc,"",t)
            if t in movie_word_count_dict:
                continue
            else:
                movie_word_count_dict[t] = count
                movie_count_word_dict[count] = t
                count += 1
with open(processdir+'movie_word_count_dict.pickle', 'wb') as handle:
    pickle.dump(movie_word_count_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(processdir+'movie_count_word_dict.pickle', 'wb') as handle:
    pickle.dump(movie_count_word_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#movie features generation
with open(processdir+"movies_feat.csv","w") as wfile:
    headline = "mid,"
    for i in range(5):
        headline = headline+"term"+str(i+1)+","
    headline = headline+ "year"
    for i in cat_names:
        headline = headline+","+i
    print(headline)
    wfile.write(headline+"\n")
    with open (raw_data_dir+"movies.dat","r",encoding='latin-1')as rfile:
        lines = rfile.readlines();
        for l in lines:
            l = l.strip("\n")
            newline = ""
            attrs = l.split("::")
            mid = attrs[0]
            mname_terms = attrs[1].split(" ")[:-1]
            mname_str = ""
            newline = newline+attrs[0]+","

            mname_count = 0 
            for t in mname_terms:
                if mname_count < 5:
                    
                    t = re.sub(punc,"",t)
                    newline = newline+str(movie_word_count_dict[t])+","
                mname_count += 1
            if len(mname_terms)<5:
                for c in range(5-len(mname_terms)):
                    newline = newline+"-1,"
            year = attrs[1].split(" ")[-1][1:-1]
            try:
                year = int(year)-1919+1 
            except:
                year = int(attrs[1][-5:-1])-1919+1
            newline = newline+str(year)
            
            #category
            cates = attrs[2].split("|")
            cate_array = np.zeros([len(cat_names)])
            for c in cates:
                ind = cat_count_dict[c]-1
                cate_array[ind] = 1
            
            for c_ in cate_array:
                newline = newline + ","+str(int(c_))
            wfile.write(newline+"\n")
gender_id_dict = {"F":1,"M":2}
id_gender_dict = {1:"F",2:"M"}

age_id_dict = {}
id_age_dict = {}

age_list = [1,18,25,35,45,50,56]
age_count = 1
for a in age_list:
    if a not in age_id_dict:
        age_id_dict[a] = age_count
        id_age_dict[age_count] = a
        age_count +=1
        
with open(processdir+'age_id_dict.pickle', 'wb') as handle:
    pickle.dump(age_id_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(processdir+'id_age_dict.pickle', 'wb') as handle:
    pickle.dump(id_age_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

post_id_dict = {}
id_post_dict = {}

with open (raw_data_dir+"users.dat","r",encoding='latin-1')as rfile:
    lines = rfile.readlines();
    post_count = 1
    for l in lines:
        l = l.strip("\n")
        splitted = l.split("::")
        postcode = splitted[4]
        if postcode not in post_id_dict:
            post_id_dict[postcode] = post_count
            id_post_dict[post_count] = postcode
            post_count += 1
            
with open(processdir+'post_id_dict.pickle', 'wb') as handle:
    pickle.dump(post_id_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(processdir+'id_post_dict.pickle', 'wb') as handle:
    pickle.dump(id_post_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(processdir+"users_feat.csv","w") as wfile:   
    with open (raw_data_dir+"users.dat","r",encoding='latin-1')as rfile:
        newline = "uid,gender_id,age_id,occ_id,post_id\n"
        wfile.write(newline)
        lines = rfile.readlines();
        for l in lines:
            l = l.strip("\n")
#             print(l)
            splitted = l.split("::")
            uid = splitted[0]
            gender = gender_id_dict[splitted[1]]
            age = age_id_dict[int(splitted[2])]
            occupation_id = int(splitted[3])+1 
            postcode_id = post_id_dict[splitted[4]]
            newline = uid+","+str(gender)+","+str(age)+","+str(occupation_id)+","+str(postcode_id)+"\n"
            wfile.write(newline)
#feature extraction finished           
with open(processdir+"rating.csv","w") as wfile:  
    with open (raw_data_dir+"ratings.dat","r",encoding='latin-1')as rfile:
        lines = rfile.readlines();
        wfile.write("uid,mid,rating,time\n")
        for l in lines:
            l = l.strip("\n")
            splitted = l.split("::")
            uid = splitted[0]
            mid = splitted[1]
            rating = splitted[2]
            time = splitted[3]
            wfile.write(uid+","+mid+","+rating+","+time+'\n')
            
#data cleasing
rate = pd.read_csv(processdir+"rating.csv")
user_feat = pd.read_csv(processdir+"users_feat.csv")
movie_feat = pd.read_csv(processdir+"movies_feat.csv")

#mid 与m_count的转换
m_id_2_count = {}
m_count_2_id = {}
m_count =0;
for m in np.array(movie_feat['mid']):
    if m not in m_id_2_count:
        m_id_2_count[m] = m_count;
        m_count_2_id[m_count] = m;
        m_count +=1;
        
#rearrange mid
movie_feat.mid = movie_feat.mid.apply(lambda x: m_id_2_count[x])


u_id_2_count = {}
u_count_2_id = {}
u_count =0;
for u in np.array(user_feat['uid']):
    if u not in u_id_2_count:
        u_id_2_count[u] = u_count;
        u_count_2_id[u_count] = u;
        u_count +=1;
user_feat.uid = user_feat.uid.apply(lambda x: u_id_2_count[x])

rate.uid = rate.uid.apply(lambda x: u_id_2_count[x])
rate.mid = rate.mid.apply(lambda x: m_id_2_count[x])
rate_user_set = set(np.array(rate.uid))
rate_movie_set = set(np.array(rate.mid))

m_count_filter_based_on_rate = {}
m_count_filter_based_on_rate_rev = {}
counter = 0
print(np.max(list(rate.mid)))#3882
for m in set(rate['mid']):
    if m not in m_count_filter_based_on_rate:
        m_count_filter_based_on_rate[m] = counter;
        m_count_filter_based_on_rate_rev[counter ] = m;
        counter +=1;
rate.mid = rate.mid.apply(lambda x: m_count_filter_based_on_rate[x])
print(np.max(list(rate.mid)))#3705
#there are discontinuous mid in rating.csv
def apply_filter_on_mid (mid):
    if mid in m_count_filter_based_on_rate:
        return m_count_filter_based_on_rate[mid]
    else:
        return -1
movie_feat.mid = movie_feat.mid.apply(lambda x:apply_filter_on_mid(x))
movie_feat = movie_feat[movie_feat['mid'] != -1]

rate.to_csv(processdir+"rate_new.csv",index=None)
movie_feat.to_csv(processdir+"movie_feat_new.csv",index=None)
user_feat.to_csv(processdir+"user_feat_new.csv",index=None)

# rate = pd.read_csv(processdir+"rate_new.csv")
# user_feat = pd.read_csv(processdir+"user_feat_new.csv")
# movie_feat = pd.read_csv(processdir+"movie_feat_new.csv")

movie_num = movie_feat['mid'].max()+1
user_num = user_feat['uid'].max()+1

def movie_new_id(movieid):
    if user_movie_same_space:
        return int(movieid)+user_num
    
    return movieid

rate_train = []
rate_test = []

for i in range(user_num):
    tar = rate[rate['uid']==i][['mid','time']]
    target_rate = list(tar.sort_values(by='time')['mid'])

    rate_len = len(target_rate)
    #select the last interation
    rand = rate_len-1
    rate_test.append([i,target_rate[rand]])
    
    for j in target_rate:
        if j !=target_rate[rand]:
            rate_train.append([i,j])
            
test_num_ng = 99 #nums for negative sampling in test set
rate_test_np = np.array(rate_test)
rate_train_np = np.array(rate_train)

ng_sampled_rate_test = []
for r in rate_test_np:
    user = r[0]
    
    pos_mids = list(rate[rate['uid'] == user].mid)
    nag_mids = []
    for i in range(test_num_ng):
        random_movie_id = random.randint(0,movie_num-1)
        
        while random_movie_id in pos_mids or random_movie_id in nag_mids:
            random_movie_id = random.randint(0,movie_num-1)
        nag_mids.append(random_movie_id) 
    for n in nag_mids:
        ng_sampled_rate_test.append([user,n])

with open(processdir+"test_negative.csv",'w') as wfile:
    with open(processdir+"test.csv",'w')as wfile2:
        for i in range(6040):
            line = ""
            
            line = line+"("+str(rate_test[i][0])+","+str(movie_new_id(rate_test[i][1]))+")"
            
            for j in ng_sampled_rate_test[i*99:(i+1)*99]:
                line = line+"\t" + str(movie_new_id(j[1]))
            line = line+"\n"
            wfile.write(line)
            
            line2 = str(rate_test[i][0])+","+str(movie_new_id(rate_test[i][1]))+"\n"
            wfile2.write(line2)
with open(processdir+"train.csv",'w') as wfile:
    for i in rate_train:
        wfile.write(str(i[0])+","+str(movie_new_id(str(i[1])))+"\n")