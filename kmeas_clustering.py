import os 

import pandas as pd
import cv2
from sklearn.cluster import KMeans

df = pd.read_csv('/home/alvaro/Documentos/video2tfrecord/results/sign_1/df_res.csv')

X = df.iloc[:, :13].values
y = df.iloc[:, 13].values

kmeans = KMeans(n_clusters=6, max_iter=500, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)

df['cluster_number'] = pred_y

for i in range(len(df)):
    img_path = df.iloc[i, 14]
    img = cv2.imread(img_path)
    
    cluster_path = '/home/alvaro/Documentos/video2tfrecord/results/sign_1/cluster_'+str(df.iloc[i, 15])
    if not os.path.exists(cluster_path):
        os.mkdir(cluster_path)

    cv2.imwrite(cluster_path+'/'+img_path.split('/')[-1], img)

df.to_csv('/home/alvaro/Documentos/video2tfrecord/results/sign_1/df_cluster.csv', index=False)