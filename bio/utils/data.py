import os
import functools
import numpy as np
import pandas as pd

BASE_DIR=os.path.join('/data','stein_thinning')

def load_goodwin(sample_generation,smoke_test=False):

    setting_dir ='goodwin'
    nrows = 1000 if smoke_test else None

    samples = pd.io.parsers.read_csv(os.path.join(BASE_DIR,setting_dir,'{}_samples.csv'.format(sample_generation)),dtype=np.float32, nrows=nrows).to_numpy()
    grads = pd.io.parsers.read_csv(os.path.join(BASE_DIR,setting_dir,'{}_grads.csv'.format(sample_generation)),dtype=np.float32, nrows=nrows).to_numpy()

    return samples, grads 

def get_min_norm_index(z,x):
    return np.argmin(np.linalg.norm(x-z,axis=1))

def load_cardiac(sample_generation,smoke_test=False):


    if sample_generation=='original':
        raise NotImplementedError('No data yet')
        setting_dir ='cardiac/..../seed_1'

    elif sample_generation=='tempered':

        setting_dir ='cardiac/Tempered_posterior/seed_1'

    
    nrows = 5001 if smoke_test else 3999998
        
    uncompressed_samples = pd.read_csv(os.path.join(BASE_DIR,setting_dir,'processed_THETA_seed_1_temp_8.csv'),dtype=np.float32,nrows=nrows).to_numpy()
    
    grad_path=os.path.join(BASE_DIR,setting_dir,'processed_GRAD_seed_1_temp_8.csv')
    
    try:
        uncompressed_grads = pd.read_csv(grad_path,dtype=np.float32,nrows=nrows,sep=' ').to_numpy()
        print("Loading processed grads succeeded, shape {}".format(uncompressed_grads.shape))
        if uncompressed_grads.shape[0]!=uncompressed_samples.shape[0]:
            raise ValueError("Shapes {} {} don't match, reprocessing".format(uncompressed_grads.shape[0],uncompressed_samples.shape[0]))

    except Exception as e:
        print(e)
        print("No uncompressed grads found, decompressing,\n takes 12hrs on 10 cores for full decompression")

        samples_name = 'processed_THETA_unique_seed_1_temp_8.csv'
        grads_name = 'processed_GRAD_unique_seed_1_temp_8.csv'
        

        compressed_samples = pd.read_csv(os.path.join(BASE_DIR,setting_dir, samples_name),dtype=np.float32).to_numpy()
        compressed_grads = pd.read_csv(os.path.join(BASE_DIR,setting_dir,grads_name),dtype=np.float32).to_numpy()
               
        print("Done loading")

        import multiprocessing
        with multiprocessing.Pool(int(multiprocessing.cpu_count()/2.0)) as p:
            f = functools.partial(get_min_norm_index, x=compressed_samples)
            sample_idx = list(p.map(f, uncompressed_samples))


        uncompressed_grads = compressed_grads[sample_idx] 
        
      
        print("Saving",grad_path)
        np.savetxt(grad_path,uncompressed_grads)
        """
        tmp_df = pd.DataFrame(uncompressed_grads)
        print(tmp_df.shape)
       
        
        sample_to_grad = {tuple(sample):grad for sample,grad in zip(compressed_samples,compressed_grads)}
    
        uncompressed_grads = np.array(list(map(lambda x:sample_to_grad[tuple(x)],uncompressed_samples)))
        """


    print(uncompressed_samples.shape)
    print(uncompressed_grads.shape)

    return uncompressed_samples, uncompressed_grads 
    

def get_samples_and_grads(problem,sample_generation,smoke_test=False):

    if problem=='goodwin':
        samples, grads = load_goodwin(sample_generation=sample_generation,smoke_test=smoke_test)
    elif problem=='cardiac':
        samples, grads = load_cardiac(sample_generation=sample_generation,smoke_test=smoke_test)

    return samples,grads

