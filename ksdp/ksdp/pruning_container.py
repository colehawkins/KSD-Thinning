import torch
from . import ksd

class PruningContainer:
    def __init__(self,kernel_type,h_method,full_mat=True,*args,**kwargs):

        self.kernel_type = kernel_type
        self.h_method=h_method
        self.full_mat=full_mat
   
    @torch.no_grad()
    def best_index(self, candidate_points, candidate_gradients):
        #Given an array of new points and gradients, select the KSD-optimal point
      
        row_sums = []
        for candidate_point,candidate_gradient in zip(candidate_points,candidate_gradients):
            stacked_points = torch.cat([self.points,candidate_point.unsqueeze(0)])
            stacked_gradients= torch.cat([self.gradients,candidate_gradient.unsqueeze(0)])
            h = ksd._get_h(stacked_points,self.h_method) if self.kernel_type=='rbf' else None
            row_sums.append(ksd.get_K_row(samples=stacked_points,gradients=stacked_gradients,kernel_type=self.kernel_type,h=h).sum())

        return torch.stack(row_sums).argmin()


    @torch.no_grad()
    def add_point(self, point, gradient):
        #Add point and gradient to container and update K matrix
        
        try:
            assert hasattr(self,'points') and hasattr(self,'gradients') 
            self.points = torch.cat([self.points,point.unsqueeze(0)])
            self.gradients = torch.cat([self.gradients,gradient.unsqueeze(0)])
            self.update_K_info(method='add_row')

        except Exception as e:
            
            try:
                assert not hasattr(self,'points') and not hasattr(self,'gradients') 
            except:
                raise e
            #pruning container not initialized, this is the first point
            self.points = point.unsqueeze(0)
            self.gradients = gradient.unsqueeze(0)
            self.update_K_info(method="from_scratch")


    @torch.no_grad()
    def update_K_info(self,method='from_scratch',removed_row_index=None):
        if self.full_mat or method=='from_scratch':
            self.high_mem_K_update(method=method,removed_row_index=removed_row_index)
        else:
            self.low_mem_K_update(method=method,removed_row_index=removed_row_index)
    
    @torch.no_grad()
    def low_mem_K_update(self,method='from_scratch',removed_row_index=None):
        #update KSD kernel matrix 
        #only supports adding one row or total recompute 

        if self.h_method=="med":
            raise ValueError("Can't update individual rows when using median heuristic, not supported for low-mem")


        if method=='add_row':
            #if we update a row we should only have one new sample
            assert self.ksd2_contrib.shape[0]==self.points.shape[0]-1
            h = ksd._get_h(samples=self.points,h_method=self.h_method) if self.kernel_type=='rbf' else None
            new_row = ksd.get_K_row(samples=self.points,gradients=self.gradients,kernel_type=self.kernel_type,h=h)
         
            assert new_row.shape[0] == self.ksd2_contrib.shape[0]+1 
            #add new row contribution
            self.ksd2_contrib = self.ksd2_contrib+2.0*new_row[:-1]
            #add new row
            new_row_sum = new_row.sum()
            self.ksd2_contrib = torch.cat([self.ksd2_contrib,(2.0*new_row_sum-new_row[-1]).reshape(1)])
            
            self.row_sums += new_row[:-1]
            self.row_sums = torch.cat([self.row_sums,new_row_sum.reshape(1)])

        elif method=='remove_row':
            if removed_row_index is None:
                raise ValueError("To remove row, needs index value")
            #remove row corresponding to index
            h = ksd._get_h(samples=self.points,h_method=self.h_method) if self.kernel_type=='rbf' else None
            removed_row = ksd.get_K_row(samples=self.points,gradients=self.gradients,kernel_type=self.kernel_type,h=h,index=removed_row_index)
            
            removed_row = removed_row.squeeze()

            self.points = torch.cat([self.points[:removed_row_index],self.points[removed_row_index+1:]])
            self.gradients = torch.cat([self.gradients[:removed_row_index],self.gradients[removed_row_index+1:]])
          

            self.ksd2_contrib = self.ksd2_contrib-removed_row
            self.ksd2_contrib = torch.cat([self.ksd2_contrib[:removed_row_index],self.ksd2_contrib[removed_row_index+1:]]) 
            
            self.row_sums -= removed_row
            self.row_sums = torch.cat([self.row_sums[:removed_row_index],self.row_sums[removed_row_index+1:]])
            
        else:
            raise NotImplementedError("Method {} is not implemented for K matrix update".format(method))

    @torch.no_grad()
    def high_mem_K_update(self,method='from_scratch',removed_row_index=None):
        #update KSD kernel matrix 
        #only supports adding one row or total recompute 

        if self.h_method=="med":
            try:
                assert method=='from_scratch'
            except:
                raise ValueError("Can't update individual rows when using median heuristic")

        if method=='from_scratch':
            self.K_matrix = ksd.get_K_matrix(samples=self.points,gradients=self.gradients,kernel_type=self.kernel_type,h_method=self.h_method)
            row_sum = self.K_matrix.sum(dim=1)
            diag = torch.diag(self.K_matrix)

            self.row_sums = row_sum
            self.ksd2_contrib = 2.0*row_sum-diag
            if self.points.shape[0]==1:
                self.ksd2_contrib = self.ksd2_contrib.reshape(1)


        elif method=='add_row':
            #if we update a row we should only have one new sample
            assert self.K_matrix.shape[0]==self.points.shape[0]-1
            assert self.ksd2_contrib.shape[0]==self.points.shape[0]-1
            h = ksd._get_h(samples=self.points,h_method=self.h_method) if self.kernel_type=='rbf' else None
            new_row = ksd.get_K_row(samples=self.points,gradients=self.gradients,kernel_type=self.kernel_type,h=h)
         
            assert new_row.shape[0] == self.K_matrix.shape[0]+1 
            #add new row contribution
            self.ksd2_contrib = self.ksd2_contrib+2.0*new_row[:-1]
            #add new row
            new_row_sum = new_row.sum()
            self.ksd2_contrib = torch.cat([self.ksd2_contrib,(2.0*new_row_sum-new_row[-1]).reshape(1)])
            
            self.K_matrix = torch.cat([self.K_matrix,new_row[:-1].unsqueeze(0)])
            self.K_matrix = torch.cat([self.K_matrix,new_row.unsqueeze(1)],dim=1)

            self.row_sums += new_row[:-1]
            self.row_sums = torch.cat([self.row_sums,new_row_sum.reshape(1)])

        elif method=='remove_row':
            #remove row corresponding to index

            removed_row = self.K_matrix[removed_row_index].squeeze()
            self.points = torch.cat([self.points[:removed_row_index],self.points[removed_row_index+1:]])
            self.gradients = torch.cat([self.gradients[:removed_row_index],self.gradients[removed_row_index+1:]])
          
            #remove row 
            self.K_matrix = torch.cat([self.K_matrix[:removed_row_index],self.K_matrix[removed_row_index+1:]])
            #remove column
            self.K_matrix = torch.cat([self.K_matrix[:,:removed_row_index],self.K_matrix[:,removed_row_index+1:]],dim=1)

            self.ksd2_contrib = self.ksd2_contrib-removed_row
            self.ksd2_contrib = torch.cat([self.ksd2_contrib[:removed_row_index],self.ksd2_contrib[removed_row_index+1:]]) 
            
            self.row_sums -= removed_row
            self.row_sums = torch.cat([self.row_sums[:removed_row_index],self.row_sums[removed_row_index+1:]])
            
        else:
            raise NotImplementedError("Method {} is not implemented for K matrix update".format(method))

    @torch.no_grad()
    def get_ksd_squared(self):

        #if the K matrix is not current recompute 
        if self.points.shape[0]!=self.K_matrix.shape[0] and self.full_mat:
            print("Called self.get_ksd without current K_matrix, recomputing")
            self.update_K_info(method='from_scratch')

        n = self.points.shape[0]
       
        return self.row_sums.sum() / n**2


    @torch.no_grad()
    def prune_to_cutoff(self, cutoff, min_samples=None):
        

        pruned_samples = []
        
        if self.points.shape[0]<=min_samples:
            return pruned_samples
        """
        test_ksd_squared = self.K_matrix.sum()/(self.K_matrix.shape[0]**2)
        print("Row sum",self.row_sums)
        print("K MAT ",self.K_matrix)
        init_ksd_squared = self.get_ksd_squared()

        print("testing that quick ksd is ok")
        print(init_ksd_squared,test_ksd_squared)
        assert torch.allclose(test_ksd_squared,init_ksd_squared)
        """
        init_ksd_squared = self.get_ksd_squared()
        ksd_squared = init_ksd_squared
        #iteratively prune until cutoff is reached

        num_pruned = 0
       
        #equality permitted to avoid breaking before starting
        while ksd_squared <= init_ksd_squared+cutoff:
            if (self.points.shape[0]-1)<min_samples:
                return pruned_samples

            removal_ksd2_contrib,least_influential_point = torch.topk(self.ksd2_contrib,1,largest=True)

            num_points = self.points.shape[0]
            ksd_squared = ((num_points**2)*ksd_squared-removal_ksd2_contrib) / (num_points-1)**2
          
            #test if removing point exceeds cutoff
            if ksd_squared > init_ksd_squared+cutoff:
                return pruned_samples
            else:
                num_pruned+=1
                pruned_samples.append(self.points[least_influential_point])
                self.update_K_info(method='remove_row',removed_row_index=least_influential_point)

        return pruned_samples
