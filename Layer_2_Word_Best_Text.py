 =   environ TF TF']='TF']='MIN']='MIN']='2'
import logging
import time import time




import
logging as
np
importlogging.getLogger('tensorflow').setLevel(logging.WARNING


class
CharRNN

,

 








 



 training_

 



 
_
vocab                 
_

 
_
_
+ 
_

 


                 
_

 

 
_
_
+ 






shape 
_
_
      



=

 

      



=

 

      = 
 


        


_= 
 
       
 
_= 
       


_=




      



=

 

      





= tf 



      


 = tf 

      



=

 

      

=


      

= 
 
     



= 
 

      = 

=
tf 
        
_
_= 
 

        
 

=




      
        


_= 
 
_      
 

=

self










                         = tf 
_= tfhidden. _* tf 



= tf 
+                          size_= tfhidden. _* tf 
+                         step  _= tf 
+ tf tf tf 
_=                         step  _= tf 
_= tf 
       


 = tf 







                                       size_ _ _= 
_
_
                                       size_
      update
= tf 

self
int64




                                 size_ _ _= 
_
_
                                    size_
      .tf.model
==.tf



   


_

=
tf.contrib







   =


= 
lstm':
     

_

=
tf.contrib







   =


= 
lstm':
     

_

=
tf.contrib





 

   =

 
     = 

= 
lstm':
     

_

bias
 =
cell





cell
   cell
state
state
 =
cell
      0 0 fn

          


_
reuse 








,
   

     

       
 
 
_      = 
 
 self.
_
_
_
        



=






           
_
_
reuse 
_
_
_


,
   

      params
 
       

higher.
_
_
       = 




 

== tf 
        = 
 
_
_



          
          


_








                 = 
 
 
       

= 
 








 def    







'):
     

 


 = tf


 _with_


_
)

 
_
) def      




=









_
          















_          




          


_
_








              = 

embedding_layer'):
     
  
 
_= tf reshape
    

   
_= 
 
_
_self           embedding
 [self.
 




embedding.embedding,
       


         

= 
 


.
_
.
)
)

 
)
)
) def     





















_
        





tf




= tf 
          = 
 
_
_
.

 
 tf 




      
=



_
'):
    


 

= tf 









                         = 
+ tf 
_self. _ 
 




_
)
)
)
* 
_
             

 

= 
 
_







       







        
 
_




       



=

 

       = 


_
'):
     

 

= tf 







concat


 



 
_tf 
)
       = 


_
'):
    


 

= tf 







concat


 


)

 
)     
  
  = 



global_ 
 sm






     w
=_tf_w


softmax
b


vocab




_size_        
logits
=_





softmax
b


vocab



     

_size 
 

size_










 1 

        
_=



_






      








   


 =
tf.










_


          





 




        



=









 def    





_
'):
     

 
 tf 











        update_loss_= tf 

tf
int64





_
)
        
      


_
_= tf 


.


)
)
. _
                                           size_self. _                                            size_ _ _
        update
loss
monitor
=
tf 




64
)
)
. _
)1 
                                                                size_ _                                             size_self.1 1                                              size_ _ _
        tf.control











          



= tf 
_
_= tf           
_= 
 
_
.
_


        




=

average

 
        summary_loss_= "perplexity"
           summary_loss_= 
 summary_summary_ppl.


)

 
_
)
        

=














 
_
      


= 












shape 


                                        size_ _
           

= tf 






'):




      


                              size_ _ _self. _        



= 
 






      




        = 
 




        

 
 
 









 _
. )
)

 
                                          size_ _ _         = 
 










        




=












 
                                                  size_ _ _           
 

self


 

 
_






 
_
                  
_

 
_
+ 
_
_

 



 
_
_
_
      

=




=
tfself






1 




      



= (self.batch





.




*

:


          += 
 

     = 
 
 
          epoch_size: %d', epoch_






 



_

















 



_















         ('















 


  _
 training

        

= 
 



      
        

= 
 
_
_

      = 
 


.




      









     

=





      = 
 
 self.
_
 
 
_


        =








       = 
 

self.
_



       = 
 

self

_




       = 
 





= 


_

 


               
_

 
_

 
_
_
     

 


=








= 
int 



 
                     
_
_ + 
         = 
 





 


        







tf



and 



 1 tf                = 
 


.
_
        =verbose 
 
 

 
step






 == 
):
    








 


, perplexity:
%.

, speed:
%.3f

",
      

              
 = 1  + tf 
_* tf 
_= tf 
vocab 
vocab 
                      time = 1  + tf 
_
_* tf 



=                       time 
_ - tf 
_
       

 
3



:
%.0f words:





    



          
ppl
 = 1  + tf 
_
_
 tf 



=                   verbose 
_
- tf 
_
      = 

 






     = 
















 


_
                   
_
_

 
_
_
+ 




       












 
     =





 
 tf tf shape, _ 
 tf 
        = 
 
.
_
        = 
 
 
_


          = 
 
_
_
_
.

 


)
          = 
 
_self.
_


                              size_ _ _= 
                               size_ _ + 
        = 
 













 
_
)
      
   
    

= 
 self,
_





       =.np array






np

tf



        





     = 
 
 
,
        

 
 
 
_






                                     size_                                     size_ _ _= 
                                     size_ _ + 
        
 = 





-
np
max(

 / temperature

       .
 
 
_= 
 
_self.
_
         = 


          = 
 


.
_
        
          = 
 




.
_



 
+ 


)
)
        



,











        








      =
(seq)
       class id object):
      = 
_


 

 
_

 



 
_
                   
_
_

 
_
_
        

=



       


_= 
 
.         
_

=

 

        
 

=




        
 

=




        
 



=

 



        





=






               = 
 
_
_= 
 

         
 =

 
 
 
 tf tf tf tf 
. _         
_
 = 
 





            = 


self,
        = 
 
_self












_
        






.

 _
          
_

 
 
_
.
_


)
)

 
_
)
)
          
 



 tfself.
_



 1 
+ self._text_
_


    ._batch

     ("%.0 
,
        = 
 
_



        = 
 
 
,
_ _
          
_self






      ._last


=
np np

    
   = 
  id string_index,
index
















 






    


 
 
      = 
 
seq, 
 tf tf tf shape. 
 
_ )
. 
 
_
)
    



 characters object,     = 
_

,

 tf tf tf 





 
   create 


,







   

   


=







    =

      











     0


 id 










dict












     
 id 




,

 






 






,

 





 tf tf tf 
   create 







lst,






 
    


, index 
      = 
 




_

       

 





 tf 

    

     = 
 


_






       






 
                         = 
+ tf tf self. 
        = tf shape.       = 
 
 
        = 
 


      
        = 
 






    
 
          id 




,

 






  



, index 
      = 
 

self




,





 

    
   
  = 
 


_

_








                         = tf tf 
      = 
 


       = 
 
 

       = 
 


      
        = 
 


    =