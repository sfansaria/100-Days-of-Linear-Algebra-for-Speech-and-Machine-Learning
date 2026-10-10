'''
Fisher information Matrix-
 When a speech model is trained, changing a weight by a tiny increment of +0.01 can
 have wildly different real-world impacts depending on where that weight sits.

 - For one parameter, a +0.01 change might barely shift the model's output at all.
 - For another parameter, a +0.01 change might completely scramble the model's confidence, 
 causing it to misclassify vowels.

Stancalculus optimization (like AdamW or SGD) is completely blind to this. It assumes all
directions on the weight grid are flat and equal.

The Fisher Information Matrix (FIM) is a specialized cousin of the Hessian. Instead of 
measuring the curvature of an arbitrary loss function, it measures the local curvature 
of the probability distribution space itself. 

Mathematically, it is the expected value of the outer product of the log-likelihood gradients:
                F = E[delta_theta log P(x|theta) delta_theta log P(x|theta)_T]

Information Theory proves that the Fisher Information Matrix is exactly equal to the Hessian of the KL Divergence.
It maps out the local metric tensor of how fast information is lost or distorted when the weight's of the model is tweaked.


'''