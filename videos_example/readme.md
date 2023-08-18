## Example of skeleton sample 
we provide here some sample of video skeleton. In best model video we give the example with preprocessing and in videos intermediate model we give example with some other preprocessing and without any preprocessing.  as you can see, our preprocessing give right now the best results we could have on this dataset, at the beginning the model couldn't even predict the right position of the skeleton.

NB: Of course, you can use your own model prediction if you train the model ( FEDFormer can be easily trained).

## Deep analysis of best model video
We give here some insight of the best model video. As a reminder, the first second is the input label and is supposed to be perfectly fit by the model and second second is the true label. the Blue skeleton is the ground truth and the red is the output of the model.


- Example 1 : the model fit with the predicted skeleton but cannot follow the hand's movement.
- Example 2 and 3 : as the sample ends at the end of the label, the models fits perfectly with the ground truth because he repeats his last frame
- Example 4: this one is a bit subtle, but  we can see that at the end of the input label, the skeleton predicted that the hand would go higher but instead stop his movement.
- Example 5: it reveals one major issue of this dataset for this task: the model doesn't move at all on our interval:( 


## Analysis of intermediate model with preprocess
 the models fits perfectly with the input label. however the model is more instable overall and can't learn the label.

 ## Analysis of model without preprocess
 as expected this model is the worst one , the model is constant and cannot learn the movement of the skeleton. He can't learn his position in space either.