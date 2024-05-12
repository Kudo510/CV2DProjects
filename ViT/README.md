conda activate 

## What got
test_loss += loss.detach().cpu().item() / len(test_loader) -so for calcualting losses - always detach and (convert to cpu) - cos losses will consume tons of memory otherwise
same for cal accuracy - detach also convert to  cpu
## what left 
the loss is incorrect -need to check the code again (see the training_loss.txt)
find better loggiing code , removing all old trainsing_loss  history before save the current one- currently all the old and new are saved (the old is not needed)
the accuracy is low- imporve it by adjusting hyperparameters, etc
