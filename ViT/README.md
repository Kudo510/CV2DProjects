conda activate 

## What got
test_loss += loss.detach().cpu().item() / len(test_loader) -so for calcualting losses - always detach and (convert to cpu) - cos losses will consume tons of memory otherwise
same for cal accuracy - detach also convert to  cpu