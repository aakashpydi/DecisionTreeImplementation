---
## Decision Tree Implementation From Scratch
#### Aakash Pydi
---
### Usage Instructions
---

The python script takes the following three arguments.
1. trainingFile --> the path to the training set
2. testFile --> the path to the test set
3. model --> the model to use. Three options
  * vanilla --> for the full decision tree. For this model provide a fourth argument, a number indicating the training set percentage to use
  * depth --> for the decision tree with static depth. For this model also provide, a fourth argument indicating the training set percentage to use, a fifth argument indicating the validation set percentage to use and a sixth argument indicating the max-depth
  * prune --> for the decision tree with post-pruning. For this model, also provide, a fourth argument indicating the training set percentage to use, and a fifth argument indicating the validation set percentage to use.   

---
### Analysis
---
#### Vanilla Model
A binary decision tree with no pruning using the ID3 algorithm.

![](/images/vanilla_node_count.png)

![](/images/vanilla_accuracies.png)

---

#### Depth Limited Model
A binary decision tree with a given maximum depth.

![](/images/depth_node_count.png)

![](/images/depth_accuracies.png)

![](/images/depth_optimal_depth.png)

---

#### Decision Tree with Post Pruning
A binary decision tree with post pruning.

![](/images/prune_node_count.png)

![](/images/prune_accuracies.png)

---
